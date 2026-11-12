"""Visión por computadora: elegir los fotogramas útiles y leer el texto en pantalla.

Selección inteligente de fotogramas (deep learning con CLIP):
  1. Se sacan muchos fotogramas candidatos (hasta ~240, varios por segundo en videos cortos).
  2. CLIP convierte cada imagen en un vector (embedding) que resume lo que se ve.
     Dos fotogramas de la misma escena tienen vectores muy parecidos (similitud coseno ~1).
  3. Se agrupan los fotogramas consecutivos parecidos en escenas y de cada escena se queda
     el más nítido (menos borroso).
  4. Si siguen siendo demasiados, se eligen los más distintos entre sí (k-center greedy),
     para cubrir todos los momentos del video sin repetir.

OCR (EasyOCR, redes neuronales CRAFT + CRNN): lee el texto escrito en los fotogramas elegidos
("200 g de harina", "180 °C") y elimina las repeticiones entre fotogramas.
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import threading
from pathlib import Path

import numpy as np

CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32")
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "240"))
SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "0.85"))  # similitud por debajo = escena nueva
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.45"))


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # GPU de los Mac con Apple Silicon
    return "cpu"


_clip = None
_clip_lock = threading.Lock()
_ocr = None
_ocr_lock = threading.Lock()


def _clip_model():
    global _clip
    if _clip is None:
        from sentence_transformers import SentenceTransformer

        _clip = SentenceTransformer(CLIP_MODEL, device=_device())
    return _clip


def _ocr_reader():
    global _ocr
    if _ocr is None:
        import easyocr

        # EasyOCR solo acelera con CUDA; en Mac y servidores sin GPU va por CPU.
        import torch

        _ocr = easyocr.Reader(["es", "en"], gpu=torch.cuda.is_available(), verbose=False)
    return _ocr


def warmup() -> None:
    """Carga (y la primera vez descarga) los modelos para que el primer video no espere."""
    with _clip_lock:
        _clip_model()
    with _ocr_lock:
        _ocr_reader()


# ---------------------------------------------------------------- candidatos


def candidate_frames(
    video: Path, duration: float, workdir: Path, crop: str | None = None
) -> list[tuple[float, Path]]:
    interval = max(duration / MAX_CANDIDATES, 0.5)
    out = workdir / "candidates"
    out.mkdir(exist_ok=True)
    filters = ([crop] if crop else []) + [f"fps=1/{interval:.3f}", "scale='min(768,iw)':-2"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", ",".join(filters),
         "-q:v", "4", "-frames:v", str(MAX_CANDIDATES), str(out / "c_%04d.jpg")],
        check=True,
    )
    return [(i * interval, p) for i, p in enumerate(sorted(out.glob("c_*.jpg")))]


# ---------------------------------------------------------------- selección


def embed_images(paths: list[Path]) -> np.ndarray:
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in paths]
    with _clip_lock:
        emb = _clip_model().encode(images, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    for im in images:
        im.close()
    return emb


def sharpness(path: Path) -> float:
    """Varianza del laplaciano: alta = imagen nítida, baja = borrosa (movimiento, desenfoque)."""
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0.0


def is_blank(path: Path) -> bool:
    """Fotograma casi todo negro (pantallas finales, transiciones a negro)."""
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return gray is None or float((gray < 30).mean()) > 0.7


def split_scenes(emb: np.ndarray, threshold: float = SCENE_THRESHOLD) -> list[list[int]]:
    """Agrupa índices consecutivos en escenas comparando cada fotograma con el inicio de su escena."""
    scenes: list[list[int]] = []
    for i in range(len(emb)):
        if not scenes or float(emb[i] @ emb[scenes[-1][0]]) < threshold:
            scenes.append([i])
        else:
            scenes[-1].append(i)
    return scenes


def k_center(emb: np.ndarray, k: int) -> list[int]:
    """Elige k puntos que cubran lo mejor posible el espacio (siempre el más lejano a los ya elegidos)."""
    if len(emb) <= k:
        return list(range(len(emb)))
    chosen = [0]
    dist = 1 - emb @ emb[0]
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, 1 - emb @ emb[nxt])
    return sorted(chosen)


def select_frames(candidates: list[tuple[float, Path]], max_frames: int) -> list[tuple[float, Path]]:
    non_blank = [c for c in candidates if not is_blank(c[1])]
    candidates = non_blank or candidates
    if not candidates:
        return []
    emb = embed_images([p for _, p in candidates])
    scenes = split_scenes(emb)

    reps = []
    for scene in scenes:
        best = max(scene, key=lambda i: sharpness(candidates[i][1]))
        reps.append(best)

    # Videos muy estáticos (una sola escena larga): asegura algunos fotogramas repartidos.
    min_frames = min(6, len(candidates), max_frames)
    if len(reps) < min_frames:
        extra = np.linspace(0, len(candidates) - 1, min_frames).round().astype(int)
        reps = sorted(set(reps) | set(int(i) for i in extra))

    if len(reps) > max_frames:
        keep = k_center(emb[reps], max_frames)
        reps = [reps[i] for i in keep]

    return [candidates[i] for i in sorted(reps)]


# ---------------------------------------------------------------- OCR


def _norm(text: str) -> str:
    return re.sub(r"[^\w°%/.,]+", " ", text.lower()).strip()


# Llamadas a la acción de redes sociales: no aportan nada a la receta.
_CTA = re.compile(
    r"suscr[ií]b|subscrib|\blike\b|coment|s[ií]gue|follow|compart|\bshare\b|guarda|\bsave\b"
    r"|link\s*en|\bbio\b|activa|campan|notific|\bbete\b",
    re.IGNORECASE,
)


def _keep_word(word: str) -> bool:
    word = word.strip()
    if _CTA.search(word):
        return False
    # Fragmentos sueltos muy cortos suelen ser ruido del OCR, salvo números ("2", "1/2").
    return any(c.isdigit() for c in word) or len(word) >= 3


def read_text(frames: list[tuple[float, Path]], fmt_ts) -> str:
    """Texto en pantalla, una línea por fotograma con texto nuevo: "[00:12] 200 g harina"."""
    lines: list[str] = []
    seen: list[str] = []
    for ts, path in frames:
        with _ocr_lock:
            results = _ocr_reader().readtext(str(path), detail=1, paragraph=False)
        words = [t.strip() for _box, t, conf in results if conf >= OCR_MIN_CONFIDENCE and _keep_word(t)]
        if not words:
            continue
        text = " ".join(words)
        norm = _norm(text)
        if any(difflib.SequenceMatcher(None, norm, s).ratio() > 0.85 for s in seen[-8:]):
            continue
        seen.append(norm)
        lines.append(f"[{fmt_ts(ts)}] {text}")
    return "\n".join(lines)

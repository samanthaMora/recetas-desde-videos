"""Video -> receta.

1. yt-dlp: metadatos, descripción, subtítulos y el video (máx. 720p).
2. Transcripción: subtítulos si existen; si no, Whisper local (faster-whisper).
3. ffmpeg: fotogramas repartidos por el video (texto en pantalla, ingredientes, técnica).
4. Claude (o tu modelo local con RECIPE_ENGINE=local): junta todo y devuelve la receta en JSON.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import logging

import anthropic
import yt_dlp

from .schema import RECIPE_JSON_SCHEMA, Recipe

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
ENGINE = os.getenv("RECIPE_ENGINE", "claude").lower()  # claude | local (modelo abierto en tu Mac, ver local_llm.py)
# Súbelo cada vez que cambies SYSTEM_PROMPT o el esquema: así el dataset sabe qué versión generó cada receta.
PROMPT_VERSION = "v2"  # v2: fotogramas elegidos con CLIP + texto en pantalla (OCR)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "20"))
FRAME_SELECTION = os.getenv("FRAME_SELECTION", "smart")  # smart (CLIP) | uniform (repartidos por igual)
USE_OCR = os.getenv("USE_OCR", "1") == "1"
USE_CLASSIFIER = os.getenv("USE_CLASSIFIER", "1") == "1"  # filtro previo con el modelo propio
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "1200"))  # 20 min
COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE")  # necesario a veces para Instagram/Facebook

ALLOWED_HOSTS = (
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "facebook.com",
    "fb.watch",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "pinterest.com",
    "pin.it",
)

log = logging.getLogger("recetas.pipeline")

URL_RE = re.compile(r"https?://[^\s<>\"']+")

Progress = Callable[[str], None]


class PipelineError(Exception):
    """Error con un mensaje apto para mostrar al usuario."""

    code = "error"


class NotRecipeError(PipelineError):
    """El clasificador propio está muy seguro de que el video no es una receta."""

    code = "not_recipe"


@dataclass
class VideoData:
    url: str
    title: str = ""
    description: str = ""
    uploader: str = ""
    duration: float = 0
    transcript: str = ""
    transcript_source: str = ""
    ocr_text: str = ""
    frame_method: str = ""
    frame_count: int = 0
    recipe_probability: float | None = None  # del clasificador propio
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    frames: list[tuple[float, Path]] = field(default_factory=list)


# ---------------------------------------------------------------- URL


def extract_url(text: str) -> str:
    """Saca el enlace del texto compartido ("Mira este video https://...") y lo valida."""
    match = URL_RE.search(text or "")
    if not match:
        raise PipelineError("No encontré ningún enlace en lo que compartiste.")
    url = match.group(0).rstrip(".,)")
    host = (urlparse(url).hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise PipelineError(
            "Por ahora solo acepto videos de YouTube, Instagram, Facebook, TikTok, X o Pinterest."
        )
    return url


# ---------------------------------------------------------------- descarga


def _ydl_base_opts() -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        # Nunca usar el extractor genérico: evita que el servidor descargue URLs arbitrarias.
        "allowed_extractors": ["default", "-generic"],
    }
    if COOKIES_FILE:
        opts["cookiefile"] = COOKIES_FILE
    return opts


def _pick_subtitles(info: dict) -> tuple[str, bool] | None:
    """Elige una pista de subtítulos: manuales primero, luego automáticos en el idioma original."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    lang = (info.get("language") or "").split("-")[0]

    manual_langs = [l for l in manual if l != "live_chat"]
    if manual_langs:
        best = next((l for l in manual_langs if lang and l.startswith(lang)), manual_langs[0])
        return best, False
    # Los automáticos de YouTube traducen a todos los idiomas; "-orig" es el original. Un video con
    # doblaje automático tiene varios "-orig" (uno por pista de audio): hay que elegir el de su idioma.
    orig = [l for l in auto if l.endswith("-orig")]
    if lang:
        same = next((l for l in orig if l.split("-")[0] == lang), None)
        if same:
            return same, True
        if lang in auto:
            return lang, True
    if len(orig) == 1:
        return orig[0], True
    return None  # idioma dudoso: mejor que Whisper escuche el audio


def check_is_recipe(data: VideoData, force: bool) -> None:
    """Filtro previo: rechaza (sin gastar en descarga ni en Claude) lo que claramente no es receta."""
    if not USE_CLASSIFIER:
        return
    try:
        from . import classifier

        clf = classifier.get()
        if clf is None:
            return
        data.recipe_probability = clf.probability(data.title, data.description)
    except Exception:
        log.exception("El clasificador falló; sigo sin filtro previo")
        return
    enough_text = len(classifier.video_text(data.title, data.description)) >= 30
    if not force and enough_text and data.recipe_probability < clf.reject_below:
        raise NotRecipeError(
            "Este video no parece una receta de cocina. Si crees que sí lo es, puedes analizarlo de todos modos."
        )


def download(url: str, workdir: Path, data: VideoData, force: bool = False) -> Path:
    try:
        with yt_dlp.YoutubeDL(_ydl_base_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise PipelineError(
            "No pude abrir ese video. Puede ser privado, haber sido borrado o la plataforma "
            "bloqueó la descarga."
        ) from e

    if info.get("_type") == "playlist":
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise PipelineError("Esa publicación no contiene ningún video.")
        info = entries[0]

    data.title = info.get("title") or ""
    data.description = info.get("description") or ""
    data.uploader = info.get("uploader") or info.get("channel") or ""
    data.duration = float(info.get("duration") or 0)
    check_is_recipe(data, force)
    if data.duration > MAX_VIDEO_SECONDS:
        raise PipelineError(
            f"El video dura más de {MAX_VIDEO_SECONDS // 60} minutos; por ahora el límite es ese."
        )

    opts = _ydl_base_opts() | {
        "outtmpl": str(workdir / "video.%(ext)s"),
        "format": "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
        "merge_output_format": "mp4",
        "max_filesize": 500 * 1024 * 1024,
    }
    sub = _pick_subtitles(info)
    if sub:
        lang, is_auto = sub
        opts |= {
            "writesubtitles": not is_auto,
            "writeautomaticsub": is_auto,
            "subtitleslangs": [lang],
            "subtitlesformat": "vtt",
        }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([info.get("webpage_url") or url])
    except yt_dlp.utils.DownloadError as e:
        raise PipelineError("La plataforma no dejó descargar el video.") from e

    videos = [p for p in workdir.glob("video.*") if p.suffix not in (".vtt", ".part")]
    if not videos:
        raise PipelineError("No se pudo descargar el video.")
    return videos[0]


# ---------------------------------------------------------------- transcripción


def _fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def parse_vtt(path: Path) -> str:
    """VTT -> "[mm:ss] texto", sin etiquetas ni las líneas repetidas de los subtítulos automáticos."""
    lines: list[str] = []
    last_text = ""
    current_ts = "00:00"
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if "-->" in raw:
            h_m_s = raw.split("-->")[0].strip().split(":")
            try:
                secs = sum(float(x) * 60**i for i, x in enumerate(reversed(h_m_s)))
                current_ts = _fmt_ts(secs)
            except ValueError:
                pass
            continue
        if not raw or raw == "WEBVTT" or raw.startswith(("Kind:", "Language:", "NOTE")) or raw.isdigit():
            continue
        text = re.sub(r"<[^>]+>", "", raw).strip()
        if text and text != last_text:
            lines.append(f"[{current_ts}] {text}")
            last_text = text
    return "\n".join(lines)


_whisper_model = None
_whisper_lock = threading.Lock()


def transcribe_with_whisper(video: Path) -> str:
    global _whisper_model
    from faster_whisper import WhisperModel

    with _whisper_lock:
        if _whisper_model is None:
            _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, _info = _whisper_model.transcribe(str(video), vad_filter=True)
        return "\n".join(f"[{_fmt_ts(s.start)}] {s.text.strip()}" for s in segments)


def has_audio(video: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


# ---------------------------------------------------------------- fotogramas


def detect_crop(video: Path) -> str | None:
    """Detecta bandas negras (p. ej. un video vertical dentro de un cuadro horizontal).

    Devuelve un filtro "crop=w:h:x:y" para ffmpeg, o None si no vale la pena recortar.
    Usa el recorte más frecuente entre muchos fotogramas, para que letreros ocasionales
    sobre el negro ("SUSCRÍBETE") no amplíen el área.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        width, height = (int(x) for x in probe.stdout.strip().split(",")[:2])
    except ValueError:
        return None
    out = subprocess.run(
        ["ffmpeg", "-i", str(video), "-vf", "fps=2,cropdetect=limit=24:round=2:reset=1",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", out.stderr)
    if not crops:
        return None
    w, h, x, y = (int(v) for v in Counter(crops).most_common(1)[0][0])
    if w <= 0 or h <= 0 or w * h > 0.85 * width * height:
        return None
    return f"crop={w}:{h}:{x}:{y}"


def extract_frames(
    video: Path, duration: float, workdir: Path, crop: str | None = None
) -> list[tuple[float, Path]]:
    if duration <= 0:
        duration = _probe_duration(video)

    interval = max(duration / MAX_FRAMES, 2.0)
    frames_dir = workdir / "frames"
    frames_dir.mkdir(exist_ok=True)
    filters = ([crop] if crop else []) + [f"fps=1/{interval:.3f}", "scale='min(768,iw)':-2"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", ",".join(filters),
         "-q:v", "4", "-frames:v", str(MAX_FRAMES), str(frames_dir / "f_%03d.jpg")],
        check=True,
    )
    files = sorted(frames_dir.glob("f_*.jpg"))
    return [(i * interval, f) for i, f in enumerate(files)]


def choose_frames(video: Path, duration: float, workdir: Path) -> tuple[list[tuple[float, Path]], str]:
    """Fotogramas elegidos con CLIP; si la visión falla, repartidos por igual."""
    crop = detect_crop(video)
    if FRAME_SELECTION == "smart":
        try:
            from . import vision

            if duration <= 0:
                duration = _probe_duration(video)
            candidates = vision.candidate_frames(video, duration, workdir, crop)
            selected = vision.select_frames(candidates, MAX_FRAMES)
            # Nombres estables f_001.jpg… para guardarlos en el dataset.
            frames = []
            for n, (ts, path) in enumerate(selected, start=1):
                target = path.with_name(f"f_{n:03d}.jpg")
                path.rename(target)
                frames.append((ts, target))
            return frames, "clip"
        except Exception:
            log.exception("La selección con CLIP falló; uso fotogramas repartidos")
    return extract_frames(video, duration, workdir, crop), "uniform"


def _probe_duration(video: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return 60.0


# ---------------------------------------------------------------- Claude

SYSTEM_PROMPT = """Eres un chef y editor de recetas. Recibes la información de un video de cocina \
de redes sociales: título, descripción del autor, transcripción del audio y fotogramas del video \
con su minuto aproximado. Tu trabajo es reconstruir la receta completa y fiel.

Cómo combinar las fuentes:
- La descripción del autor suele traer la receta escrita con cantidades exactas: si existe, es la \
fuente principal.
- El texto que aparece en pantalla en los fotogramas y lo que se dice en el audio completan o \
corrigen la descripción.
- El bloque <texto_en_pantalla> es OCR automático de los fotogramas: suele traer cantidades y \
temperaturas exactas, pero puede tener letras mal leídas. Contrástalo con la imagen.
- Si una cantidad no se dice ni se escribe en ninguna parte, estímala por lo que se ve en los \
fotogramas y marca ese ingrediente con estimated=true. Si ni siquiera se puede estimar, deja \
quantity en null.
- No inventes ingredientes ni pasos que no aparezcan. Lo que no quede claro va en uncertainties.
- Los pasos deben poder seguirse sin ver el video: incluye temperaturas, tiempos, fuego y señales \
de punto ("hasta que dore", "hasta que espese").
- Ignora promociones, saludos, códigos de descuento y llamadas a seguir la cuenta.
- Si el video no presenta una receta, responde is_recipe=false, explica en description qué \
muestra el video y deja las listas vacías.

Escribe toda la receta en {language}, aunque el video esté en otro idioma. Usa unidades métricas \
y, si el video usa tazas o cucharadas, conserva esas medidas tal como aparecen."""


def _build_content(data: VideoData) -> list[dict]:
    content: list[dict] = []
    for ts, path in data.frames:
        content.append({"type": "text", "text": f"Fotograma ~{_fmt_ts(ts)}"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(path.read_bytes()).decode("ascii"),
            },
        })

    transcript = data.transcript.strip() or "(el video no tiene audio hablado)"
    content.append({
        "type": "text",
        "text": (
            f"Enlace: {data.url}\n"
            f"Autor: {data.uploader or 'desconocido'}\n"
            f"Duración: {_fmt_ts(data.duration)}\n"
            f"Título: {data.title or '(sin título)'}\n\n"
            f"<descripcion_del_autor>\n{data.description.strip() or '(vacía)'}\n</descripcion_del_autor>\n\n"
            f"<transcripcion fuente=\"{data.transcript_source}\">\n{transcript}\n</transcripcion>\n\n"
            f"<texto_en_pantalla>\n{data.ocr_text.strip() or '(no se detectó texto)'}\n</texto_en_pantalla>\n\n"
            "Reconstruye la receta de este video."
        ),
    })
    return content


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise PipelineError(
                "Falta la clave de la API de Claude: ponla en ANTHROPIC_API_KEY (archivo backend/.env)."
            )
        # Si la conexión se queda muda (p. ej. el equipo se suspendió), cortar y reintentar
        # en vez de bloquear un proceso para siempre.
        _client = anthropic.Anthropic(timeout=300.0, max_retries=2)
    return _client


def analyze_with_claude(data: VideoData, language: str) -> Recipe:
    client = _get_client()
    try:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=32000,
            # Si el modelo rechaza la petición, la API reintenta con otro modelo automáticamente.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": RECIPE_JSON_SCHEMA},
            },
            system=SYSTEM_PROMPT.format(language=language),
            messages=[{"role": "user", "content": _build_content(data)}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise PipelineError("La clave de la API de Claude no es válida.") from e
    except anthropic.RateLimitError as e:
        raise PipelineError("Hay demasiadas solicitudes ahora mismo; prueba en un minuto.") from e
    except anthropic.APIStatusError as e:
        raise PipelineError(f"Error del servicio de análisis ({e.status_code}).") from e
    except anthropic.APIConnectionError as e:
        raise PipelineError("No pude conectar con el servicio de análisis.") from e

    if message.stop_reason == "refusal":
        raise PipelineError("No se pudo analizar este video.")
    if message.stop_reason == "max_tokens":
        raise PipelineError("La receta salió demasiado larga; inténtalo de nuevo.")

    data.model_used = message.model
    data.input_tokens = message.usage.input_tokens
    data.output_tokens = message.usage.output_tokens

    text = next((b.text for b in message.content if b.type == "text"), None)
    if not text:
        raise PipelineError("El análisis no devolvió ninguna receta.")
    return Recipe.model_validate(json.loads(text))


# ---------------------------------------------------------------- todo junto


def run(
    shared_text: str,
    language: str = "español",
    progress: Progress = lambda _: None,
    frames_dest: Path | None = None,
    force: bool = False,
    on_data: Callable[[VideoData], object] | None = None,
    engine: str | None = None,
) -> tuple[Recipe, VideoData]:
    """frames_dest: carpeta donde conservar los fotogramas (para el dataset de entrenamiento).
    force: saltarse el filtro del clasificador (el usuario insiste en que es una receta).
    on_data: recibe los datos del video desde el principio (útiles aunque luego haya un error).
    engine: "claude" o "local" (por defecto, RECIPE_ENGINE)."""
    engine = (engine or ENGINE).lower()
    if engine not in ("claude", "local"):
        raise PipelineError(f"Motor desconocido: {engine} (usa claude o local).")
    url = extract_url(shared_text)
    data = VideoData(url=url)
    if on_data:
        on_data(data)

    with tempfile.TemporaryDirectory(prefix="receta_") as tmp:
        workdir = Path(tmp)

        progress("downloading")
        video = download(url, workdir, data, force)

        progress("transcribing")
        subs = list(workdir.glob("*.vtt"))
        if subs:
            data.transcript = parse_vtt(subs[0])
            data.transcript_source = "subtítulos"
        if not data.transcript and has_audio(video):
            data.transcript = transcribe_with_whisper(video)
            data.transcript_source = "whisper"

        progress("extracting_frames")
        data.frames, data.frame_method = choose_frames(video, data.duration, workdir)
        data.frame_count = len(data.frames)
        if USE_OCR:
            try:
                from . import vision

                data.ocr_text = vision.read_text(data.frames, _fmt_ts)
            except Exception:
                log.exception("OCR falló; sigo sin texto en pantalla")
        if frames_dest is not None:
            frames_dest.mkdir(parents=True, exist_ok=True)
            kept = []
            for ts, path in data.frames:
                target = frames_dest / path.name
                shutil.copy2(path, target)
                kept.append((ts, target))
            data.frames = kept

        progress("analyzing")
        if engine == "local":
            from .local_llm import analyze_local

            recipe = analyze_local(data, language)
        else:
            recipe = analyze_with_claude(data, language)

    if frames_dest is None:
        data.frames = []  # los archivos temporales ya no existen
    return recipe, data

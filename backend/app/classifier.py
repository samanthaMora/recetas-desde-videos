"""Clasificador propio «¿este video es una receta?» (fase 4).

Se entrena con scripts/train_classifier.py (y se explica en notebooks/clasificador_recetas.ipynb).
Aquí solo está lo que necesita la app en producción: preparar el texto y predecir.

Uso en la app: antes de descargar el video y pagar el análisis, si el modelo está MUY seguro de
que no es una receta, se rechaza al momento. El umbral se elige en el entrenamiento para que
(en el conjunto de validación) ninguna receta real quede bloqueada.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "recipe_classifier"

_URL = re.compile(r"https?://\S+")


def video_text(title: str, description: str) -> str:
    """Texto que ve el modelo: título + inicio de la descripción, sin enlaces.

    El nombre del canal NO se usa a propósito: el modelo aprendería atajos ("canal de cocina")
    que no sirven para Instagram/TikTok ni para canales nuevos.
    """
    desc = _URL.sub(" ", description or "")
    desc = desc.replace("#", " ")
    text = f"{title or ''}\n{desc[:1000]}"
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- modelos


class _TfidfModel:
    def __init__(self, path: Path) -> None:
        import joblib

        self.pipeline = joblib.load(path / "tfidf_lr.joblib")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)[:, 1]


class _EmbeddingsModel:
    def __init__(self, path: Path) -> None:
        import joblib

        self.lr = joblib.load(path / "embeddings_lr.joblib")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        from .embeddings import _encode

        emb = _encode(["query: " + t for t in texts])
        return self.lr.predict_proba(emb)[:, 1]


class _FinetunedModel:
    def __init__(self, path: Path) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        from .vision import _device

        self.torch = torch
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(path / "finetuned")
        self.encoder = AutoModel.from_pretrained(path / "finetuned").to(self.device).eval()
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(path / "finetuned" / "head.pt", map_location="cpu"))
        self.head.to(self.device).eval()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        batch = self.tokenizer(["query: " + t for t in texts], padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            return torch.sigmoid(self.head(pooled)).squeeze(-1).float().cpu().numpy()


_LOADERS = {"tfidf_lr": _TfidfModel, "embeddings_lr": _EmbeddingsModel, "finetuned": _FinetunedModel}


class RecipeClassifier:
    def __init__(self, path: Path = MODEL_DIR) -> None:
        self.config = json.loads((path / "config.json").read_text())
        self.model = _LOADERS[self.config["type"]](path)
        self.reject_below = float(self.config["reject_below"])

    def probability(self, title: str, description: str) -> float:
        return float(self.model.predict_proba([video_text(title, description)])[0])


_instance: RecipeClassifier | None = None
_lock = threading.Lock()
_failed = False


def get() -> RecipeClassifier | None:
    """El clasificador entrenado, o None si todavía no hay ninguno (la app funciona igual)."""
    global _instance, _failed
    if _instance is None and not _failed:
        with _lock:
            if _instance is None and not _failed:
                try:
                    _instance = RecipeClassifier()
                except FileNotFoundError:
                    _failed = True
    return _instance

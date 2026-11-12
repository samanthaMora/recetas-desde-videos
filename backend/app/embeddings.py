"""Búsqueda semántica y recetas parecidas con embeddings de texto.

Un modelo de embeddings (multilingual-e5-small) convierte cada receta en un vector de 384 números.
Recetas con significado parecido quedan cerca en ese espacio, aunque usen palabras distintas:
"crema de brócoli" queda cerca de "sopa verde cremosa".

- Búsqueda: se convierte la consulta en vector y se ordenan las recetas por similitud coseno.
  Es búsqueda híbrida: a la similitud semántica se le suma un pequeño extra por cada palabra de la
  consulta que aparece literalmente en la receta ("pollo"), porque los embeddings a veces
  confunden ingredientes concretos.
- Recetas parecidas: los vecinos más cercanos del vector de una receta.

Los vectores se guardan en SQLite y se mantienen en memoria en una matriz numpy. Con decenas de
miles de recetas esto va sobrado; con millones habría que pasar a pgvector o similar.
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata

import numpy as np

from . import db
from .schema import Recipe

TEXT_MODEL = os.getenv("TEXT_EMBED_MODEL", "intfloat/multilingual-e5-small")
LEXICAL_BONUS = 0.03  # por palabra de la consulta encontrada literalmente
LEXICAL_MAX = 0.09
# e5 da puntuaciones comprimidas (casi todo entre 0.75 y 0.95), así que se corta así:
# mínimo absoluto (por debajo nada es relevante) y ventana relativa al mejor resultado.
# Medido con consultas de prueba; recalibrar cuando haya más recetas.
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.84"))
SEARCH_WINDOW = float(os.getenv("SEARCH_WINDOW", "0.05"))
SIMILAR_MIN_SCORE = float(os.getenv("SIMILAR_MIN_SCORE", "0.86"))  # sobre vectores del texto corto

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        from .vision import _device

        _model = SentenceTransformer(TEXT_MODEL, device=_device())
    return _model


def _encode(texts: list[str]) -> np.ndarray:
    with _model_lock:
        return _get_model().encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)


# ---------------------------------------------------------------- texto


def _fold(text: str) -> str:
    """minúsculas y sin tildes: "Brócoli" -> "brocoli"."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


_STOPWORDS = {
    "con", "sin", "para", "por", "que", "una", "uno", "unos", "unas", "los", "las", "del", "algo",
    "receta", "recetas", "quiero", "hacer", "como", "facil", "rica", "rico", "mas", "muy",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9ñ]+", _fold(text)) if len(t) >= 3 and t not in _STOPWORDS}


def recipe_text(recipe: Recipe) -> str:
    parts = [recipe.title + ".", recipe.description]
    if recipe.ingredients:
        parts.append("Ingredientes: " + ", ".join(i.name for i in recipe.ingredients) + ".")
    if recipe.total_time_minutes:
        quick = " (rápida)" if recipe.total_time_minutes <= 30 else ""
        parts.append(f"Tiempo total: {recipe.total_time_minutes:g} minutos{quick}.")
    if recipe.difficulty:
        parts.append(f"Dificultad: {recipe.difficulty}.")
    if recipe.steps:
        parts.append("Pasos: " + " ".join(s.instruction for s in recipe.steps))
    return " ".join(parts)[:2000]


def recipe_text_short(recipe: Recipe) -> str:
    """Sin pasos: los verbos genéricos ("mezclar", "servir") hacen que todas las recetas se parezcan.

    Medido con pares etiquetados a mano: separa mejor "parecidas" de "distintas" que el texto completo.
    """
    names = ", ".join(i.name for i in recipe.ingredients)
    return f"{recipe.title}. {recipe.description} Ingredientes: {names}."


# ---------------------------------------------------------------- índice


SHORT_KEY = TEXT_MODEL + "#short"  # nombre con el que se guardan los vectores del texto corto


class _Index:
    """Dos matrices: `full` (texto completo, para buscar) y `short` (para recetas parecidas)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ids: list[str] = []
        self.urls: list[str] = []
        self.words: list[set[str]] = []
        self.full = np.zeros((0, 384), dtype=np.float32)
        self.short = np.zeros((0, 384), dtype=np.float32)

    def upsert(self, analysis_id: str, url: str, full: np.ndarray, short: np.ndarray, words: set[str]) -> None:
        full, short = full.astype(np.float32), short.astype(np.float32)
        with self.lock:
            if analysis_id in self.ids:
                i = self.ids.index(analysis_id)
                self.full[i], self.short[i] = full, short
                self.words[i], self.urls[i] = words, url
                return
            if self.full.shape[1] != full.shape[0]:
                self.full = np.zeros((0, full.shape[0]), dtype=np.float32)
                self.short = np.zeros((0, full.shape[0]), dtype=np.float32)
            self.ids.append(analysis_id)
            self.urls.append(url)
            self.words.append(words)
            self.full = np.vstack([self.full, full[None, :]])
            self.short = np.vstack([self.short, short[None, :]])

    def scores(self, vector: np.ndarray, short: bool = False) -> np.ndarray:
        with self.lock:
            matrix = self.short if short else self.full
            return matrix @ vector if len(self.ids) else np.zeros(0)


_index = _Index()


def _current_recipe(analysis_id: str) -> tuple[Recipe, str] | None:
    row = db.get_analysis(analysis_id)
    if not row or row["status"] != "done" or not row["is_recipe"]:
        return None
    recipe_json = db.latest_correction(analysis_id) or row["recipe_json"]
    return Recipe.model_validate_json(recipe_json), row["url"]


def index_recipe(analysis_id: str) -> None:
    """Calcula (o recalcula tras una corrección) el vector de una receta y lo guarda."""
    found = _current_recipe(analysis_id)
    if not found:
        return
    recipe, url = found
    text = recipe_text(recipe)
    full, short = _encode(["passage: " + text, "passage: " + recipe_text_short(recipe)])
    db.save_embedding(analysis_id, TEXT_MODEL, full.astype(np.float32).tobytes())
    db.save_embedding(analysis_id, SHORT_KEY, short.astype(np.float32).tobytes())
    _index.upsert(analysis_id, url, full, short, _tokens(text))


def load_index() -> int:
    """Carga los vectores guardados y calcula los que falten. Devuelve cuántas recetas hay."""
    full = {i: (url, blob) for i, url, blob in db.load_embeddings(TEXT_MODEL)}
    short = {i: blob for i, _url, blob in db.load_embeddings(SHORT_KEY)}
    for analysis_id, (url, blob) in full.items():
        found = _current_recipe(analysis_id)
        if found and analysis_id in short:
            _index.upsert(
                analysis_id, url,
                np.frombuffer(blob, dtype=np.float32), np.frombuffer(short[analysis_id], dtype=np.float32),
                _tokens(recipe_text(found[0])),
            )
    for analysis_id in db.ids_missing_embedding(SHORT_KEY):
        index_recipe(analysis_id)
    return len(_index.ids)


def search(query: str, allowed_ids: list[str] | None = None, limit: int = 20) -> list[tuple[str, float]]:
    query = query.strip()
    if not query:
        return []
    qvec = _encode(["query: " + query])[0]
    scores = _index.scores(qvec)
    qwords = _tokens(query)
    allowed = set(allowed_ids) if allowed_ids is not None else None

    results = []
    with _index.lock:
        for i, analysis_id in enumerate(_index.ids):
            if allowed is not None and analysis_id not in allowed:
                continue
            bonus = min(LEXICAL_MAX, LEXICAL_BONUS * len(qwords & _index.words[i]))
            results.append((analysis_id, float(scores[i]) + bonus))
    results.sort(key=lambda r: r[1], reverse=True)
    if not results:
        return []
    cutoff = max(SEARCH_MIN_SCORE, results[0][1] - SEARCH_WINDOW)
    return [r for r in results if r[1] >= cutoff][:limit]


def similar(analysis_id: str, limit: int = 6) -> list[tuple[str, float]]:
    with _index.lock:
        if analysis_id not in _index.ids:
            return []
        i = _index.ids.index(analysis_id)
        vector = _index.short[i].copy()
        url = _index.urls[i]
    scores = _index.scores(vector, short=True)
    results = []
    with _index.lock:
        for j, other in enumerate(_index.ids):
            if other == analysis_id or _index.urls[j] == url or scores[j] < SIMILAR_MIN_SCORE:
                continue
            results.append((other, float(scores[j])))
    results.sort(key=lambda r: r[1], reverse=True)
    return results[:limit]

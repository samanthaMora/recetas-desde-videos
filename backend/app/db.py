"""Base de datos SQLite: cada análisis y cada corrección quedan guardados.

Esto sirve para dos cosas:
- reemplaza los trabajos en memoria (sobreviven a reinicios del servidor);
- es el dataset para entrenar modelos propios (ver scripts/export_dataset.py).

Tablas
  analyses     una fila por video analizado: entradas (descripción, transcripción, fotogramas),
               salida (receta), modelo usado y tokens.
  corrections  recetas corregidas por personas: las etiquetas de mejor calidad para entrenar.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DB_PATH = DATA_DIR / "recetas.db"
FRAMES_DIR = DATA_DIR / "frames"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                TEXT PRIMARY KEY,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    status            TEXT NOT NULL,
    error             TEXT,
    shared_text       TEXT NOT NULL,
    url               TEXT NOT NULL,
    platform          TEXT,
    language          TEXT NOT NULL,
    title             TEXT,
    description       TEXT,
    uploader          TEXT,
    duration          REAL,
    transcript        TEXT,
    transcript_source TEXT,
    frames_json       TEXT,      -- [{"ts": 12.5, "file": "frames/<id>/f_001.jpg"}, ...]
    recipe_json       TEXT,
    is_recipe         INTEGER,
    model             TEXT,
    prompt_version    TEXT,
    input_tokens      INTEGER,
    output_tokens     INTEGER
);
CREATE INDEX IF NOT EXISTS analyses_url ON analyses (url, language, status);

CREATE TABLE IF NOT EXISTS corrections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id  TEXT NOT NULL REFERENCES analyses(id),
    created_at   REAL NOT NULL,
    recipe_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS corrections_analysis ON corrections (analysis_id);

CREATE TABLE IF NOT EXISTS recipe_embeddings (
    analysis_id  TEXT NOT NULL REFERENCES analyses(id),
    model        TEXT NOT NULL,
    vector       BLOB NOT NULL,   -- float32
    created_at   REAL NOT NULL,
    PRIMARY KEY (analysis_id, model)
);
"""


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        _migrate(_conn)
    return _conn


# Columnas añadidas después de la primera versión: se agregan a bases de datos existentes.
_ADDED_COLUMNS = {
    "ocr_text": "TEXT",
    "frame_method": "TEXT",
    "classifier_score": "REAL",  # probabilidad de receta según el clasificador propio
    "error_code": "TEXT",        # p. ej. "not_recipe"
}


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(analyses)")}
    for name, sql_type in _ADDED_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE analyses ADD COLUMN {name} {sql_type}")
    conn.commit()


def _execute(sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
    with _lock:
        conn = _db()
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def platform_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    for name, keys in {
        "youtube": ("youtube.com", "youtu.be"),
        "instagram": ("instagram.com",),
        "facebook": ("facebook.com", "fb.watch"),
        "tiktok": ("tiktok.com",),
        "x": ("x.com", "twitter.com"),
        "pinterest": ("pinterest.com", "pin.it"),
    }.items():
        if any(host == k or host.endswith("." + k) for k in keys):
            return name
    return "otro"


def create_analysis(analysis_id: str, shared_text: str, url: str, language: str) -> None:
    now = time.time()
    _execute(
        "INSERT INTO analyses (id, created_at, updated_at, status, shared_text, url, platform, language)"
        " VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)",
        (analysis_id, now, now, shared_text, url, platform_of(url), language),
    )


def set_status(
    analysis_id: str,
    status: str,
    error: str | None = None,
    error_code: str | None = None,
    classifier_score: float | None = None,
) -> None:
    _execute(
        "UPDATE analyses SET status = ?, error = ?, error_code = ?, updated_at = ?,"
        " classifier_score = COALESCE(?, classifier_score) WHERE id = ?",
        (status, error, error_code, time.time(), classifier_score, analysis_id),
    )


def save_result(analysis_id: str, data: Any, recipe: Any, prompt_version: str) -> None:
    """data: pipeline.VideoData; recipe: schema.Recipe."""
    frames = [
        {"ts": round(ts, 2), "file": str(Path(p).relative_to(DATA_DIR))}
        for ts, p in data.frames
    ]
    _execute(
        """UPDATE analyses SET
             status = 'done', error = NULL, updated_at = ?,
             title = ?, description = ?, uploader = ?, duration = ?,
             transcript = ?, transcript_source = ?, frames_json = ?,
             ocr_text = ?, frame_method = ?, classifier_score = ?,
             recipe_json = ?, is_recipe = ?, model = ?, prompt_version = ?,
             input_tokens = ?, output_tokens = ?
           WHERE id = ?""",
        (
            time.time(), data.title, data.description, data.uploader, data.duration,
            data.transcript, data.transcript_source, json.dumps(frames),
            data.ocr_text, data.frame_method, data.recipe_probability,
            recipe.model_dump_json(), int(recipe.is_recipe), data.model_used, prompt_version,
            data.input_tokens, data.output_tokens, analysis_id,
        ),
    )


def get_analysis(analysis_id: str) -> dict | None:
    with _lock:
        row = _db().execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    return dict(row) if row else None


def find_done(url: str, language: str, engine: str = "claude") -> str | None:
    """Último análisis terminado de ese video con ese motor (los de Claude y los locales no se mezclan)."""
    engine_filter = "model LIKE 'local:%'" if engine == "local" else "(model IS NULL OR model NOT LIKE 'local:%')"
    with _lock:
        row = _db().execute(
            "SELECT id FROM analyses WHERE url = ? AND language = ? AND status = 'done' AND "
            + engine_filter + " ORDER BY created_at DESC LIMIT 1",
            (url, language),
        ).fetchone()
    return row["id"] if row else None


def add_correction(analysis_id: str, recipe_json: str) -> None:
    _execute(
        "INSERT INTO corrections (analysis_id, created_at, recipe_json) VALUES (?, ?, ?)",
        (analysis_id, time.time(), recipe_json),
    )


def latest_correction(analysis_id: str) -> str | None:
    with _lock:
        row = _db().execute(
            "SELECT recipe_json FROM corrections WHERE analysis_id = ? ORDER BY created_at DESC LIMIT 1",
            (analysis_id,),
        ).fetchone()
    return row["recipe_json"] if row else None


def save_embedding(analysis_id: str, model: str, vector: bytes) -> None:
    _execute(
        "INSERT OR REPLACE INTO recipe_embeddings (analysis_id, model, vector, created_at) VALUES (?, ?, ?, ?)",
        (analysis_id, model, vector, time.time()),
    )


def load_embeddings(model: str) -> list[tuple[str, str, bytes]]:
    with _lock:
        rows = _db().execute(
            "SELECT e.analysis_id, a.url, e.vector FROM recipe_embeddings e"
            " JOIN analyses a ON a.id = e.analysis_id WHERE e.model = ?",
            (model,),
        ).fetchall()
    return [(r["analysis_id"], r["url"], r["vector"]) for r in rows]


def ids_missing_embedding(model: str) -> list[str]:
    with _lock:
        rows = _db().execute(
            "SELECT a.id FROM analyses a LEFT JOIN recipe_embeddings e"
            " ON e.analysis_id = a.id AND e.model = ?"
            " WHERE a.status = 'done' AND a.is_recipe = 1 AND e.analysis_id IS NULL",
            (model,),
        ).fetchall()
    return [r["id"] for r in rows]


def mark_interrupted() -> None:
    """Al arrancar: los trabajos que quedaron a medias por un reinicio pasan a error."""
    _execute(
        "UPDATE analyses SET status = 'error', error = 'El servidor se reinició; inténtalo de nuevo.'"
        " WHERE status NOT IN ('done', 'error')"
    )


def iter_rows(sql: str, params: tuple = ()):
    with _lock:
        rows = _db().execute(sql, params).fetchall()
    for r in rows:
        yield dict(r)

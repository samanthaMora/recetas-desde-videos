"""API HTTP para las apps (iPhone, Android y web).

POST /api/jobs                   {"text": "<enlace o texto compartido>", "language": "español"} -> trabajo
GET  /api/jobs/{id}              estado del trabajo y, al terminar, la receta
POST /api/jobs/{id}/correction   {"recipe": {...}} receta corregida por el usuario (dato de entrenamiento)

El análisis tarda entre 30 s y unos minutos, así que se hace en segundo plano y la app
consulta el estado cada pocos segundos. Todo queda guardado en data/recetas.db.

NOTA para publicar: falta añadir cuentas de usuario y límites de uso, y pasar a Postgres + una cola
si hay varios servidores.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from . import db, embeddings, pipeline  # noqa: E402
from .schema import Recipe  # noqa: E402

log = logging.getLogger("recetas")

Status = Literal["queued", "downloading", "transcribing", "extracting_frames", "analyzing", "done", "error"]


class JobIn(BaseModel):
    text: str = Field(..., max_length=2000, description="Enlace del video o el texto compartido que lo contiene")
    language: str = Field("español", max_length=30)
    force: bool = Field(False, description="Analizar aunque el clasificador diga que no es una receta")


class CorrectionIn(BaseModel):
    recipe: Recipe


class SearchIn(BaseModel):
    query: str = Field(..., max_length=200)
    ids: Optional[list[str]] = Field(None, max_length=5000, description="Limitar la búsqueda a estas recetas")
    limit: int = Field(20, ge=1, le=100)


class SearchHit(BaseModel):
    id: str
    score: float


class SourceInfo(BaseModel):
    url: str
    title: str
    uploader: str
    duration: float
    transcript_source: str


class Job(BaseModel):
    id: str
    status: Status
    created_at: float
    recipe: Optional[Recipe] = None
    source: Optional[SourceInfo] = None
    error: Optional[str] = None
    error_code: Optional[str] = None  # "not_recipe": la app ofrece "Analizar de todos modos"


class SimilarHit(BaseModel):
    score: float
    job: Job


app = FastAPI(title="Recetas desde videos")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
db.mark_interrupted()


def _warmup() -> None:
    try:
        n = embeddings.load_index()
        log.info("Índice de búsqueda listo: %d recetas", n)
    except Exception:
        log.exception("No se pudo cargar el índice de búsqueda")
    if pipeline.ENGINE == "local":
        try:
            from . import local_llm

            local_llm.warmup()
            log.info("Modelo local listo: %s", local_llm.model_name())
        except Exception:
            log.exception("No se pudo cargar el modelo local")
    if pipeline.FRAME_SELECTION != "smart" and not pipeline.USE_OCR:
        return
    try:
        from . import vision

        vision.warmup()
        log.info("Modelos de visión cargados")
    except Exception:
        log.exception("No se pudieron cargar los modelos de visión; se usará el método simple")


def _reindex(job_id: str) -> None:
    try:
        embeddings.index_recipe(job_id)
    except Exception:
        log.exception("No se pudo indexar la receta %s", job_id)


# Carga los modelos y el índice en segundo plano al arrancar, para que nadie espere.
threading.Thread(target=_warmup, daemon=True).start()


def _to_job(row: dict) -> Job:
    done = row["status"] == "done"
    return Job(
        id=row["id"],
        status=row["status"],
        created_at=row["created_at"],
        error=row["error"],
        error_code=row.get("error_code"),
        recipe=Recipe.model_validate_json(row["recipe_json"]) if done and row["recipe_json"] else None,
        source=SourceInfo(
            url=row["url"],
            title=row["title"] or "",
            uploader=row["uploader"] or "",
            duration=row["duration"] or 0,
            transcript_source=row["transcript_source"] or "",
        ) if done else None,
    )


def _process(job_id: str, url: str, language: str, force: bool = False) -> None:
    data_ref: dict = {}
    try:
        recipe, data = pipeline.run(
            url,
            language,
            progress=lambda s: db.set_status(job_id, s),
            frames_dest=db.FRAMES_DIR / job_id,
            force=force,
            on_data=lambda d: data_ref.setdefault("data", d),
        )
        db.save_result(job_id, data, recipe, pipeline.PROMPT_VERSION)
        if recipe.is_recipe:
            _reindex(job_id)
    except pipeline.PipelineError as e:
        score = getattr(data_ref.get("data"), "recipe_probability", None)
        db.set_status(job_id, "error", str(e), e.code, score)
    except Exception:
        log.exception("Fallo inesperado en el trabajo %s", job_id)
        db.set_status(job_id, "error", "Algo salió mal al analizar el video.")


@app.post("/api/jobs", response_model=Job)
def create_job(body: JobIn) -> Job:
    try:
        url = pipeline.extract_url(body.text)
    except pipeline.PipelineError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cached = db.find_done(url, body.language, pipeline.ENGINE)
    if cached:
        return _to_job(db.get_analysis(cached))

    job_id = uuid.uuid4().hex
    db.create_analysis(job_id, body.text, url, body.language)
    _executor.submit(_process, job_id, url, body.language, body.force)
    return _to_job(db.get_analysis(job_id))


@app.get("/api/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    row = db.get_analysis(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="No existe ese análisis.")
    return _to_job(row)


@app.post("/api/jobs/{job_id}/correction")
def add_correction(job_id: str, body: CorrectionIn) -> dict:
    row = db.get_analysis(job_id)
    if not row or row["status"] != "done":
        raise HTTPException(status_code=404, detail="No existe ese análisis.")
    if body.recipe.model_dump() == json.loads(row["recipe_json"]):
        return {"ok": True, "changed": False}
    db.add_correction(job_id, body.recipe.model_dump_json())
    _executor.submit(_reindex, job_id)  # la búsqueda usa la versión corregida
    return {"ok": True, "changed": True}


@app.post("/api/search", response_model=list[SearchHit])
def search(body: SearchIn) -> list[SearchHit]:
    hits = embeddings.search(body.query, body.ids, body.limit)
    return [SearchHit(id=i, score=round(s, 4)) for i, s in hits]


@app.get("/api/jobs/{job_id}/similar", response_model=list[SimilarHit])
def similar(job_id: str, limit: int = 6) -> list[SimilarHit]:
    limit = max(1, min(limit, 20))
    out = []
    for other_id, score in embeddings.similar(job_id, limit):
        row = db.get_analysis(other_id)
        if row:
            out.append(SimilarHit(score=round(score, 4), job=_to_job(row)))
    return out


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "engine": pipeline.ENGINE}

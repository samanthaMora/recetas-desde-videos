"""Motor local: genera la receta con un modelo abierto en tu Mac (MLX), sin llamar a Claude.

Se activa con RECIPE_ENGINE=local. De momento solo lee texto (título, descripción, transcripción y
el texto en pantalla que saca el OCR); no mira los fotogramas, así que no estima cantidades por la
imagen. Devuelve la misma estructura Recipe que Claude, así la app no nota la diferencia.

    LOCAL_MODEL    modelo de Hugging Face en formato MLX (por defecto Qwen2.5-7B-Instruct 4-bit, Apache 2.0)
    LOCAL_ADAPTER  carpeta con un adaptador LoRA propio entrenado para recetas completas (opcional)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import TYPE_CHECKING, Any

from .schema import Recipe

if TYPE_CHECKING:
    from .pipeline import VideoData

log = logging.getLogger(__name__)

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
LOCAL_ADAPTER = os.getenv("LOCAL_ADAPTER") or None
MAX_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "3000"))
# Límites de texto de entrada: más contexto = más memoria y más lento en un Mac de 16 GB.
MAX_DESCRIPTION = 5000
MAX_TRANSCRIPT = 12000
MAX_OCR = 2000

SYSTEM_PROMPT = """Eres un chef y editor de recetas. Recibes la información de un video de cocina \
de redes sociales: título, descripción del autor, transcripción del audio y el texto que aparece \
en pantalla (leído automáticamente, puede tener letras mal leídas). Reconstruye la receta completa y fiel.

Reglas:
- La descripción del autor, si trae la receta con cantidades, es la fuente principal. La \
transcripción y el texto en pantalla la completan o corrigen.
- No inventes ingredientes, cantidades ni pasos. Si una cantidad no se dice ni se escribe, pon \
quantity: null. Lo que no quede claro va en "uncertainties".
- Los pasos deben poder seguirse sin ver el video: temperaturas, tiempos, fuego y señales de \
punto ("hasta que dore").
- Ignora promociones, saludos y llamadas a seguir la cuenta.
- Si el video no presenta una receta: "is_recipe": false, explica en "description" qué muestra y \
deja las listas vacías.
- Escribe todo en {language}, aunque el video esté en otro idioma.

Responde SOLO con un objeto JSON con exactamente esta forma (sin texto antes ni después):
{{
  "is_recipe": true,
  "title": "nombre del plato",
  "description": "una o dos frases",
  "servings": "4 porciones" o null,
  "prep_time_minutes": número o null,
  "cook_time_minutes": número o null,
  "total_time_minutes": número o null,
  "difficulty": "fácil" | "media" | "difícil" | null,
  "ingredients": [{{"name": "harina", "quantity": "200", "unit": "g", "notes": null}}],
  "steps": [{{"number": 1, "instruction": "…", "duration_minutes": número o null, "timestamp": "mm:ss" o null}}],
  "equipment": ["…"],
  "tips": ["…"],
  "uncertainties": ["…"]
}}"""

_lock = threading.Lock()  # un solo modelo en memoria y una generación a la vez
_model: Any = None
_tokenizer: Any = None


def _load() -> tuple[Any, Any]:
    global _model, _tokenizer
    if _model is None:
        try:
            from mlx_lm import load
        except ImportError as e:  # p. ej. en un servidor Linux
            from .pipeline import PipelineError

            raise PipelineError("El motor local necesita un Mac con Apple Silicon (mlx-lm).") from e
        log.info("Cargando modelo local %s%s", LOCAL_MODEL, f" + {LOCAL_ADAPTER}" if LOCAL_ADAPTER else "")
        _model, _tokenizer = load(LOCAL_MODEL, adapter_path=LOCAL_ADAPTER)
    return _model, _tokenizer


def warmup() -> None:
    with _lock:
        _load()


def model_name() -> str:
    name = LOCAL_MODEL.split("/")[-1]
    return f"local:{name}" + ("+lora" if LOCAL_ADAPTER else "")


def _cut(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " […]"


def build_prompt(data: VideoData) -> str:
    from .pipeline import _fmt_ts

    return (
        f"Autor: {data.uploader or 'desconocido'}\n"
        f"Duración: {_fmt_ts(data.duration)}\n"
        f"Título: {data.title or '(sin título)'}\n\n"
        f"<descripcion_del_autor>\n{_cut(data.description, MAX_DESCRIPTION) or '(vacía)'}\n</descripcion_del_autor>\n\n"
        f"<transcripcion>\n{_cut(data.transcript, MAX_TRANSCRIPT) or '(el video no tiene audio hablado)'}\n</transcripcion>\n\n"
        f"<texto_en_pantalla>\n{_cut(data.ocr_text, MAX_OCR) or '(no se detectó texto)'}\n</texto_en_pantalla>\n\n"
        "Reconstruye la receta de este video en JSON."
    )


# ---------------------------------------------------------------- JSON tolerante

def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no hay JSON en la respuesta")
    return json.loads(text[start:end + 1])


def _str_or_none(v: Any) -> str | None:
    if v is None or v == "":
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v)


def _num_or_none(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"\d+(?:[.,]\d+)?", v)
        return float(m.group().replace(",", ".")) if m else None
    return None


def _str_list(v: Any) -> list[str]:
    return [str(x) for x in v if x] if isinstance(v, list) else []


def coerce(obj: dict) -> Recipe:
    """Corrige los despistes típicos de un modelo pequeño (números como texto, campos que faltan)."""
    ingredients = []
    for i in obj.get("ingredients") or []:
        if isinstance(i, str):
            i = {"name": i}
        if not isinstance(i, dict) or not i.get("name"):
            continue
        ingredients.append({
            "name": str(i["name"]), "quantity": _str_or_none(i.get("quantity")),
            "unit": _str_or_none(i.get("unit")), "notes": _str_or_none(i.get("notes")),
            "estimated": False,  # no mira la imagen: nunca estima cantidades
        })
    steps = []
    for n, s in enumerate(obj.get("steps") or [], 1):
        if isinstance(s, str):
            s = {"instruction": s}
        if not isinstance(s, dict) or not s.get("instruction"):
            continue
        steps.append({
            "number": n, "instruction": str(s["instruction"]),
            "duration_minutes": _num_or_none(s.get("duration_minutes")),
            "timestamp": _str_or_none(s.get("timestamp")),
        })
    difficulty = str(obj.get("difficulty") or "").strip().lower()
    return Recipe.model_validate({
        "is_recipe": bool(obj.get("is_recipe", bool(ingredients or steps))),
        "title": str(obj.get("title") or "Receta"),
        "description": str(obj.get("description") or ""),
        "servings": _str_or_none(obj.get("servings")),
        "prep_time_minutes": _num_or_none(obj.get("prep_time_minutes")),
        "cook_time_minutes": _num_or_none(obj.get("cook_time_minutes")),
        "total_time_minutes": _num_or_none(obj.get("total_time_minutes")),
        "difficulty": difficulty if difficulty in ("fácil", "media", "difícil") else None,
        "ingredients": ingredients,
        "steps": steps,
        "equipment": _str_list(obj.get("equipment")),
        "tips": _str_list(obj.get("tips")),
        "uncertainties": _str_list(obj.get("uncertainties")),
    })


# ---------------------------------------------------------------- análisis

def analyze_local(data: VideoData, language: str) -> Recipe:
    from mlx_lm import stream_generate

    from .pipeline import PipelineError

    with _lock:
        model, tokenizer = _load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
            {"role": "user", "content": build_prompt(data)},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text, last = "", None
        for last in stream_generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS):
            text += last.text

    data.model_used = model_name()
    if last is not None:
        data.input_tokens = last.prompt_tokens
        data.output_tokens = last.generation_tokens
        if last.finish_reason == "length":
            raise PipelineError("La receta salió demasiado larga para el modelo local; inténtalo de nuevo.")
    try:
        return coerce(_extract_json(text))
    except (ValueError, json.JSONDecodeError) as e:
        log.warning("El modelo local no devolvió JSON válido: %s\n%s", e, text[:500])
        raise PipelineError("El modelo local no devolvió una receta válida.") from e

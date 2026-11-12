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
- El texto en pantalla suele traer los ingredientes y cantidades cuando el video solo tiene música.
- Si el video no presenta una receta: "is_recipe": false y deja las listas vacías.
- Escribe todo en {language}, aunque el video esté en otro idioma.

Responde SOLO con un objeto JSON con exactamente esta forma (sin texto antes ni después):
{{
  "is_recipe": true,
  "title": "nombre del plato",
  "servings": "4 porciones" o null,
  "ingredients": [{{"name": "harina", "quantity": "200", "unit": "g", "notes": null}}],
  "steps": ["primer paso", "segundo paso"]
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


_NOISE_TAG = re.compile(r"\[(?!\d{1,2}:\d{2}\])[^\]\n]{1,25}\]")  # [Music], [Música], [음악]… (no los minutos)
_EMPTY_TS = re.compile(r"^\[\d{1,2}:\d{2}\]\s*$", re.M)


def clean_transcript(text: str) -> str:
    """Quita las etiquetas de música y descarta las «transcripciones» de videos sin voz.

    Whisper y los subtítulos automáticos inventan cosas cuando solo hay música ("You", "mm", "k").
    """
    text = _EMPTY_TS.sub("", _NOISE_TAG.sub("", text))
    text = re.sub(r"\n{2,}", "\n", text).strip()
    words = re.findall(r"[^\W\d_]{2,}", re.sub(r"\[\d{1,2}:\d{2}\]", "", text))
    return text if len(words) >= 8 else ""


def messages_for(data: VideoData, language: str) -> list[dict]:
    """Mensajes de chat del motor local. El entrenamiento usa esta misma función."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
        {"role": "user", "content": build_prompt(data)},
    ]


def build_prompt(data: VideoData) -> str:
    from .pipeline import _fmt_ts

    transcript = clean_transcript(data.transcript)

    return (
        f"Autor: {data.uploader or 'desconocido'}\n"
        f"Duración: {_fmt_ts(data.duration)}\n"
        f"Título: {data.title or '(sin título)'}\n\n"
        f"<descripcion_del_autor>\n{_cut(data.description, MAX_DESCRIPTION) or '(vacía)'}\n</descripcion_del_autor>\n\n"
        f"<transcripcion>\n{_cut(transcript, MAX_TRANSCRIPT) or '(el video no tiene voz, solo música)'}\n</transcripcion>\n\n"
        f"<texto_en_pantalla>\n{_cut(data.ocr_text, MAX_OCR) or '(no se detectó texto)'}\n</texto_en_pantalla>\n\n"
        "Reconstruye la receta de este video en JSON."
    )


# ---------------------------------------------------------------- ¿hay información suficiente?
# El modelo local no ve la imagen: si el video no dice ni escribe la receta, se la inventa.
# En Instagram es muy común «comenta RECETA y te la mando» con un video solo de música.

_QTY = re.compile(
    r"(?<![\w#])(\d+(?:[.,/]\d+)?|½|¼|¾|⅓|un[ao]?|media|medio)\s*"
    r"(g|gr|grs|gramos?|kg|kilos?|ml|l|litros?|tazas?|cups?|cucharad(?:a|ita)s?|cdas?|cdtas?|"
    r"tbsps?|tsps?|oz|onzas?|libras?|lbs?|dientes?|pizcas?|huevos?|eggs?)\b", re.I)
_CTA = re.compile(
    r"coment(a|á|e)|comment|te (la )?(mando|envío|envio|paso)|\bdm\b|link (in|en) (la )?bio|"
    r"mi (web|página|blog)|my (website|blog)|receta completa en", re.I)


def has_enough_info(data: VideoData) -> bool:
    """Hay voz, o la receta está escrita (descripción o pantalla) con al menos un par de cantidades."""
    if clean_transcript(data.transcript):
        return True
    written = f"{data.description}\n{data.ocr_text}"
    if len(_QTY.findall(written)) >= 3:
        return True
    ocr_words = re.findall(r"[^\W\d_]{3,}", re.sub(r"\[\d{1,2}:\d{2}\]", "", data.ocr_text or ""))
    if len(set(w.lower() for w in ocr_words)) >= 15:  # ingredientes escritos en pantalla, aunque sin cantidades
        return True
    lines = [l for l in written.splitlines() if re.match(r"\s*([-•*·▪️✔️]|\d+[.)])\s*\w", l)]
    return len(lines) >= 4


def no_info_message(data: VideoData) -> str:
    msg = ("Este video no trae la receta: no se explica en voz alta, no aparece escrita en pantalla "
           "y la descripción no la incluye.")
    if _CTA.search(data.description or ""):
        msg += " El autor la comparte aparte (por mensaje o en su web); revisa la descripción del video."
    return msg


# ---------------------------------------------------------------- JSON tolerante

def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no hay JSON en la respuesta")
    return json.loads(text[start:end + 1])


def _str_or_none(v: Any) -> str | None:
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a", "-")):
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


_UNITS = re.compile(
    r"(g|gr|grs|gramos?|kg|ml|l|litros?|tazas?|cups?|cucharad(?:a|ita)s?|cdas?|cdtas?|tbsps?|tsps?|"
    r"oz|onzas?|libras?|lbs?|dientes?|pizcas?|piezas?|latas?|sobres?|chorritos?)", re.I)
_NO_UNIT = {"porción", "porciones", "porcion", "each", "unidad", "unidades", "u", "pieces"}
_LEADING_QTY = re.compile(r"^(\d+(?:[.,/]\d+)?(?:\s+\d/\d|\s*[½¼¾⅓])?|½|¼|¾|un[ao]?|media|medio)\s+(.+)$", re.I)


def _fix_units(quantity: str | None, unit: str | None) -> tuple[str | None, str | None]:
    """Corrige «2 libras» + «libras», «5» + «porciones», «null» + «grs» y «1 cucharada» + null."""
    if unit and unit.lower() in _NO_UNIT:
        unit = None
    if quantity and unit and quantity.lower().endswith(" " + unit.lower()):
        quantity = quantity[: -len(unit)].strip()
    if quantity and not unit:
        m = _LEADING_QTY.match(quantity)
        if m and _UNITS.fullmatch(m.group(2)):
            quantity, unit = m.group(1), m.group(2)
    if not quantity and unit and _UNITS.fullmatch(unit):
        unit = None  # una unidad sin cantidad no dice nada
    return quantity, unit


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
        quantity, unit = _fix_units(_str_or_none(i.get("quantity")), _str_or_none(i.get("unit")))
        ingredients.append({
            "name": str(i["name"]), "quantity": quantity,
            "unit": unit, "notes": _str_or_none(i.get("notes")),
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


# ---------------------------------------------------------------- bucles de repetición
# Los modelos pequeños a veces repiten el mismo ingrediente o paso hasta agotar los tokens.
# Se detecta mientras genera, se corta la repetición y, si era en los ingredientes, se le
# pide que siga con los pasos.

_OBJ = re.compile(r"\{[^{}]*\}")
_STR = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def _ingredient_loop(text: str) -> int | None:
    """Dónde empieza la repetición si un mismo ingrediente sale 3 veces (2 puede ser legítimo)."""
    start = text.find('"ingredients"')
    if start < 0:
        return None
    seen: dict[str, list[int]] = {}
    for m in _OBJ.finditer(text, start):
        pos = seen.setdefault(_norm(m.group()), [])
        pos.append(m.start())
        if len(pos) >= 3:
            return pos[1]
    return None


def _step_loop(text: str) -> int | None:
    """Dónde empieza la repetición si un paso sale dos veces."""
    start = text.find("[", text.find('"steps"'))
    seen: set[str] = set()
    for m in _STR.finditer(text, max(start, 0)):
        key = _norm(m.group(1))
        if len(key) < 15:  # claves JSON y textos cortos
            continue
        if key in seen:
            return m.start()
        seen.add(key)
    return None


def generate_recipe_text(model: Any, tokenizer: Any, messages: list[dict], max_tokens: int) -> tuple[str, Any]:
    """Genera el JSON de la receta cortando los bucles. Devuelve el texto y la última respuesta de MLX."""
    from mlx_lm import stream_generate

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    text, last, budget = "", None, max_tokens
    while budget > 0:
        cut = None
        for last in stream_generate(model, tokenizer, prompt=prompt + text, max_tokens=budget):
            text += last.text
            budget -= 1
            if "}" in last.text or "," in last.text:
                cut = _step_loop(text) if '"steps"' in text else _ingredient_loop(text)
                if cut is not None:
                    break
        if cut is None:
            break
        text = text[:cut].rstrip().rstrip(",")
        log.info("El modelo local entró en bucle; corto la repetición")
        if '"steps"' in text:
            text += "]}"
            last = None  # la receta está completa: no contar como «demasiado larga»
            break
        text += '], "steps": ['  # que siga con los pasos
    return text, last


# ---------------------------------------------------------------- análisis

def analyze_local(data: VideoData, language: str) -> Recipe:
    from .pipeline import NoRecipeInfoError, PipelineError

    if not has_enough_info(data):
        data.model_used = model_name()
        raise NoRecipeInfoError(no_info_message(data))
    with _lock:
        model, tokenizer = _load()
        text, last = generate_recipe_text(model, tokenizer, messages_for(data, language), MAX_TOKENS)

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

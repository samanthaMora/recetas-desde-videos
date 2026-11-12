"""Estructura de la receta que devuelve el análisis.

El modelo Pydantic valida la respuesta; RECIPE_JSON_SCHEMA es el esquema que se
envía a Claude (structured outputs) para garantizar JSON válido con esta forma.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class Ingredient(BaseModel):
    name: str
    quantity: Optional[str]  # texto: "200", "1/2", "al gusto"
    unit: Optional[str]
    notes: Optional[str]
    estimated: bool  # True si la cantidad se dedujo viendo el video, no se dijo/escribió


class Step(BaseModel):
    number: int
    instruction: str
    duration_minutes: Optional[float]
    timestamp: Optional[str]  # "mm:ss" aproximado del video


class Recipe(BaseModel):
    is_recipe: bool
    title: str
    description: str
    servings: Optional[str]
    prep_time_minutes: Optional[float]
    cook_time_minutes: Optional[float]
    total_time_minutes: Optional[float]
    difficulty: Optional[Literal["fácil", "media", "difícil"]]
    ingredients: list[Ingredient]
    steps: list[Step]
    equipment: list[str]
    tips: list[str]
    uncertainties: list[str]  # cosas que no quedaron claras en el video


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def _obj(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STR = {"type": "string"}
_NUM = {"type": "number"}

RECIPE_JSON_SCHEMA = _obj(
    {
        "is_recipe": {"type": "boolean"},
        "title": _STR,
        "description": _STR,
        "servings": _nullable(_STR),
        "prep_time_minutes": _nullable(_NUM),
        "cook_time_minutes": _nullable(_NUM),
        "total_time_minutes": _nullable(_NUM),
        "difficulty": _nullable({"type": "string", "enum": ["fácil", "media", "difícil"]}),
        "ingredients": {
            "type": "array",
            "items": _obj(
                {
                    "name": _STR,
                    "quantity": _nullable(_STR),
                    "unit": _nullable(_STR),
                    "notes": _nullable(_STR),
                    "estimated": {"type": "boolean"},
                }
            ),
        },
        "steps": {
            "type": "array",
            "items": _obj(
                {
                    "number": {"type": "integer"},
                    "instruction": _STR,
                    "duration_minutes": _nullable(_NUM),
                    "timestamp": _nullable(_STR),
                }
            ),
        },
        "equipment": {"type": "array", "items": _STR},
        "tips": {"type": "array", "items": _STR},
        "uncertainties": {"type": "array", "items": _STR},
    }
)

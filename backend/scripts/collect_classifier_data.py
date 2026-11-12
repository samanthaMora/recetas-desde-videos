"""Reúne un dataset para el clasificador «¿es un video de receta?» (fase 4). Gratis: no usa Claude.

    cd backend && .venv/bin/python scripts/collect_classifier_data.py

Idea: «etiquetas débiles» (weak labels). En vez de etiquetar miles de videos a mano, la búsqueda
que trajo cada video decide su etiqueta: lo que sale al buscar «receta de pan casero» es receta (1),
lo que sale al buscar «resumen de fútbol» no lo es (0). Habrá algo de ruido (un video de noticias
sobre una receta, p. ej.); en el notebook se mide y se corrige.

Incluye negativos difíciles: videos de COMIDA que no son recetas (reseñas de restaurantes, mukbang,
tours de comida callejera). Sin ellos el modelo aprendería «habla de comida = receta».

Salida: data/classifier/raw.jsonl con {id, title, description, channel, query, label, kind}.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yt_dlp  # noqa: E402

from app import db  # noqa: E402

PER_QUERY = 25

POSITIVE = [
    "receta fácil", "cómo hacer pan casero", "recetas de pollo", "postres sin horno", "receta de pasta casera",
    "comida mexicana receta", "receta de pastel de chocolate", "sopa casera receta", "receta vegana fácil",
    "galletas caseras receta", "receta de empanadas", "cómo preparar arroz", "receta de pescado al horno",
    "desayuno saludable receta", "receta shorts cocina", "asado a la parrilla cómo hacer",
    "easy dinner recipe", "how to make pizza dough", "receita de bolo fácil", "receita de frango",
    "receta de flan", "cocina peruana receta", "receta de tortilla de papas", "salsa casera receta",
]

NEGATIVE = [
    "gameplay en español", "noticias de hoy", "tutorial de maquillaje", "rutina de ejercicios en casa",
    "reseña de celular", "vlog de viaje", "resumen de fútbol", "tutorial de excel", "storytime",
    "unboxing", "clase de matemáticas", "podcast entrevista", "cómo cambiar el aceite del auto",
    "tutorial de guitarra", "documental de animales", "music video", "stand up comedy",
    "cómo programar en python", "decoración de casa ideas", "reseña de película",
]

# Comida, pero no receta: los ejemplos que más enseñan al modelo.
HARD_NEGATIVE = [
    "reseña de restaurante", "probando comida callejera", "mukbang", "food tour", "probando comida rara",
    "comiendo en el mercado", "ranking de hamburguesas", "probando snacks de otros países",
    "la peor comida que he probado", "precios de comida en otro país",
]


def search(query: str) -> list[dict]:
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
        result = ydl.extract_info(f"ytsearch{PER_QUERY}:{query}", download=False)
    return [e for e in result.get("entries") or [] if e and e.get("id")]


def full_metadata(video_id: str) -> dict | None:
    """Descripción completa (la búsqueda solo trae un fragmento). No descarga el video."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        return {"description": (info.get("description") or "")[:3000], "channel": info.get("channel") or ""}
    except Exception:
        return None


def main() -> None:
    out_dir = db.DATA_DIR / "classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw.jsonl"

    items: dict[str, dict] = {}
    for kind, queries, label in (("receta", POSITIVE, 1), ("no_comida", NEGATIVE, 0), ("comida_no_receta", HARD_NEGATIVE, 0)):
        for q in queries:
            try:
                entries = search(q)
            except Exception as e:
                print(f"  búsqueda fallida «{q}»: {e}", file=sys.stderr)
                continue
            for e in entries:
                if e["id"] in items:
                    continue  # el mismo video en dos búsquedas: se queda la primera etiqueta
                items[e["id"]] = {
                    "id": e["id"], "title": e.get("title") or "", "description": e.get("description") or "",
                    "channel": e.get("channel") or e.get("uploader") or "", "query": q, "label": label, "kind": kind,
                }
            print(f"{kind:17} «{q}»: {len(entries)} videos (total {len(items)})", file=sys.stderr)

    print(f"Descargando descripciones completas de {len(items)} videos…", file=sys.stderr)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(full_metadata, vid): vid for vid in items}
        for fut in as_completed(futures):
            meta = fut.result()
            if meta:
                items[futures[fut]].update({k: v for k, v in meta.items() if v})
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(items)}", file=sys.stderr)

    with out_path.open("w", encoding="utf-8") as f:
        for item in items.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    pos = sum(i["label"] for i in items.values())
    print(f"Guardado {out_path}: {len(items)} videos ({pos} recetas, {len(items) - pos} no recetas)")


if __name__ == "__main__":
    main()

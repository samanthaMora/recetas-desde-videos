"""Exporta la base de datos a datasets JSONL para entrenar modelos propios.

    cd backend
    .venv/bin/python scripts/export_dataset.py            # todo
    .venv/bin/python scripts/export_dataset.py --human    # solo recetas corregidas por personas

Genera en data/dataset/:
  extraction.jsonl      entrada (título, descripción, transcripción, fotogramas) -> receta.
                        Para la fase 5: ajustar un modelo de lenguaje que extraiga recetas.
  classification.jsonl  título + descripción + inicio de la transcripción -> ¿es receta? (0/1).
                        Para la fase 4: el clasificador propio.

Cada ejemplo dice de dónde sale su etiqueta (target_source / label_source):
  "human"  = un usuario corrigió la receta  -> la etiqueta de más calidad
  "claude" = la generó Claude sin correcciones. Antes de entrenar con estas, revisa los
             términos comerciales de Anthropic sobre usar sus resultados para entrenar modelos.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="solo ejemplos corregidos por personas")
    parser.add_argument("--out", type=Path, default=db.DATA_DIR / "dataset")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    latest_correction = {
        r["analysis_id"]: r["recipe_json"]
        for r in db.iter_rows("SELECT analysis_id, recipe_json FROM corrections ORDER BY created_at")
    }

    stats: Counter = Counter()
    with (args.out / "extraction.jsonl").open("w", encoding="utf-8") as ext, \
         (args.out / "classification.jsonl").open("w", encoding="utf-8") as cls:
        for row in db.iter_rows("SELECT * FROM analyses WHERE status = 'done' ORDER BY created_at"):
            human = row["id"] in latest_correction
            if args.human and not human:
                continue
            target = json.loads(latest_correction.get(row["id"]) or row["recipe_json"])
            source = "human" if human else "claude"
            frames = [
                {"ts": f["ts"], "path": str(db.DATA_DIR / f["file"])}
                for f in json.loads(row["frames_json"] or "[]")
            ]

            ext.write(json.dumps({
                "id": row["id"],
                "url": row["url"],
                "platform": row["platform"],
                "language": row["language"],
                "title": row["title"],
                "description": row["description"],
                "transcript": row["transcript"],
                "transcript_source": row["transcript_source"],
                "frames": frames,
                "frame_method": row.get("frame_method"),
                "ocr_text": row.get("ocr_text"),
                "target": target,
                "target_source": source,
                "model": row["model"],
                "prompt_version": row["prompt_version"],
            }, ensure_ascii=False) + "\n")

            cls.write(json.dumps({
                "id": row["id"],
                "platform": row["platform"],
                "text": "\n".join(filter(None, [row["title"], row["description"], (row["transcript"] or "")[:1000]])),
                "label": int(bool(target.get("is_recipe"))),
                "label_source": source,
            }, ensure_ascii=False) + "\n")

            stats["total"] += 1
            stats[f"fuente_{source}"] += 1
            stats["recetas" if target.get("is_recipe") else "no_recetas"] += 1
            stats[f"plataforma_{row['platform']}"] += 1

    print(f"Dataset en {args.out}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

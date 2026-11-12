"""Prueba el motor desde la terminal:

    .venv/bin/python -m app.cli "https://www.youtube.com/shorts/..."
    .venv/bin/python -m app.cli "https://www.youtube.com/shorts/..." --motor local   # tu modelo, sin Claude
"""

import argparse
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from . import pipeline  # noqa: E402


def _fmt_ingredient(i) -> str:
    parts = [p for p in (i.quantity, i.unit, i.name) if p]
    line = " ".join(parts)
    if i.notes:
        line += f" ({i.notes})"
    if i.estimated:
        line += "  [estimado]"
    return line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("idioma", nargs="?", default="español")
    parser.add_argument("--motor", choices=["claude", "local"], help="por defecto, RECIPE_ENGINE del .env")
    args = parser.parse_args()

    start = time.time()
    try:
        recipe, data = pipeline.run(
            args.url, args.idioma, progress=lambda s: print(f"… {s}", file=sys.stderr), engine=args.motor
        )
    except pipeline.PipelineError as e:
        sys.exit(f"Error: {e}")

    print(f"\nFuente: {data.title} — {data.uploader} (transcripción: {data.transcript_source or 'ninguna'})")
    print(
        f"Fotogramas: {data.frame_count} ({data.frame_method}) · texto en pantalla: "
        f"{len(data.ocr_text.splitlines())} líneas · tokens: {data.input_tokens} entrada / {data.output_tokens} salida"
        f"\nModelo: {data.model_used} · {time.time() - start:.0f} s\n"
    )
    if not recipe.is_recipe:
        print("Este video no parece una receta:", recipe.description)
        return

    print(f"# {recipe.title}\n\n{recipe.description}\n")
    meta = {
        "Porciones": recipe.servings,
        "Preparación": recipe.prep_time_minutes and f"{recipe.prep_time_minutes:g} min",
        "Cocción": recipe.cook_time_minutes and f"{recipe.cook_time_minutes:g} min",
        "Dificultad": recipe.difficulty,
    }
    print(" · ".join(f"{k}: {v}" for k, v in meta.items() if v))
    print("\n## Ingredientes")
    for i in recipe.ingredients:
        print(f"- {_fmt_ingredient(i)}")
    print("\n## Pasos")
    for s in recipe.steps:
        print(f"{s.number}. {s.instruction}")
    for title, items in (("Utensilios", recipe.equipment), ("Consejos", recipe.tips), ("Dudas", recipe.uncertainties)):
        if items:
            print(f"\n## {title}")
            for x in items:
                print(f"- {x}")


if __name__ == "__main__":
    main()

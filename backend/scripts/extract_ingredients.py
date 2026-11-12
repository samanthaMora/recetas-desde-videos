"""Usa TU modelo (fase 5) para sacar los ingredientes de un video. Local y gratis: no llama a Claude.

    cd backend && .venv/bin/python scripts/extract_ingredients.py "https://www.youtube.com/watch?v=…"
    .venv/bin/python scripts/extract_ingredients.py "<enlace>" --base     # el modelo sin entrenar, para comparar

Solo lee los subtítulos (no descarga el video ni mira la imagen), así que no sustituye al análisis
completo de la app: da los ingredientes que se DICEN en el video. Necesita un Mac con Apple Silicon
(MLX); para un servidor Linux habría que fusionar el adaptador y convertir el modelo.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yt_dlp  # noqa: E402

from app import pipeline  # noqa: E402
import train_ingredients_llm as llm  # noqa: E402


def title_and_transcript(url: str) -> tuple[str, str]:
    base = {"quiet": True, "no_warnings": True, "skip_download": True, "noprogress": True}
    with tempfile.TemporaryDirectory() as tmp:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
        sub = pipeline._pick_subtitles(info)
        if not sub:
            sys.exit("Este video no tiene subtítulos; usa el análisis completo de la app.")
        lang, is_auto = sub
        opts = base | {
            "outtmpl": str(Path(tmp) / "v.%(ext)s"), "writesubtitles": not is_auto,
            "writeautomaticsub": is_auto, "subtitleslangs": [lang], "subtitlesformat": "vtt",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        vtts = list(Path(tmp).glob("*.vtt"))
        if not vtts:
            sys.exit("No pude descargar los subtítulos.")
        transcript = re.sub(r"^\[\d+:\d+\] ", "", pipeline.parse_vtt(vtts[0]), flags=re.M).replace("\n", " ")
    return info.get("title") or "", transcript


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--base", action="store_true", help="usar el modelo base sin tu entrenamiento")
    args = parser.parse_args()

    from mlx_lm import load

    title, transcript = title_and_transcript(pipeline.extract_url(args.url))
    adapter = None if args.base else llm.ADAPTER
    if adapter is not None and not (adapter / "adapters.safetensors").exists():
        sys.exit("Aún no hay modelo entrenado: ejecuta scripts/train_ingredients_llm.py entrenar")
    model, tokenizer = load(llm.BASE_MODEL, adapter_path=str(adapter) if adapter else None)
    prompt = tokenizer.apply_chat_template(llm.messages_for(title, transcript), add_generation_prompt=True, tokenize=False)
    start = time.time()
    items = llm.generate_list(model, tokenizer, prompt)
    print(f"\n{title}\n")
    for line in items:
        print(f"- {line}")
    print(f"\n({'modelo base' if args.base else 'tu modelo'} · {time.time() - start:.1f} s)")


if __name__ == "__main__":
    main()

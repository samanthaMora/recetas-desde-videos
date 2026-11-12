"""Dataset para la fase 5: transcripción del video -> lista de ingredientes. Gratis, sin Claude.

    cd backend && .venv/bin/python scripts/collect_ingredients_data.py            # usa los videos ya recogidos
    .venv/bin/python scripts/collect_ingredients_data.py --buscar                 # además busca videos nuevos

La etiqueta la puso una persona: muchos autores escriben la lista de ingredientes en la descripción
del video. Esa lista es la respuesta correcta (target). La entrada del modelo es lo que se DICE en el
video (subtítulos), sin la descripción: si no, el modelo solo aprendería a copiar.

Salida: data/ingredients/examples.jsonl con {id, title, channel, language, transcript, ingredients}.
Va despacio a propósito (YouTube bloquea si se piden muchos subtítulos seguidos) y se puede
reanudar: los videos ya procesados no se vuelven a pedir.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yt_dlp  # noqa: E402

from app import db, pipeline  # noqa: E402

RAW = db.DATA_DIR / "classifier" / "raw.jsonl"
OUT_DIR = db.DATA_DIR / "ingredients"
OUT = OUT_DIR / "examples.jsonl"
SEEN = OUT_DIR / "seen.json"  # ids ya intentados (con o sin éxito)

EXTRA_QUERIES = [
    "receta ingredientes en la descripción", "receta paso a paso ingredientes", "receta de galletas ingredientes",
    "receta de pan ingredientes", "receta de pastel ingredientes", "receta de pollo ingredientes",
    "receta de carne ingredientes", "receta de sopa ingredientes", "receta de postre ingredientes",
    "receta de salsa ingredientes", "receta de arroz ingredientes", "receta de ensalada ingredientes",
    "receta de pescado ingredientes", "receta de tacos ingredientes", "receta de pizza ingredientes",
    "receta de brownies", "receta de cheesecake", "receta de lasaña", "receta de albóndigas", "receta de tamales",
    "receta de mole", "receta de pozole", "receta de arepas", "receta de empanadas argentinas", "receta de paella",
    "receta de croquetas", "receta de tortilla española", "receta de ceviche", "receta de lomo saltado",
    "receta de alfajores", "receta de churros", "receta de tres leches", "receta de pan de muerto",
    "recipe ingredients in description", "easy cookie recipe", "banana bread recipe", "chicken curry recipe",
    "homemade pasta sauce recipe", "chocolate cake recipe", "beef stew recipe", "pancake recipe",
]

# ---------------------------------------------------------------- lista de ingredientes

_HEADER = re.compile(r"ingredientes?\b|ingredients?\b", re.I)
_STOP = re.compile(
    r"preparaci[oó]n|procedimiento|instrucciones|elaboraci[oó]n|\bpasos\b|modo de prepar|method\b|instructions|"
    r"directions|\bnotas?\b|\bnotes?\b|s[ií]gue|suscr[ií]b|subscribe|redes sociales|instagram|facebook|tiktok|"
    r"https?://|www\.|cap[ií]tulos|chapters|m[uú]sica|music|contacto|utensilios|equipment|\d{1,2}:\d{2}",
    re.I,
)
_BULLET = re.compile(r"^[\s\-–—•*·▢▪◦●○✅✔️☑️🔸🔹🔶👉➡️►▶︎✨🟢🟠⭐️]+")
_QTY = re.compile(
    r"^(\d+[\d/.,\s]*|½|¼|¾|⅓|⅔|un|una|uno|one|a|media|medio|half)\s*"
    r"(g|gr|grs|gramos?|kg|kilos?|ml|l|lt|litros?|tazas?|cups?|cdas?|cdtas?|cucharadas?|cucharaditas?|tbsp|tsp|"
    r"oz|lb|lbs|libras?|piezas?|pzas?|dientes?|ramas?|latas?|sobres?|pizcas?|x[ií]caras?|colher(es)?|"
    r"tablespoons?|teaspoons?|cloves?|pinch|sticks?|unidades?)?\b",
    re.I,
)


_COMPILATION = re.compile(r"^\s*\d+\s.*\b(recetas|recipes|receitas|ideas|postres|desserts)\b", re.I)


def _clean(line: str) -> str:
    line = _BULLET.sub("", line).strip()
    return re.sub(r"\s+", " ", line)


def parse_ingredients(description: str) -> list[str]:
    """Líneas de la lista de ingredientes que el autor escribió en la descripción ([] si no hay)."""
    lines = description.splitlines()
    start = next((i for i, l in enumerate(lines) if _HEADER.search(l) and len(l) < 60), None)

    found: list[str] = []
    if start is not None:
        blanks = 0
        for raw in lines[start + 1:]:
            line = _clean(raw)
            if not line:
                blanks += 1
                if blanks >= 2 and len(found) >= 3:
                    break
                continue
            if not re.search(r"[^\W_]", line):  # separadores: "______", "-----"
                if found:
                    break
                continue
            if _STOP.search(line) or len(line) > 90:
                if found:
                    break
                continue
            blanks = 0
            if line.endswith(":") or line.isupper() and len(line.split()) <= 4:
                continue  # subtítulos: "Para la masa:", "RELLENO"
            found.append(line)
    else:
        # Sin encabezado: el bloque más largo de líneas seguidas que empiezan por una cantidad.
        block: list[str] = []
        for raw in lines + [""]:
            line = _clean(raw)
            if line and _QTY.match(line) and len(line) < 90:
                block.append(line)
            else:
                if len(block) > len(found):
                    found = block
                block = []

    with_qty = sum(1 for l in found if _QTY.match(l))
    if not (4 <= len(found) <= 40) or with_qty < max(3, len(found) // 2):
        return []
    return found


# ---------------------------------------------------------------- YouTube


def fetch(video_id: str) -> dict | None:
    """Metadatos + subtítulos (sin descargar el video). None si no hay subtítulos o falla."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    base = {"quiet": True, "no_warnings": True, "skip_download": True, "noprogress": True}
    with tempfile.TemporaryDirectory() as tmp:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
        ingredients = parse_ingredients(info.get("description") or "")
        if not ingredients:
            return None
        # Recopilaciones ("10 recetas de…") mezclan varias listas; videos largos no caben en el modelo.
        if (info.get("duration") or 0) > 900 or _COMPILATION.search(info.get("title") or ""):
            return None
        sub = pipeline._pick_subtitles(info)
        if not sub:
            return None
        lang, is_auto = sub
        opts = base | {
            "outtmpl": str(Path(tmp) / "v.%(ext)s"),
            "writesubtitles": not is_auto, "writeautomaticsub": is_auto,
            "subtitleslangs": [lang], "subtitlesformat": "vtt",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        vtts = list(Path(tmp).glob("*.vtt"))
        if not vtts:
            return None
        transcript = re.sub(r"^\[\d+:\d+\] ", "", pipeline.parse_vtt(vtts[0]), flags=re.M).replace("\n", " ")
    if len(transcript) < 200:
        return None  # videos solo con música
    return {
        "id": video_id, "title": info.get("title") or "", "channel": info.get("channel") or "",
        "language": lang.split("-")[0], "duration": info.get("duration") or 0,
        "transcript": transcript, "ingredients": ingredients,
    }


def search_ids(queries: list[str], per_query: int) -> list[str]:
    ids: list[str] = []
    for q in queries:
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
                result = ydl.extract_info(f"ytsearch{per_query}:{q}", download=False)
            ids += [e["id"] for e in result.get("entries") or [] if e and e.get("id") and (e.get("duration") or 0) <= 900]
        except Exception as e:
            print(f"  búsqueda fallida «{q}»: {e}", file=sys.stderr)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buscar", action="store_true", help="buscar videos nuevos además de los ya recogidos")
    parser.add_argument("--por-busqueda", type=int, default=30)
    parser.add_argument("--pausa", type=float, default=1.5, help="segundos entre videos")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set(json.loads(SEEN.read_text())) if SEEN.exists() else set()

    # 1) Videos ya recogidos para el clasificador cuya descripción trae lista de ingredientes.
    candidates = [
        item["id"] for item in map(json.loads, RAW.read_text(encoding="utf-8").splitlines())
        if item["label"] == 1 and parse_ingredients(item["description"])
    ]
    # 2) Opcional: videos nuevos (la descripción se comprueba al pedir cada uno).
    if args.buscar:
        candidates += search_ids(EXTRA_QUERIES, args.por_busqueda)
    todo = [v for v in dict.fromkeys(candidates) if v not in seen]
    print(f"{len(todo)} videos por procesar ({len(seen)} ya vistos)", file=sys.stderr)

    ok = blocked = 0
    with OUT.open("a", encoding="utf-8") as out:
        for n, video_id in enumerate(todo, 1):
            try:
                example = fetch(video_id)
                blocked = 0
            except Exception as e:
                example = None
                if "not a bot" in str(e) or "429" in str(e):
                    blocked += 1
                    if blocked >= 5:
                        print("YouTube está bloqueando las peticiones; paro aquí. Vuelve a ejecutar más tarde.", file=sys.stderr)
                        break
                    time.sleep(30)
                    continue  # no se marca como visto: se reintenta la próxima vez
            seen.add(video_id)
            if example:
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                out.flush()
                ok += 1
            if n % 10 == 0:
                SEEN.write_text(json.dumps(sorted(seen)))
                print(f"  {n}/{len(todo)} · {ok} ejemplos nuevos", file=sys.stderr)
            time.sleep(args.pausa)
    SEEN.write_text(json.dumps(sorted(seen)))
    total = sum(1 for _ in OUT.open(encoding="utf-8"))
    print(f"Listo: {ok} ejemplos nuevos · {total} en total en {OUT}")


if __name__ == "__main__":
    main()

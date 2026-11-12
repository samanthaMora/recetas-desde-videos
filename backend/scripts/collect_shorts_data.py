"""Dataset de videos cortos (formato Reels/TikTok) con la receta completa. Gratis, sin Claude.

    cd backend
    .venv/bin/python scripts/collect_shorts_data.py canales            # 1) busca y puntúa canales
    .venv/bin/python scripts/collect_shorts_data.py recoger --objetivo 300   # 2) recoge ejemplos

TikTok e Instagram no dejan buscar videos en masa sin iniciar sesión, así que se usan YouTube Shorts:
mismo formato (vertical, < 3 min, música, texto en pantalla) y a menudo los mismos creadores.

La respuesta correcta (target) es la receta que el autor escribió en la descripción: ingredientes
y pasos. La entrada es lo que la app ve SIN esa descripción: la voz (subtítulos o Whisper), el
texto en pantalla (OCR) y los fotogramas (guardados para la futura fase de visión).

Salida en data/shorts/: examples.jsonl, frames/<id>/*.jpg, channels.json y seen.json (reanudable:
lo ya visto no se vuelve a pedir). Los videos se borran nada más procesarlos.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yt_dlp  # noqa: E402

from app import db, pipeline  # noqa: E402
from collect_ingredients_data import _COMPILATION, _clean, parse_ingredients  # noqa: E402

OUT_DIR = db.DATA_DIR / "shorts"
EXAMPLES = OUT_DIR / "examples.jsonl"
FRAMES = OUT_DIR / "frames"
CHANNELS = OUT_DIR / "channels.json"
SEEN = OUT_DIR / "seen.json"
RAW = db.DATA_DIR / "classifier" / "raw.jsonl"

MAX_SECONDS = 180
SEARCH_QUERIES = [
    "receta rápida shorts", "receta fácil shorts", "receta casera shorts", "receta en 1 minuto",
    "postre fácil shorts", "receta de pollo shorts", "receta saludable shorts", "receta sin horno shorts",
    "receta mexicana shorts", "receta de cena fácil shorts", "receta de desayuno shorts", "receta de pasta shorts",
]
BASE = {"quiet": True, "no_warnings": True, "noprogress": True}

# ---------------------------------------------------------------- pasos de la receta

_PREP_HEADER = re.compile(
    r"preparaci[oó]n|procedimiento|instrucciones|elaboraci[oó]n|modo de prepar|\bpasos\b|paso a paso|"
    r"c[oó]mo (se )?hac|method|instructions|directions|how to make|modo de fazer|preparo",
    re.I,
)
_STEP_STOP = re.compile(
    r"^#|https?://|www\.|s[ií]gue(me|nos)|suscr[ií]b|subscribe|redes sociales|instagram|facebook|tiktok|"
    r"\bnotas?\b:?$|\bnotes?\b:?$|m[uú]sica|music|contacto|colaboraci|patrocin|sponsor|\d{1,2}:\d{2}",
    re.I,
)
_NUMBERED = re.compile(
    r"^\(?\d{1,2}\s*[.)\-:]\s*|^\d{1,2}\s+(?=[A-ZÁÉÍÓÚÑ¡¿])|^paso\s*\d+\s*[.:\-]?\s*|^step\s*\d+\s*[.:\-]?\s*", re.I
)
_EMOJI = re.compile(r"[\U0001F000-\U0001FFFF\u2190-\u21FF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u20E3\u200B-\u200F\u2060-\u206F]")
_CTA = re.compile(r"gu[aá]rda|coment|compart|dale (al|like|me gusta)|like|suscr|s[ií]gue|link|enlace|descarga|save|share", re.I)
_OUTRO = re.compile(r"^¡?\s*(listo|disfruta|buen provecho|a disfrutar|enjoy)\b", re.I)


def _no_emoji(text: str) -> str:
    return re.sub(r"\s+", " ", _EMOJI.sub("", text)).strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¡¿])", text)
    return [p.strip() for p in parts if p.strip()]


def parse_steps(description: str) -> list[str]:
    """Pasos de preparación escritos por el autor ([] si no hay)."""
    lines = description.splitlines()
    start = next((i for i, l in enumerate(lines) if _PREP_HEADER.search(l) and len(l) < 60), None)
    if start is None:
        return []
    steps: list[str] = []
    blanks = 0
    for raw in lines[start + 1:]:
        line = _no_emoji(_clean(raw))
        if not line:
            blanks += 1
            if blanks >= 2 and steps:
                break
            continue
        blanks = 0
        if _STEP_STOP.search(line) or _OUTRO.match(line) or re.match(r"ingredientes?\b|ingredients?\b", line, re.I):
            if steps:
                break
            continue
        if not re.search(r"[^\W\d_]{3}", line):  # separadores, emojis sueltos
            continue
        if line.endswith(":") and len(line) < 50:
            continue  # subtítulos: "Para la salsa:"
        line = _NUMBERED.sub("", line).strip()
        steps += _split_sentences(line) if len(line) > 600 else [line]
    if len(steps) == 1:  # toda la preparación en un solo párrafo
        steps = _split_sentences(steps[0])
    steps = [s for s in steps if 8 <= len(s) <= 600]
    return steps if 2 <= len(steps) <= 25 else []


def parse_recipe(description: str) -> tuple[list[str], list[str]]:
    ingredients = [_no_emoji(l) for l in parse_ingredients(description)]
    ingredients = [
        l for l in ingredients
        if l and not _CTA.search(l) and not re.match(r"ingredientes?\b|ingredients?\b", l, re.I)
    ]
    if len(ingredients) < 3:
        ingredients = []
    return ingredients, parse_steps(description)


# ---------------------------------------------------------------- YouTube

def info_of(video_id: str) -> dict:
    with yt_dlp.YoutubeDL(BASE | {"skip_download": True}) as ydl:
        return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)


def shorts_of(channel_url: str, limit: int) -> list[str]:
    url = channel_url.rstrip("/") + "/shorts"
    with yt_dlp.YoutubeDL(BASE | {"extract_flat": True, "playlistend": limit}) as ydl:
        result = ydl.extract_info(url, download=False)
    return [e["id"] for e in result.get("entries") or [] if e and e.get("id")]


def _blocked(error: Exception) -> bool:
    return "not a bot" in str(error) or "429" in str(error)


# ---------------------------------------------------------------- 1) canales

def find_channels(sample: int, pause: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    channels: dict[str, dict] = json.loads(CHANNELS.read_text()) if CHANNELS.exists() else {}

    # a) canales que ya escriben ingredientes en sus videos largos (dataset del clasificador)
    seeds: dict[str, str] = {}
    for item in map(json.loads, RAW.read_text(encoding="utf-8").splitlines()):
        if str(item["label"]) == "1" and item.get("channel") and parse_ingredients(item["description"]):
            seeds.setdefault(item["channel"], item["id"])
    candidates: dict[str, str] = {}  # channel_url -> nombre
    for name, video_id in seeds.items():
        if any(c["name"] == name for c in channels.values()):
            continue
        try:
            info = info_of(video_id)
            if info.get("channel_url"):
                candidates[info["channel_url"]] = name
        except Exception as e:
            print(f"  {name}: {str(e)[:80]}", file=sys.stderr)
        time.sleep(pause)

    # b) canales que aparecen al buscar Shorts de recetas
    for q in SEARCH_QUERIES:
        try:
            with yt_dlp.YoutubeDL(BASE | {"extract_flat": True}) as ydl:
                result = ydl.extract_info(f"ytsearch40:{q}", download=False)
            for e in result.get("entries") or []:
                if e and e.get("channel_url") and (e.get("duration") or 999) <= MAX_SECONDS:
                    candidates.setdefault(e["channel_url"], e.get("channel") or "")
        except Exception as e:
            print(f"  búsqueda «{q}»: {str(e)[:80]}", file=sys.stderr)
    candidates = {u: n for u, n in candidates.items() if u not in channels}
    print(f"{len(candidates)} canales por puntuar", file=sys.stderr)

    # c) puntuar: de unos pocos Shorts de cada canal, ¿cuántos traen ingredientes y pasos?
    for n, (url, name) in enumerate(candidates.items(), 1):
        stats = {"name": name, "shorts": 0, "sampled": 0, "ingredients": 0, "full": 0}
        try:
            ids = shorts_of(url, 30)
            stats["shorts"] = len(ids)
            for video_id in ids[:: max(1, len(ids) // sample)][:sample]:
                info = info_of(video_id)
                ingredients, steps = parse_recipe(info.get("description") or "")
                stats["sampled"] += 1
                stats["ingredients"] += bool(ingredients)
                stats["full"] += bool(ingredients and steps)
                time.sleep(pause)
        except Exception as e:
            if _blocked(e):
                print("YouTube está bloqueando; paro aquí. Vuelve a ejecutar más tarde.", file=sys.stderr)
                break
            stats["error"] = str(e)[:120]
        channels[url] = stats
        CHANNELS.write_text(json.dumps(channels, indent=1, ensure_ascii=False))
        print(f"  {n}/{len(candidates)} {name[:30]:30} shorts={stats['shorts']:3} "
              f"ingredientes={stats['ingredients']}/{stats['sampled']} completas={stats['full']}", file=sys.stderr)

    good = sorted((c for c in channels.values() if c.get("full")), key=lambda c: -c["full"])
    print(f"Listo: {len(good)} canales con recetas completas en sus Shorts")


# ---------------------------------------------------------------- 2) recoger

def process(video_id: str, info: dict, ingredients: list[str], steps: list[str]) -> dict:
    """Descarga el Short, saca voz + texto en pantalla + fotogramas y arma el ejemplo."""
    from app import vision

    data = pipeline.VideoData(url=f"https://www.youtube.com/watch?v={video_id}")
    with tempfile.TemporaryDirectory(prefix="short_") as tmp:
        workdir = Path(tmp)
        video = pipeline.download(data.url, workdir, data, force=True)
        subs = list(workdir.glob("*.vtt"))
        if subs:
            data.transcript, data.transcript_source = pipeline.parse_vtt(subs[0]), "subtítulos"
        elif pipeline.has_audio(video):
            data.transcript, data.transcript_source = pipeline.transcribe_with_whisper(video), "whisper"
        frames, method = pipeline.choose_frames(video, data.duration, workdir)
        ocr = vision.read_text(frames, pipeline._fmt_ts)
        dest = FRAMES / video_id
        dest.mkdir(parents=True, exist_ok=True)
        kept = []
        for ts, path in frames:
            shutil.copy2(path, dest / path.name)
            kept.append({"ts": round(ts, 2), "file": f"frames/{video_id}/{path.name}"})
    return {
        "id": video_id, "title": info.get("title") or "", "channel": info.get("channel") or "",
        "duration": info.get("duration") or 0, "language": (info.get("language") or "").split("-")[0],
        "transcript": data.transcript, "transcript_source": data.transcript_source,
        "ocr_text": ocr, "frames": kept, "frame_method": method,
        "description": info.get("description") or "",  # se guarda, pero NO se usa como entrada
        "ingredients": ingredients, "steps": steps,
    }


def collect(target: int, per_channel: int, max_per_channel: int, min_full: int, pause: float) -> None:
    from app import vision

    if not CHANNELS.exists():
        sys.exit("Primero ejecuta: collect_shorts_data.py canales")
    channels = json.loads(CHANNELS.read_text())
    good = sorted(((u, c) for u, c in channels.items() if c.get("full", 0) >= min_full), key=lambda x: -x[1]["full"])
    seen: set[str] = set(json.loads(SEEN.read_text())) if SEEN.exists() else set()
    previous = [json.loads(l)["channel"] for l in EXAMPLES.open(encoding="utf-8")] if EXAMPLES.exists() else []
    done = len(previous)
    print(f"{len(good)} canales · {done} ejemplos ya recogidos · objetivo {target}", file=sys.stderr)
    vision.warmup()

    blocked = 0
    started = time.time()
    with EXAMPLES.open("a", encoding="utf-8") as out:
        for url, stats in good:
            if done >= target:
                break
            try:
                ids = [v for v in shorts_of(url, per_channel) if v not in seen]
            except Exception as e:
                print(f"  {stats['name']}: {str(e)[:80]}", file=sys.stderr)
                continue
            got = previous.count(stats["name"])  # al reanudar, cuenta lo ya recogido de este canal
            for video_id in ids:
                if done >= target or got >= max_per_channel:
                    break
                try:
                    info = info_of(video_id)
                    blocked = 0
                    ingredients, steps = parse_recipe(info.get("description") or "")
                    ok = (ingredients and steps and (info.get("duration") or 999) <= MAX_SECONDS
                          and not _COMPILATION.search(info.get("title") or ""))
                    if ok:
                        out.write(json.dumps(process(video_id, info, ingredients, steps), ensure_ascii=False) + "\n")
                        out.flush()
                        done += 1
                        got += 1
                except Exception as e:
                    if _blocked(e):
                        blocked += 1
                        if blocked >= 5:
                            SEEN.write_text(json.dumps(sorted(seen)))
                            sys.exit("YouTube está bloqueando las peticiones; vuelve a ejecutar más tarde.")
                        time.sleep(60)
                        continue  # no se marca como visto
                    print(f"  {video_id}: {str(e)[:80]}", file=sys.stderr)
                seen.add(video_id)
                if len(seen) % 10 == 0:
                    SEEN.write_text(json.dumps(sorted(seen)))
                time.sleep(pause)
            hours = (time.time() - started) / 3600
            print(f"  {stats['name'][:30]:30} +{got} · total {done}/{target} · {hours:.1f} h", file=sys.stderr)
    SEEN.write_text(json.dumps(sorted(seen)))
    print(f"Listo: {done} ejemplos en {EXAMPLES}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("canales", help="buscar y puntuar canales")
    c.add_argument("--muestra", type=int, default=4, help="Shorts que se miran por canal")
    c.add_argument("--pausa", type=float, default=1.5)
    r = sub.add_parser("recoger", help="recoger ejemplos de los mejores canales")
    r.add_argument("--objetivo", type=int, default=300)
    r.add_argument("--por-canal", type=int, default=200, help="máximo de Shorts que se miran por canal")
    r.add_argument("--max-por-canal", type=int, default=30, help="ejemplos máximos por canal (variedad)")
    r.add_argument("--minimo", type=int, default=1, help="recetas completas mínimas en la muestra del canal")
    r.add_argument("--pausa", type=float, default=1.5)
    args = parser.parse_args()
    if args.cmd == "canales":
        find_channels(args.muestra, args.pausa)
    else:
        collect(args.objetivo, args.por_canal, args.max_por_canal, args.minimo, args.pausa)


if __name__ == "__main__":
    main()

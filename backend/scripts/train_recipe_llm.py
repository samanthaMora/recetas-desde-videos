"""Tu modelo de recetas completas (ingredientes + pasos) para el motor local. MLX, sin Claude.

    cd backend
    .venv/bin/python scripts/train_recipe_llm.py preparar     # data/shorts/examples.jsonl -> formato chat
    .venv/bin/python scripts/train_recipe_llm.py entrenar     # LoRA sobre Qwen2.5-3B
    .venv/bin/python scripts/train_recipe_llm.py evaluar      # 3B base vs 3B + LoRA vs 7B base (el actual)

Para usarlo en la app (backend/.env):
    RECIPE_ENGINE=local
    LOCAL_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit
    LOCAL_ADAPTER=models/recipe_lora

Datos: YouTube Shorts cuyo autor escribió la receta en la descripción (scripts/collect_shorts_data.py).
- Entrada: exactamente lo que ve el motor local en la app (app/local_llm.messages_for): título, voz,
  texto en pantalla y la descripción. En el 80 % de los ejemplos se le QUITA la receta a la
  descripción (tiene que sacarla del video); en el 20 % se deja (aprende a aprovecharla).
- Salida: el JSON del motor local con la receta del autor.
- División por canal: train / valid / test sin canales compartidos.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import db, local_llm  # noqa: E402
from app.pipeline import VideoData  # noqa: E402
from collect_shorts_data import _PREP_HEADER, _clean, _no_emoji  # noqa: E402
import train_ingredients_llm as ing  # noqa: E402

BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
REFERENCE_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"  # el que usa hoy la app
EXAMPLES = db.DATA_DIR / "shorts" / "examples.jsonl"
MLX_DATA = db.DATA_DIR / "shorts" / "mlx"
ADAPTER = Path(__file__).resolve().parent.parent / "models" / "recipe_lora"
KEEP_DESCRIPTION = 0.2
SEED = 42

# ---------------------------------------------------------------- receta del autor -> JSON

_NUM = r"(?:\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?(?:\s*(?:-|a|o)\s*\d+(?:[.,]\d+)?)?|\d+\s+\d/\d|½|¼|¾|⅓|⅔|una?|uno|media|medio|dos|tres)"
_UNIT = (
    r"(?:kg|kilos?|g|gr|grs|gramos?|mg|ml|cc|cl|dl|l|lt|litros?|tazas?|cups?|cdas?|cdtas?|chtas?|cucharadas?(?: soperas?)?|"
    r"cucharaditas?|cdita|tbsp|tsp|oz|lb|lbs|libras?|piezas?|pzas?|dientes?|ramas?|ramitas?|latas?|sobres?|pizcas?|"
    r"vasos?|chorritos?|chorros?|puñados?|hojas?|rebanadas?|lonchas?|trozos?|unidades?|uds?|botes?|paquetes?|tarros?|"
    r"tablespoons?|teaspoons?|cloves?|pinch|cans?|slices?|sticks?)"
)
_LINE = re.compile(rf"^(?P<q>{_NUM})\s*(?:(?P<u>{_UNIT})\.?(?=\s|$))?\s*(?:de\s+|del\s+|of\s+)?(?P<n>.*)$", re.I)
_SERVINGS = re.compile(r"para\s+(\d+)\s*(personas|porciones|raciones|comensales|unidades)|(\d+)\s*(porciones|raciones|servings)", re.I)


def ingredient_json(line: str) -> dict:
    """'200 g de harina (tamizada)' -> {name: harina, quantity: 200, unit: g, notes: tamizada}."""
    notes = None
    m = re.search(r"\(([^)]*)\)", line)
    if m:
        notes = m[1].strip() or None
        line = (line[:m.start()] + line[m.end():]).strip()
    if ":" in line:  # "Harina: 200 g" -> "200 g Harina"
        left, right = (x.strip() for x in line.split(":", 1))
        if re.match(_NUM, right, re.I) and left:
            line = f"{right} {left}"
    m = _LINE.match(line)
    if m and m["n"].strip():
        return {"name": m["n"].strip(" ,.-"), "quantity": m["q"], "unit": m["u"], "notes": notes}
    return {"name": line.strip(" ,.-"), "quantity": None, "unit": None, "notes": notes}


def clean_title(title: str) -> str:
    title = _no_emoji(re.sub(r"#\S+", "", title))
    title = re.split(r"\s[|•]\s", title)[0]
    return title.strip(" !¡?¿.-:") or "Receta"


def target_json(ex: dict) -> str:
    m = _SERVINGS.search(ex["description"])
    servings = f"{m[1] or m[3]} {(m[2] or m[4]).lower()}" if m else None
    return json.dumps({
        "is_recipe": True,
        "title": clean_title(ex["title"]),
        "servings": servings,
        "ingredients": [ingredient_json(i) for i in ex["ingredients"] if not re.search(r"@|https?://|www\.", i)],
        "steps": ex["steps"],
    }, ensure_ascii=False)


def strip_recipe(description: str, ex: dict) -> str:
    """La descripción sin las líneas de la receta (encabezados, ingredientes y pasos)."""
    items = [x.lower() for x in ex["ingredients"] + ex["steps"]]
    kept = []
    for raw in description.splitlines():
        line = _no_emoji(_clean(raw)).lower()
        is_header = len(line) < 60 and (re.search(r"ingredientes?\b|ingredients?\b", line) or _PREP_HEADER.search(line))
        is_item = len(line) > 2 and any(line in it or it in line for it in items)
        is_numbered_step = re.match(r"^\d{1,2}\s*[.)\-:]?\s+\S", line) and (len(line) > 30 or len(line) < 40)
        if not (is_header or is_item or is_numbered_step):
            kept.append(raw)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _english(ex: dict) -> bool:
    text = " ".join(ex["ingredients"] + ex["steps"]).lower()
    en = len(re.findall(r"\b(the|and|with|cup|cups|until|add|into)\b", text))
    es = len(re.findall(r"\b(de|la|el|con|y|hasta|añade|agrega|en)\b", text))
    return en > es


def video_data(ex: dict, keep_description: bool) -> VideoData:
    return VideoData(
        url=f"https://www.youtube.com/watch?v={ex['id']}", title=ex["title"], uploader=ex["channel"],
        duration=ex["duration"], transcript=ex["transcript"], ocr_text=ex["ocr_text"],
        description=ex["description"] if keep_description else strip_recipe(ex["description"], ex),
    )


def language_of(ex: dict) -> str:
    return "inglés" if _english(ex) else "español"


# ---------------------------------------------------------------- preparar

def split_by_channel(examples: list[dict]) -> dict[str, list[dict]]:
    """Reparte canales enteros. El test debe incluir videos sin voz (solo música), como muchos Reels."""
    channels = sorted({e["channel"] for e in examples})
    size = {c: sum(1 for e in examples if e["channel"] == c) for c in channels}
    mute = {c: sum(1 for e in examples if e["channel"] == c and not local_llm.clean_transcript(e["transcript"]))
            for c in channels}
    for seed in range(SEED, SEED + 500):
        order = channels[:]
        random.Random(seed).shuffle(order)
        split, counts = {}, {"test": 0, "valid": 0}
        for c in order:
            name = ("test" if counts["test"] < 0.17 * len(examples)
                    else "valid" if counts["valid"] < 0.12 * len(examples) else "train")
            split[c] = name
            if name in counts:
                counts[name] += size[c]
        test_mute = sum(mute[c] for c in channels if split[c] == "test") / max(1, counts["test"])
        if 0.2 <= test_mute <= 0.5 and counts["valid"] > 0:
            break
    out = {"train": [], "valid": [], "test": []}
    for e in examples:
        out[split[e["channel"]]].append(e)
    return out


def prepare() -> None:
    examples = [json.loads(l) for l in EXAMPLES.read_text(encoding="utf-8").splitlines()]
    parts = split_by_channel(examples)
    rng = random.Random(SEED)
    MLX_DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        with (MLX_DATA / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for ex in rows:
                # En test nunca se deja la receta en la descripción: se mide lo difícil.
                keep = name != "test" and rng.random() < KEEP_DESCRIPTION
                msgs = local_llm.messages_for(video_data(ex, keep), language_of(ex))
                msgs.append({"role": "assistant", "content": target_json(ex)})
                f.write(json.dumps({"id": ex["id"], "messages": msgs}, ensure_ascii=False) + "\n")
        chans = sorted({e["channel"] for e in rows})
        no_voice = sum(1 for e in rows if not local_llm.clean_transcript(e["transcript"]))
        print(f"{name:5} {len(rows):3} ejemplos · {len(chans)} canales · {no_voice} sin voz · {', '.join(chans)}")


# ---------------------------------------------------------------- evaluar

_STEP_STOP = ing._STOP | {"que", "se", "lo", "le", "por", "hasta", "unos", "unas", "minutos", "min", "bien", "todo", "muy"}


def step_tokens(steps: list[str]) -> set[str]:
    text = ing._fold(" ".join(steps))
    return {ing._singular(t) for t in re.findall(r"[a-z]{3,}", text) if t not in _STEP_STOP}


def ingredient_line(i) -> str:
    return " ".join(x for x in (i.quantity, i.unit, i.name) if x)


def generate(model_name: str, adapter: Path | None, rows: list[dict]) -> list[dict]:
    from mlx_lm import load, stream_generate

    model, tokenizer = load(model_name, adapter_path=str(adapter) if adapter else None)
    results = []
    for n, row in enumerate(rows, 1):
        prompt = tokenizer.apply_chat_template(row["messages"][:2], add_generation_prompt=True, tokenize=False)
        start, text = time.time(), ""
        for r in stream_generate(model, tokenizer, prompt=prompt, max_tokens=1500):
            text += r.text
        try:
            recipe = local_llm.coerce(local_llm._extract_json(text))
        except Exception:
            recipe = None
        results.append({"recipe": recipe, "seconds": time.time() - start, "raw": text})
        if n % 10 == 0:
            print(f"  {n}/{len(rows)}", file=sys.stderr)
    del model
    return results


def evaluate(limit: int | None, skip_reference: bool) -> None:
    rows = [json.loads(l) for l in (MLX_DATA / "test.jsonl").read_text(encoding="utf-8").splitlines()][:limit]
    gold = [json.loads(r["messages"][2]["content"]) for r in rows]
    gold_ing = [[ingredient_line(i) for i in local_llm.coerce({"ingredients": g["ingredients"]}).ingredients] for g in gold]
    gold_steps = [step_tokens(g["steps"]) for g in gold]

    candidates = [("3B sin entrenar", BASE_MODEL, None), ("3B + tu LoRA", BASE_MODEL, ADAPTER)]
    if not skip_reference:
        candidates.append(("7B sin entrenar (app actual)", REFERENCE_MODEL, None))
    report, details = {}, {}
    for name, model_name, adapter in candidates:
        if adapter is not None and not (adapter / "adapters.safetensors").exists():
            print("Aún no hay adaptador; ejecuta `entrenar`.")
            continue
        print(f"Generando con {name}…", file=sys.stderr)
        res = generate(model_name, adapter, rows)
        scores, step_p, step_r, n_steps = [], [], [], []
        for r, gi, gs in zip(res, gold_ing, gold_steps):
            rec = r["recipe"]
            pred_ing = [ingredient_line(i) for i in rec.ingredients] if rec else []
            scores.append(ing.score(pred_ing, gi))
            ps = step_tokens([s.instruction for s in rec.steps]) if rec else set()
            hit = len(ps & gs)
            step_p.append(hit / len(ps) if ps else 0.0)
            step_r.append(hit / len(gs) if gs else 0.0)
            n_steps.append(len(rec.steps) if rec else 0)
        summary = ing.summarize(scores)
        p, rc = sum(step_p) / len(rows), sum(step_r) / len(rows)
        report[name] = {
            "json_valido": round(sum(1 for r in res if r["recipe"]) / len(rows), 3),
            "ingredientes_f1": summary["f1"], "ingredientes_precision": summary["precision"],
            "ingredientes_recall": summary["recall"], "cantidad_correcta": summary["cantidad_correcta"],
            "pasos_cobertura": round(rc, 3), "pasos_precision": round(p, 3),
            "pasos_f1": round(2 * p * rc / max(1e-9, p + rc), 3),
            "pasos_por_receta": round(sum(n_steps) / len(rows), 1),
            "segundos_por_video": round(sum(r["seconds"] for r in res) / len(rows), 1),
        }
        details[name] = res
        print(f"{name:30} {report[name]}", flush=True)

    ADAPTER.mkdir(parents=True, exist_ok=True)
    (ADAPTER / "eval.json").write_text(json.dumps({
        "videos": len(rows), "resultados": report,
        "ejemplos": [
            {"id": row["id"], "autor": g, **{k: (v[i]["recipe"].model_dump() if v[i]["recipe"] else v[i]["raw"][:800])
                                            for k, v in details.items()}}
            for i, (row, g) in enumerate(zip(rows, gold))
        ],
    }, indent=1, ensure_ascii=False))
    print(f"Detalle por video en {ADAPTER / 'eval.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preparar")
    t = sub.add_parser("entrenar")
    t.add_argument("--iters", type=int)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--capas", type=int, default=16)
    t.add_argument("--dropout", type=float, default=0.1)
    e = sub.add_parser("evaluar")
    e.add_argument("--limite", type=int)
    e.add_argument("--sin-7b", action="store_true", help="no comparar con el 7B (más rápido)")
    args = parser.parse_args()
    if args.cmd == "preparar":
        prepare()
    elif args.cmd == "entrenar":
        ing.train(args.iters, args.lr, args.capas, args.dropout,
                  base_model=BASE_MODEL, data_dir=MLX_DATA, adapter=ADAPTER, max_seq=4096)
    else:
        evaluate(args.limite, args.sin_7b)


if __name__ == "__main__":
    main()

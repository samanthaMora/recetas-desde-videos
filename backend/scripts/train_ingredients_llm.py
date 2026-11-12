"""Fase 5: tu propio modelo de lenguaje. Ajusta (LoRA) un modelo pequeño para que saque la lista
de ingredientes de la transcripción de un video. Todo corre en la GPU del Mac con MLX; no usa Claude.

    cd backend
    .venv/bin/python scripts/train_ingredients_llm.py preparar    # divide los datos y los pasa a formato chat
    .venv/bin/python scripts/train_ingredients_llm.py entrenar    # LoRA (unos minutos)
    .venv/bin/python scripts/train_ingredients_llm.py evaluar     # modelo base vs. modelo ajustado

Conceptos:
- Modelo base: Qwen2.5-1.5B-Instruct (1.500 millones de parámetros, licencia Apache 2.0), en 4 bits.
- LoRA: en vez de cambiar todos los pesos, se añaden matrices pequeñas a algunas capas y solo se
  entrenan esas (menos del 1 % de los parámetros). El resultado es un "adaptador" de pocos MB.
- Entrada: título + lo que se dice en el video. Salida: una línea por ingrediente.
- Evaluación: se compara cada lista generada con la que escribió el autor del video, en canales
  que el modelo no vio. Precisión = de lo que dice el modelo, cuánto es correcto. Recall = de los
  ingredientes reales, cuántos encuentra. "Cantidad correcta" = el número coincide con el del autor.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
EXAMPLES = db.DATA_DIR / "ingredients" / "examples.jsonl"
MLX_DATA = db.DATA_DIR / "ingredients" / "mlx"
ADAPTER = Path(__file__).resolve().parent.parent / "models" / "ingredients_lora"
MAX_CHARS = 7000  # de transcripción (~2.500 tokens)
SEED = 42

SYSTEM = (
    "Eres un asistente de cocina. A partir del título y la transcripción de un video de receta, "
    "escribe la lista de ingredientes con sus cantidades. Una línea por ingrediente, empezando por "
    "'- '. No escribas nada más."
)


def messages_for(title: str, transcript: str, ingredients: list[str] | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Título: {title}\n\nTranscripción:\n{transcript[:MAX_CHARS]}"},
    ]
    if ingredients is not None:
        msgs.append({"role": "assistant", "content": "\n".join(f"- {i}" for i in ingredients)})
    return msgs


def _clean_line(line: str) -> str:
    return re.sub(r"^\s*[-•*]+\s*", "", line).strip()


def parse_output(text: str) -> list[str]:
    lines = [_clean_line(l) for l in text.splitlines()]
    return [l for l in lines if l and len(l) < 120][:50]


def generate_list(model, tokenizer, prompt: str, max_tokens: int = 450) -> list[str]:
    """Genera la lista línea a línea y para en cuanto el modelo repite un ingrediente.

    Los modelos pequeños con decodificación voraz caen en bucles ("1 lata de frijoles" x 30);
    una línea repetida es la señal de que ya terminó la lista real.
    """
    from mlx_lm import stream_generate

    items: list[str] = []
    seen: set[str] = set()
    buffer = ""
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        buffer += response.text
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            item = _clean_line(line)
            if not item or len(item) >= 120:
                continue
            if item.lower() in seen:
                return items
            seen.add(item.lower())
            items.append(item)
    tail = _clean_line(buffer)
    if tail and len(tail) < 120 and tail.lower() not in seen:
        items.append(tail)
    return items[:50]


# ---------------------------------------------------------------- preparar


def load_examples() -> list[dict]:
    seen, out = set(), []
    for line in EXAMPLES.read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        if ex["id"] in seen or ex.get("language") not in ("es", "en", "pt"):
            continue  # repetidos, o audio en un idioma distinto al de la lista escrita
        seen.add(ex["id"])
        out.append(ex)
    return out


def prepare() -> None:
    from sklearn.model_selection import GroupShuffleSplit

    examples = load_examples()
    groups = [e["channel"] or e["id"] for e in examples]
    idx = list(range(len(examples)))
    # Por canal, como en el clasificador: los canales de prueba no aparecen en entrenamiento.
    trainval, test = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(idx, groups=groups))
    tv_groups = [groups[i] for i in trainval]
    tr, va = next(GroupShuffleSplit(1, test_size=0.12, random_state=SEED).split(trainval, groups=tv_groups))
    parts = {"train": [trainval[i] for i in tr], "valid": [trainval[i] for i in va], "test": list(test)}

    MLX_DATA.mkdir(parents=True, exist_ok=True)
    for name, ids in parts.items():
        with (MLX_DATA / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for i in ids:
                e = examples[i]
                f.write(json.dumps({"messages": messages_for(e["title"], e["transcript"], e["ingredients"]),
                                    "id": e["id"]}, ensure_ascii=False) + "\n")
        print(f"{name:6} {len(ids):4} ejemplos · {len({groups[i] for i in ids})} canales")
    long = sum(1 for e in examples if len(e["transcript"]) > MAX_CHARS)
    print(f"transcripciones recortadas a {MAX_CHARS} caracteres: {long} de {len(examples)}")


# ---------------------------------------------------------------- entrenar


EVAL_EVERY = 50


def train(
    iters: int | None, lr: float, layers: int, dropout: float,
    base_model: str = BASE_MODEL, data_dir: Path = MLX_DATA, adapter: Path = ADAPTER, max_seq: int = 3584,
) -> None:
    """LoRA con early stopping: guarda cada EVAL_EVERY pasos y se queda con la mejor validación.

    En la primera prueba (61 ejemplos, 150 pasos, lr 1e-4) la pérdida de entrenamiento bajó de
    1,39 a 0,32 mientras la de validación subía de 1,28 a 1,45: memorizaba. Por eso: menos
    épocas, tasa más baja, dropout y elegir el punto con menor pérdida en validación.
    """
    import shutil

    n_train = sum(1 for _ in (data_dir / "train.jsonl").open())
    iters = iters or max(EVAL_EVERY, round(n_train * 1.5 / EVAL_EVERY) * EVAL_EVERY)  # ~1,5 épocas
    if adapter.exists():
        shutil.rmtree(adapter)
    adapter.mkdir(parents=True)
    config = adapter / "lora_config.yaml"
    config.write_text(f"lora_parameters:\n  rank: 8\n  dropout: {dropout}\n  scale: 20.0\n")
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora", "--config", str(config),
        "--model", base_model, "--train", "--data", str(data_dir),
        "--adapter-path", str(adapter),
        "--iters", str(iters), "--batch-size", "1", "--num-layers", str(layers),
        "--learning-rate", str(lr), "--max-seq-length", str(max_seq),
        "--mask-prompt",        # la pérdida solo cuenta la respuesta, no la transcripción
        "--grad-checkpoint",    # menos memoria (16 GB de RAM)
        "--steps-per-report", "25", "--steps-per-eval", str(EVAL_EVERY), "--val-batches", "-1",
        "--save-every", str(EVAL_EVERY), "--seed", str(SEED),
    ]
    print(" ".join(cmd[2:]), flush=True)

    # Se lee la salida en vivo para anotar la pérdida de validación de cada punto guardado.
    val_losses: dict[int, float] = {}
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = re.search(r"^\s*(\d+)\s+val\s+([\d.]+)", ansi.sub("", line))
        if m:
            val_losses[int(m[1])] = float(m[2])
    if proc.wait() != 0:
        raise SystemExit("El entrenamiento falló")

    saved = {it: loss for it, loss in val_losses.items() if (adapter / f"{it:07d}_adapters.safetensors").exists()}
    if saved:
        best = min(saved, key=saved.get)
        shutil.copy(adapter / f"{best:07d}_adapters.safetensors", adapter / "adapters.safetensors")
        print(f"\nPérdida de validación por paso: {dict(sorted(val_losses.items()))}")
        print(f"Me quedo con el paso {best} (validación {saved[best]:.3f}) -> {adapter / 'adapters.safetensors'}")
    (adapter / "training.json").write_text(json.dumps(
        {"modelo_base": base_model, "ejemplos_entrenamiento": n_train, "pasos": iters, "lr": lr,
         "dropout": dropout, "capas": layers, "validacion_por_paso": val_losses}, indent=2))


# ---------------------------------------------------------------- evaluar

_UNITS = {
    "g", "gr", "grs", "gramo", "kg", "kilo", "ml", "l", "lt", "litro", "taza", "cup", "cda", "cdta", "cucharada",
    "cucharadita", "tbsp", "tsp", "oz", "lb", "libra", "pieza", "pza", "diente", "rama", "lata", "sobre", "pizca",
    "tablespoon", "teaspoon", "clove", "pinch", "unidad", "xicara", "colher", "puñado", "punado", "trozo", "rebanada",
}
_STOP = {
    "de", "del", "la", "el", "los", "las", "un", "una", "uno", "y", "o", "al", "en", "con", "para", "the", "of", "and",
    "or", "a", "gusto", "taste", "to", "poco", "media", "medio", "half", "fresh", "fresco", "fresca", "picado", "picada",
    "chopped", "grande", "pequeño", "pequeno", "mediano", "large", "small", "medium", "aproximadamente", "opcional",
}
_FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}
_WORD_NUM = {"un": 1, "una": 1, "uno": 1, "one": 1, "media": 0.5, "medio": 0.5, "half": 0.5, "dos": 2, "two": 2, "tres": 3, "three": 3}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _singular(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("es") and not tok.endswith("ses"):
        return tok[:-2]
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok


def name_tokens(line: str) -> frozenset[str]:
    line = re.sub(r"\([^)]*\)", " ", _fold(line))
    toks = [_singular(t) for t in re.findall(r"[a-zñ]+", line)]
    return frozenset(t for t in toks if len(t) >= 3 and t not in _STOP and t not in _UNITS)


def quantity(line: str) -> float | None:
    s = line.strip().lower()
    for ch, val in _FRACTIONS.items():
        s = s.replace(ch, f" {val} ")
    m = re.match(r"\s*(\d+)\s+(?:y\s+)?(\d+)\s*/\s*(\d+)", s)            # 1 1/2 · 1 y 3/4
    if m:
        return int(m[1]) + int(m[2]) / max(1, int(m[3]))
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)", s)                             # 1/2
    if m:
        return int(m[1]) / max(1, int(m[2]))
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", s)                             # 200 · 1,5
    if m:
        return float(m[1].replace(",", "."))
    first = re.match(r"\s*([a-zñ]+)", _fold(s))
    return _WORD_NUM.get(first[1]) if first else None


def _similar(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    if a <= b or b <= a:
        return 1.0
    jac = len(a & b) / len(a | b)
    fuzzy = difflib.SequenceMatcher(None, " ".join(sorted(a)), " ".join(sorted(b))).ratio()
    return max(jac, fuzzy if fuzzy >= 0.8 else 0.0)


def score(pred: list[str], gold: list[str]) -> dict:
    p_tok, g_tok = [name_tokens(x) for x in pred], [name_tokens(x) for x in gold]
    used: set[int] = set()
    matched = qty_total = qty_ok = 0
    for gi, g in enumerate(g_tok):
        best, best_s = None, 0.5
        for pi, p in enumerate(p_tok):
            s = _similar(g, p)
            if pi not in used and s >= best_s and s > 0:
                best, best_s = pi, s + 1e-9
        if best is None:
            continue
        used.add(best)
        matched += 1
        gq = quantity(gold[gi])
        if gq is not None:
            qty_total += 1
            pq = quantity(pred[best])
            qty_ok += int(pq is not None and abs(pq - gq) <= 0.05 * max(gq, 1e-9))
    return {"matched": matched, "pred": len(pred), "gold": len(gold), "qty_total": qty_total, "qty_ok": qty_ok}


def summarize(scores: list[dict]) -> dict:
    tot = {k: sum(s[k] for s in scores) for k in scores[0]}
    precision = tot["matched"] / max(1, tot["pred"])
    recall = tot["matched"] / max(1, tot["gold"])
    return {
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / max(1e-9, precision + recall), 3),
        "cantidad_correcta": round(tot["qty_ok"] / max(1, tot["qty_total"]), 3),
        "videos": len(scores),
    }


def generate_all(rows: list[dict], adapter: Path | None) -> list[list[str]]:
    from mlx_lm import load

    model, tokenizer = load(BASE_MODEL, adapter_path=str(adapter) if adapter else None)
    outputs = []
    for n, row in enumerate(rows, 1):
        prompt = tokenizer.apply_chat_template(row["messages"][:2], add_generation_prompt=True, tokenize=False)
        outputs.append(generate_list(model, tokenizer, prompt))
        if n % 10 == 0:
            print(f"  {n}/{len(rows)}", file=sys.stderr)
    return outputs


def evaluate(limit: int | None) -> None:
    rows = [json.loads(l) for l in (MLX_DATA / "test.jsonl").read_text(encoding="utf-8").splitlines()][:limit]
    gold = [parse_output(r["messages"][2]["content"]) for r in rows]
    report, preds = {}, {}
    for name, adapter in (("base (sin entrenar)", None), ("ajustado (tu modelo)", ADAPTER)):
        if adapter is not None and not (adapter / "adapters.safetensors").exists():
            print("Aún no hay adaptador entrenado; ejecuta primero `entrenar`.")
            continue
        print(f"Generando con el modelo {name}…", file=sys.stderr)
        preds[name] = generate_all(rows, adapter)
        report[name] = summarize([score(p, g) for p, g in zip(preds[name], gold)])
        print(f"{name:22} {report[name]}")

    out = ADAPTER / "eval.json"
    if ADAPTER.exists():
        out.write_text(json.dumps({"modelo_base": BASE_MODEL, "resultados": report, "ejemplos": [
            {"id": r["id"], "autor": g, **{k: v[i] for k, v in preds.items()}} for i, (r, g) in enumerate(zip(rows, gold))
        ]}, indent=2, ensure_ascii=False))
        print(f"Detalle por video en {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preparar")
    t = sub.add_parser("entrenar")
    t.add_argument("--iters", type=int)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--capas", type=int, default=12, help="capas del modelo que reciben LoRA")
    t.add_argument("--dropout", type=float, default=0.1)
    e = sub.add_parser("evaluar")
    e.add_argument("--limite", type=int)
    args = parser.parse_args()
    if args.cmd == "preparar":
        prepare()
    elif args.cmd == "entrenar":
        train(args.iters, args.lr, args.capas, args.dropout)
    else:
        evaluate(args.limite)


if __name__ == "__main__":
    main()

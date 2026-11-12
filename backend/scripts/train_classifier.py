"""Entrena y compara tres clasificadores «¿es receta?» y guarda el mejor para la app.

    cd backend && .venv/bin/python scripts/train_classifier.py

Modelos, de más simple a más complejo:
  A. tfidf_lr       TF-IDF + regresión logística. Machine learning clásico: cada palabra (y par
                    de palabras) recibe un peso; "receta", "ingredientes", "horno" suman.
  B. embeddings_lr  Transfer learning: e5-small (ya entrenado con millones de textos) convierte el
                    texto en un vector y una regresión logística aprende a separar los vectores.
  C. finetuned      Deep learning: se reentrena la red e5-small completa (118 M de parámetros) junto
                    con una capa de clasificación, con PyTorch en la GPU del Mac.

Evaluación honesta:
  - Los canales de prueba no aparecen en entrenamiento (GroupShuffleSplit por canal): si no, el
    modelo podría "acertar" reconociendo el estilo de un canal que ya vio.
  - Métrica de producción: con el umbral elegido, ¿cuántas recetas reales se bloquearían por error?
    (tiene que ser ~0) y ¿cuántos videos que no son recetas se detectan?
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app import db  # noqa: E402
from app.classifier import MODEL_DIR, video_text  # noqa: E402

RAW = db.DATA_DIR / "classifier" / "raw.jsonl"
LABEL_FIXES = db.DATA_DIR / "classifier" / "label_fixes.json"  # {video_id: 0|1} revisados a mano
SEED = 42


# ---------------------------------------------------------------- datos


@dataclass
class Dataset:
    items: list[dict]
    texts: list[str]
    labels: np.ndarray
    groups: np.ndarray  # canal de cada video

    def subset(self, idx) -> "Dataset":
        idx = np.asarray(idx)
        return Dataset([self.items[i] for i in idx], [self.texts[i] for i in idx], self.labels[idx], self.groups[idx])


def load_dataset(path: Path = RAW) -> Dataset:
    fixes = json.loads(LABEL_FIXES.read_text()) if LABEL_FIXES.exists() else {}
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["id"] in fixes:
            item["label"] = int(fixes[item["id"]])
            item["fixed"] = True
        items.append(item)
    texts = [video_text(i["title"], i["description"]) for i in items]
    labels = np.array([i["label"] for i in items])
    groups = np.array([i["channel"] or i["id"] for i in items])
    return Dataset(items, texts, labels, groups)


def split(ds: Dataset, test_size: float = 0.2, val_size: float = 0.15):
    """train / val / test sin compartir canales entre conjuntos."""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    trainval_idx, test_idx = next(gss.split(ds.texts, ds.labels, ds.groups))
    trainval = ds.subset(trainval_idx)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=SEED)
    train_idx, val_idx = next(gss2.split(trainval.texts, trainval.labels, trainval.groups))
    return trainval.subset(train_idx), trainval.subset(val_idx), ds.subset(test_idx)


# ---------------------------------------------------------------- modelos


def train_tfidf(train: Dataset):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", lowercase=True,
            # Solo palabras con letras: en la primera versión el número "2026" era la señal más fuerte
            # de "no receta" (las búsquedas de noticias/gameplay traían videos recientes). Un atajo
            # del dataset que penalizaría cualquier receta que mencione el año.
            token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
        ),
        LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000),
    )
    model.fit(train.texts, train.labels)
    return model


def embed(texts: list[str]) -> np.ndarray:
    from app.embeddings import _encode

    return _encode(["query: " + t for t in texts])


def train_embeddings_lr(train: Dataset, train_emb: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000)
    model.fit(train_emb, train.labels)
    return model


def train_finetuned(train: Dataset, val: Dataset, epochs: int = 3, lr: float = 3e-5, batch_size: int = 16,
                    log=print):
    """Ajuste fino de e5-small completo + capa lineal, con PyTorch."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from app.embeddings import TEXT_MODEL
    from app.vision import _device

    torch.manual_seed(SEED)
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    encoder = AutoModel.from_pretrained(TEXT_MODEL).to(device)
    head = torch.nn.Linear(encoder.config.hidden_size, 1).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    # float32: la GPU de los Mac (MPS) no admite float64.
    ratio = float((train.labels == 0).sum() / max(1, (train.labels == 1).sum()))
    pos_weight = torch.tensor([ratio], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(texts):
        batch = tokenizer(["query: " + t for t in texts], padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to(device)
        hidden = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return head((hidden * mask).sum(1) / mask.sum(1)).squeeze(-1)

    def predict(texts):
        encoder.eval(); head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                out.append(torch.sigmoid(forward(texts[i:i + 32])).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    history = []
    rng = np.random.default_rng(SEED)
    for epoch in range(epochs):
        encoder.train(); head.train()
        order = rng.permutation(len(train.texts))
        losses = []
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            logits = forward([train.texts[j] for j in idx])
            loss = loss_fn(logits, torch.tensor(train.labels[idx], dtype=torch.float32, device=device))
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(loss.item())
        val_pred = predict(val.texts)
        val_acc = float(((val_pred >= 0.5) == val.labels).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_accuracy": val_acc})
        log(f"  época {epoch + 1}: pérdida {np.mean(losses):.4f} · exactitud validación {val_acc:.3f}")

    return {"encoder": encoder, "head": head, "tokenizer": tokenizer, "predict": predict, "history": history}


# ---------------------------------------------------------------- evaluación


def choose_threshold(val_labels: np.ndarray, val_proba: np.ndarray, max_recipe_loss: float = 0.0) -> float:
    """Umbral para RECHAZAR: el más alto que no bloquea (casi) ninguna receta de validación.

    Se aplica un margen (x0.5) porque validación es pequeña y en producción habrá videos distintos.
    """
    recipe_scores = np.sort(val_proba[val_labels == 1])
    if not len(recipe_scores):
        return 0.0
    k = int(np.floor(max_recipe_loss * len(recipe_scores)))
    return float(recipe_scores[k]) * 0.5


def metrics(labels: np.ndarray, proba: np.ndarray, reject_below: float) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    pred = (proba >= 0.5).astype(int)
    rejected = proba < reject_below
    return {
        "exactitud": round(float(accuracy_score(labels, pred)), 4),
        "precision": round(float(precision_score(labels, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(labels, proba)), 4) if len(set(labels)) > 1 else None,
        "umbral_rechazo": round(reject_below, 4),
        "recetas_bloqueadas": int((rejected & (labels == 1)).sum()),
        "recetas_total": int((labels == 1).sum()),
        "no_recetas_detectadas": int((rejected & (labels == 0)).sum()),
        "no_recetas_total": int((labels == 0).sum()),
    }


def app_recipes() -> list[str]:
    """Videos reales analizados por la app (todos son recetas): comprobación extra."""
    return [video_text(r["title"] or "", r["description"] or "")
            for r in db.iter_rows("SELECT title, description FROM analyses WHERE status='done' AND is_recipe=1")]


# ---------------------------------------------------------------- guardar


def save(kind: str, model, reject_below: float, report: dict) -> None:
    import shutil

    import joblib

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True)
    if kind == "tfidf_lr":
        joblib.dump(model, MODEL_DIR / "tfidf_lr.joblib")
    elif kind == "embeddings_lr":
        joblib.dump(model, MODEL_DIR / "embeddings_lr.joblib")
    elif kind == "finetuned":
        import torch

        out = MODEL_DIR / "finetuned"
        model["encoder"].save_pretrained(out)
        model["tokenizer"].save_pretrained(out)
        torch.save(model["head"].state_dict(), out / "head.pt")
    (MODEL_DIR / "config.json").write_text(json.dumps(
        {"type": kind, "reject_below": reject_below, "trained_at": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
    (MODEL_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- todo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-ajuste-fino", action="store_true", help="saltar el modelo C (tarda varios minutos)")
    args = parser.parse_args()

    ds = load_dataset()
    train, val, test = split(ds)
    print(f"Datos: {len(ds.texts)} videos · train {len(train.texts)} · val {len(val.texts)} · test {len(test.texts)}")
    print(f"Recetas: train {train.labels.mean():.0%} · val {val.labels.mean():.0%} · test {test.labels.mean():.0%}\n")

    real = app_recipes()
    results, models = {}, {}

    def evaluate(name, val_p, test_p, real_p, secs):
        t = choose_threshold(val.labels, val_p)
        m = metrics(test.labels, test_p, t)
        mv = metrics(val.labels, val_p, t)
        m["f1_validacion"] = mv["f1"]
        # Lo que importa en producción: % de no-recetas rechazadas sin bloquear recetas (en validación).
        m["filtro_validacion"] = round(mv["no_recetas_detectadas"] / max(1, mv["no_recetas_total"]), 4)
        m["app_recetas_bloqueadas"] = int((np.asarray(real_p) < t).sum()) if len(real_p) else 0
        m["app_recetas_total"] = len(real_p)
        m["segundos_entrenamiento"] = round(secs, 1)
        results[name] = m
        print(f"{name}: {m}\n")

    t0 = time.time()
    tfidf = train_tfidf(train)
    models["tfidf_lr"] = tfidf
    evaluate("tfidf_lr", tfidf.predict_proba(val.texts)[:, 1], tfidf.predict_proba(test.texts)[:, 1],
             tfidf.predict_proba(real)[:, 1] if real else [], time.time() - t0)

    t0 = time.time()
    tr_e, va_e, te_e = embed(train.texts), embed(val.texts), embed(test.texts)
    emb_lr = train_embeddings_lr(train, tr_e)
    models["embeddings_lr"] = emb_lr
    evaluate("embeddings_lr", emb_lr.predict_proba(va_e)[:, 1], emb_lr.predict_proba(te_e)[:, 1],
             emb_lr.predict_proba(embed(real))[:, 1] if real else [], time.time() - t0)

    if not args.sin_ajuste_fino:
        t0 = time.time()
        print("Ajuste fino de e5-small:")
        ft = train_finetuned(train, val)
        models["finetuned"] = ft
        evaluate("finetuned", ft["predict"](val.texts), ft["predict"](test.texts),
                 ft["predict"](real) if real else [], time.time() - t0)

    # Se elige con VALIDACIÓN (test solo se usa para informar) y con la métrica de PRODUCCIÓN:
    # qué fracción de no-recetas se rechaza con el umbral que no bloquea recetas. Un buen F1 no
    # basta: un modelo puede clasificar bien en 0.5 y casi nunca dar probabilidades bajísimas, y
    # entonces como filtro no sirve. Ante empate (±0.02) gana el más simple (más rápido y barato).
    order = [k for k in ["tfidf_lr", "embeddings_lr", "finetuned"] if k in results]
    best_rate = max(results[k]["filtro_validacion"] for k in order)
    best = next(k for k in order if results[k]["filtro_validacion"] >= best_rate - 0.02)
    reject_below = results[best]["umbral_rechazo"]
    save(best, models[best], reject_below, {"elegido": best, "resultados": results})
    print(f"Elegido: {best} (umbral de rechazo {reject_below:.3f}) -> {MODEL_DIR}")


if __name__ == "__main__":
    main()

"""Genera notebooks/clasificador_recetas.ipynb (y se ejecuta con nbconvert para guardar resultados).

    cd backend && .venv/bin/python notebooks/build_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/clasificador_recetas.ipynb
"""

from pathlib import Path

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Tu primer modelo propio: ¿este video es una receta?

Este notebook entrena y compara **tres clasificadores** que, leyendo solo el **título y la descripción** de un video,
deciden si es una receta de cocina. En la app sirven para rechazar al instante los videos que claramente no son recetas,
**antes** de descargarlos y de pagar el análisis con Claude.

Lo que vas a ver:

1. **Datos con etiquetas débiles** — cómo conseguir ~1000 ejemplos etiquetados gratis.
2. **División honesta** — por qué los canales de prueba no pueden aparecer en entrenamiento.
3. **Modelo A: TF-IDF + regresión logística** — machine learning clásico, y qué palabras aprendió.
4. **Modelo B: embeddings + regresión logística** — *transfer learning* con un modelo ya entrenado.
5. **Modelo C: ajuste fino (fine-tuning)** — *deep learning*: reentrenar una red transformer completa con PyTorch.
6. **Comparación**, curvas ROC y errores.
7. **Ruido en las etiquetas** — usar el propio modelo para encontrar ejemplos mal etiquetados.
8. **El umbral de producción** — cómo decidir cuándo rechazar sin bloquear recetas reales.

Todo el código de entrenamiento vive en `scripts/train_classifier.py`; aquí lo usamos paso a paso."""),
    code("""import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))          # carpeta backend/
sys.path.insert(0, str(Path.cwd().parent / "scripts"))

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import train_classifier as tc

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"receta": "#C4512B", "comida_no_receta": "#E0A030", "no_comida": "#5B7A99"}"""),
    md("""## 1. Datos con etiquetas débiles

Etiquetar 1000 videos a mano llevaría horas. En su lugar usamos **etiquetas débiles** (*weak labels*):
la búsqueda que trajo cada video decide su etiqueta.

- Lo que sale al buscar «receta de pan casero» → **receta (1)**
- Lo que sale al buscar «resumen de fútbol» → **no receta (0)**
- Lo que sale al buscar «reseña de restaurante», «mukbang»… → **no receta (0)**: son los **negativos difíciles**, videos de comida que no son recetas.
  Sin ellos el modelo aprendería el atajo «habla de comida = receta».

Las etiquetas débiles tienen ruido (algún video de la búsqueda «receta fácil» puede no serlo). Lo medimos en la sección 7."""),
    code("""ds = tc.load_dataset()
kinds = Counter(i["kind"] for i in ds.items)
print(f"{len(ds.texts)} videos")
for k, n in kinds.most_common():
    print(f"  {k:18} {n}")

for kind in ["receta", "comida_no_receta", "no_comida"]:
    print(f"\\n— Ejemplos de «{kind}»:")
    for item in [i for i in ds.items if i["kind"] == kind][:3]:
        print("  ·", item["title"][:90])"""),
    md("""Así ve el texto el modelo: título + inicio de la descripción, sin enlaces. **No ve el nombre del canal** a propósito:
aprendería atajos («canal de cocina») que no sirven para Instagram, TikTok ni canales nuevos."""),
    code("""print(ds.texts[0][:400])"""),
    md("""## 2. División honesta: entrenamiento, validación y prueba

- **Entrenamiento**: con lo que aprende el modelo.
- **Validación**: para elegir el modelo y el umbral.
- **Prueba**: solo para medir al final. Nunca se usa para decidir nada.

Un canal suele subir videos muy parecidos. Si un canal aparece en entrenamiento y en prueba, el modelo puede «acertar» reconociendo
el estilo del canal en vez de entender si es una receta. Por eso dividimos **por canal** (`GroupShuffleSplit`)."""),
    code("""train, val, test = tc.split(ds)
for name, part in [("entrenamiento", train), ("validación", val), ("prueba", test)]:
    print(f"{name:14} {len(part.texts):4} videos · {part.labels.mean():.0%} recetas · {len(set(part.groups))} canales")
print("canales compartidos entre entrenamiento y prueba:", len(set(train.groups) & set(test.groups)))"""),
    md("""## 3. Modelo A — TF-IDF + regresión logística (machine learning clásico)

**TF-IDF** convierte cada texto en un vector con una posición por palabra (y por par de palabras). El valor es alto si la palabra
aparece en ese texto pero es rara en el resto. La **regresión logística** aprende un peso por posición: positivo si empuja hacia
«receta», negativo si empuja hacia «no receta». Es rápido, barato e **interpretable**: podemos leer qué aprendió."""),
    code("""tfidf = tc.train_tfidf(train)
p_val_A, p_test_A = tfidf.predict_proba(val.texts)[:, 1], tfidf.predict_proba(test.texts)[:, 1]
thr_A = tc.choose_threshold(val.labels, p_val_A)
tc.metrics(test.labels, p_test_A, thr_A)"""),
    code("""vec, lr = tfidf.named_steps["tfidfvectorizer"], tfidf.named_steps["logisticregression"]
words, weights = np.array(vec.get_feature_names_out()), lr.coef_[0]
order = np.argsort(weights)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, idx, color, title in [(axes[0], order[-15:], COLORS["receta"], "Empujan hacia RECETA"),
                              (axes[1], order[:15][::-1], COLORS["no_comida"], "Empujan hacia NO RECETA")]:
    ax.barh(words[idx], np.abs(weights[idx]), color=color)
    ax.set_title(title); ax.tick_params(labelsize=9)
plt.tight_layout(); plt.show()"""),
    md("""**Un atajo que encontramos así:** en la primera versión, la señal más fuerte de «no receta» era el número **«2026»**.
No tiene nada que ver con cocinar: las búsquedas de noticias, gameplays y restaurantes traían videos recientes que mencionan el año,
y las de recetas no. El modelo aprendió una correlación del *dataset*, no de la tarea, y habría penalizado cualquier receta que dijera
«receta navideña 2026». Lo corregimos haciendo que TF-IDF solo use palabras con letras (`token_pattern`). El rendimiento no bajó.

Mirar qué aprendió el modelo, y no solo su exactitud, es la mejor forma de detectar estos atajos."""),
    md("""## 4. Modelo B — embeddings + regresión logística (transfer learning)

En vez de contar palabras, usamos **e5-small**, una red neuronal que ya fue entrenada con millones de textos en muchos idiomas
(la misma de la búsqueda de la fase 3). Convierte cada texto en un vector de 384 números que captura su **significado**:
«receta de pan» y «cómo hornear pan casero» quedan cerca aunque no compartan palabras.

Encima ponemos una regresión logística. Es *transfer learning*: reutilizamos lo que la red ya sabe y solo entrenamos la última capa."""),
    code("""E_train, E_val, E_test = tc.embed(train.texts), tc.embed(val.texts), tc.embed(test.texts)
emb_lr = tc.train_embeddings_lr(train, E_train)
p_val_B, p_test_B = emb_lr.predict_proba(E_val)[:, 1], emb_lr.predict_proba(E_test)[:, 1]
thr_B = tc.choose_threshold(val.labels, p_val_B)
tc.metrics(test.labels, p_test_B, thr_B)"""),
    md("""**¿Cómo ve la red los videos?** Reducimos los vectores de 384 dimensiones a 2 con PCA para dibujarlos. Cada punto es un video.
Si los colores forman grupos separados, la red ya «entiende» la diferencia antes de entrenar nada."""),
    code("""from sklearn.decomposition import PCA
E_all = np.vstack([E_train, E_val, E_test])
kinds_all = [i["kind"] for part in (train, val, test) for i in part.items]
xy = PCA(n_components=2, random_state=0).fit_transform(E_all)
plt.figure(figsize=(7, 5.5))
for kind, color in COLORS.items():
    m = np.array([k == kind for k in kinds_all])
    plt.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.6, c=color, label=kind)
plt.legend(); plt.title("Embeddings e5-small proyectados a 2D (PCA)"); plt.xticks([]); plt.yticks([]); plt.show()"""),
    md("""## 5. Modelo C — ajuste fino de la red completa (deep learning)

Ahora **reentrenamos toda la red e5-small** (118 millones de parámetros) junto con una capa de clasificación nueva.
Con PyTorch, en la GPU del Mac:

- Cada **época** recorre todos los ejemplos de entrenamiento en lotes de 16.
- Para cada lote, la red predice, se mide el error (*binary cross-entropy*) y el optimizador **AdamW** ajusta todos los pesos un poco
  en la dirección que reduce el error (*backpropagation*).
- Usamos una tasa de aprendizaje pequeña (3e-5) para no «olvidar» lo que la red ya sabía.

Tarda unos minutos."""),
    code("""ft = tc.train_finetuned(train, val, epochs=3)
p_val_C, p_test_C = ft["predict"](val.texts), ft["predict"](test.texts)
thr_C = tc.choose_threshold(val.labels, p_val_C)
tc.metrics(test.labels, p_test_C, thr_C)"""),
    code("""h = ft["history"]
fig, ax1 = plt.subplots(figsize=(6, 3.5))
ax1.plot([x["epoch"] for x in h], [x["train_loss"] for x in h], "o-", color=COLORS["receta"], label="pérdida (entrenamiento)")
ax1.set_xlabel("época"); ax1.set_ylabel("pérdida")
ax2 = ax1.twinx(); ax2.plot([x["epoch"] for x in h], [x["val_accuracy"] for x in h], "s--", color=COLORS["no_comida"], label="exactitud (validación)")
ax2.set_ylabel("exactitud"); ax1.set_title("Curva de aprendizaje del ajuste fino"); fig.legend(loc="center right"); plt.tight_layout(); plt.show()"""),
    md("""## 6. Comparación

- **Exactitud**: % de aciertos.  **Precisión**: de lo que el modelo llama receta, cuánto lo es.  **Recall**: de las recetas reales, cuántas detecta.
- **F1**: equilibrio entre precisión y recall.  **AUC**: qué tan bien ordena (1 = perfecto, 0,5 = al azar).
- **Recetas bloqueadas**: con el umbral de producción, recetas reales que se rechazarían por error. **Tiene que ser 0.**"""),
    code("""import pandas as pd
rows = {}
for name, pv, pt, thr in [("A · TF-IDF", p_val_A, p_test_A, thr_A), ("B · embeddings", p_val_B, p_test_B, thr_B), ("C · ajuste fino", p_val_C, p_test_C, thr_C)]:
    m = tc.metrics(test.labels, pt, thr)
    mv = tc.metrics(val.labels, pv, thr)
    m["f1_validacion"] = mv["f1"]
    m["filtro_validacion"] = round(mv["no_recetas_detectadas"] / max(1, mv["no_recetas_total"]), 3)
    rows[name] = m
pd.DataFrame(rows).T[["exactitud", "f1", "auc", "f1_validacion", "filtro_validacion", "umbral_rechazo", "recetas_bloqueadas", "no_recetas_detectadas", "no_recetas_total"]]"""),
    md("""### La lección: elige la métrica según el uso real

Si miramos solo el **F1**, gana el ajuste fino (la red neuronal). Pero en la app el modelo no se usa en la frontera del 50 %:
se usa como **filtro** que solo rechaza cuando está *muy* seguro. Con el umbral que no bloquea ninguna receta,
la red ajustada casi nunca da probabilidades tan bajas y **no rechazaría prácticamente nada** (columna `filtro_validacion`).

El modelo clásico TF-IDF, que entrena en una fracción de segundo y ocupa 200 KB, es el que más videos inútiles filtra sin bloquear recetas.
Por eso es el que va a producción. Más complejo no siempre es mejor: depende de **para qué** se usa el modelo."""),
    code("""from sklearn.metrics import roc_curve
plt.figure(figsize=(5.5, 5))
for name, p, color in [("A · TF-IDF", p_test_A, COLORS["no_comida"]), ("B · embeddings", p_test_B, COLORS["comida_no_receta"]), ("C · ajuste fino", p_test_C, COLORS["receta"])]:
    fpr, tpr, _ = roc_curve(test.labels, p)
    plt.plot(fpr, tpr, label=name, color=color)
plt.plot([0, 1], [0, 1], ":", color="gray")
plt.xlabel("falsos positivos"); plt.ylabel("verdaderos positivos"); plt.title("Curvas ROC (conjunto de prueba)"); plt.legend(); plt.show()"""),
    md("""### ¿En qué se equivoca cada modelo?

Mirar los errores enseña más que cualquier métrica: a veces el «error» es del modelo y a veces de la etiqueta."""),
    code("""for name, p in [("A · TF-IDF", p_test_A), ("B · embeddings", p_test_B), ("C · ajuste fino", p_test_C)]:
    wrong = np.where((p >= 0.5) != test.labels)[0]
    print(f"\\n{name}: {len(wrong)} errores de {len(test.labels)}")
    for i in wrong[:6]:
        print(f"  [etiqueta {test.labels[i]} · prob. receta {p[i]:.2f} · {test.items[i]['kind']}] {test.items[i]['title'][:80]}")"""),
    md("""## 7. Ruido en las etiquetas: el modelo como revisor

Las etiquetas débiles tienen errores. Una técnica (*confident learning*) es entrenar con validación cruzada, de forma que cada video
reciba una predicción de un modelo que **no lo vio**, y revisar los casos donde el modelo está muy seguro de lo contrario de la etiqueta.
Los que de verdad estaban mal se corrigen en `data/classifier/label_fixes.json` y se reentrena."""),
    code("""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
E_ds = tc.embed(ds.texts)
oof = cross_val_predict(LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000), E_ds, ds.labels,
                        groups=ds.groups, cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
disagree = np.argsort(-np.abs(oof - ds.labels))[:15]
for i in disagree:
    it = ds.items[i]
    fixed = " (ya corregida)" if it.get("fixed") else ""
    print(f"etiqueta {ds.labels[i]} · modelo {oof[i]:.2f} · búsqueda «{it['query']}»{fixed}\\n    {it['title'][:95]}")"""),
    md("""## 8. El umbral de producción

En la app el modelo **solo rechaza** cuando está muy seguro: si la probabilidad de receta es menor que el umbral.
Elegimos el umbral en **validación** como la mitad de la puntuación de la receta peor puntuada: así ninguna receta de validación se
habría bloqueado, con margen para videos nuevos. Todo lo que queda por encima se analiza normalmente con Claude."""),
    code("""best = json.loads((tc.MODEL_DIR / "config.json").read_text()) if (tc.MODEL_DIR / "config.json").exists() else None
p_best = {"tfidf_lr": p_test_A, "embeddings_lr": p_test_B, "finetuned": p_test_C}[best["type"]] if best else p_test_B
thr = best["reject_below"] if best else thr_B
plt.figure(figsize=(7, 3.8))
bins = np.linspace(0, 1, 41)
plt.hist(p_best[test.labels == 1], bins=bins, alpha=0.7, color=COLORS["receta"], label="recetas")
plt.hist(p_best[test.labels == 0], bins=bins, alpha=0.7, color=COLORS["no_comida"], label="no recetas")
plt.axvline(thr, color="black", ls="--", label=f"umbral de rechazo {thr:.3f}")
plt.yscale("log"); plt.xlabel("probabilidad de receta"); plt.ylabel("videos (escala log)")
plt.title(f"Modelo en producción: {best['type'] if best else '—'} (conjunto de prueba)"); plt.legend(); plt.show()"""),
    md("""## Conclusión

`scripts/train_classifier.py` hace todo esto de forma reproducible y guarda en `models/recipe_classifier/` el modelo elegido:
el de mejor F1 en **validación**, y ante empate el más simple (más rápido y barato de servir).
Su informe completo está en `models/recipe_classifier/report.json`.

**Siguientes pasos para mejorarlo:**
- Añadir ejemplos reales de Instagram y TikTok (sus descripciones son distintas a las de YouTube).
- Usar los videos que la app rechaza y que el usuario manda «analizar de todos modos»: son errores del modelo, los mejores ejemplos para reentrenar.
- Revisar más etiquetas con la sección 7 y volver a entrenar."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "clasificador_recetas.ipynb"
nbf.write(nb, out)
print(f"Notebook escrito en {out}")

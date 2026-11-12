"""Evalúa modelos de embeddings y formas de describir la receta con ejemplos etiquetados a mano.

    cd backend && .venv/bin/python scripts/eval_embeddings.py

Mide: (1) separación entre pares que deberían ser parecidos y pares distintos (margen),
(2) en qué posición queda la respuesta correcta de cada búsqueda. Amplía `similar_pairs`,
`dissimilar_pairs` y `queries` a medida que haya más recetas, y úsalo para recalibrar
SEARCH_MIN_SCORE / SIMILAR_MIN_SCORE en app/embeddings.py.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer
from app.schema import Recipe
from app import db as appdb
from app.embeddings import recipe_text, recipe_text_short

db = sqlite3.connect(appdb.DB_PATH)
rows = db.execute("select json_extract(recipe_json,'$.title'), recipe_json from analyses where status='done' and is_recipe=1").fetchall()
recipes = [(t, Recipe.model_validate_json(j)) for t, j in rows]
def find(prefix): return next(i for i,(t,_) in enumerate(recipes) if t.lower().startswith(prefix))

# Pares que deberían ser parecidos / búsquedas con su respuesta correcta (etiquetado a mano)
similar_pairs = [("postre de naranja","postre frío de limón"), ("chilaquiles","enchiladas")]
dissimilar_pairs = [("cóctel de camarón","chilaquiles"), ("postre de naranja","chilaquiles"), ("masa para pizza","cóctel de camarón"), ("crema de brócoli","postre frío de limón")]
queries = {"mariscos":"cóctel de camarón", "postre sin horno":"postre", "sopa verde cremosa":"crema de brócoli",
           "comida mexicana":"enchiladas", "masa casera":"masa para pizza", "algo dulce de limón":"postre frío de limón",
           "camarones":"cóctel de camarón", "algo con queso gratinado":"enchiladas"}

variants = {"completo": recipe_text, "corto": recipe_text_short}
for model_name in ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base"]:
    m = SentenceTransformer(model_name, device="mps" if __import__("torch").backends.mps.is_available() else "cpu")
    for vname, fn in variants.items():
        E = m.encode(["passage: "+fn(r) for _, r in recipes], normalize_embeddings=True)
        S = E @ E.T
        sim = [S[find(a), find(b)] for a,b in similar_pairs]
        dis = [S[find(a), find(b)] for a,b in dissimilar_pairs]
        # búsqueda: ¿la respuesta correcta queda primera?
        Q = m.encode(["query: "+q for q in queries], normalize_embeddings=True)
        top1 = 0; ranks=[]
        for qi,(q,ans) in enumerate(queries.items()):
            order = np.argsort(-(E @ Q[qi]))
            rank = [recipes[i][0].lower().startswith(ans) for i in order].index(True)+1
            ranks.append(rank); top1 += rank==1
        print(f"{model_name.split('/')[1]:26} {vname:9} | parecidos {np.mean(sim):.3f} vs distintos {np.mean(dis):.3f} (margen {np.mean(sim)-np.mean(dis):+.3f}) | búsqueda top-1 {top1}/{len(queries)}  posiciones {ranks}")

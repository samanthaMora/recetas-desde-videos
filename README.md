# Recetas desde videos

Compartes un video de cocina (YouTube, Instagram, TikTok, Facebook, X o Pinterest) y la app te devuelve
la receta: ingredientes con cantidades, pasos, tiempos y consejos.

```
backend/   Servidor en Python: descarga el video, lo transcribe, saca fotogramas y le pide la receta a Claude
mobile/    App Expo (iPhone, Android y web) con la opción «Compartir con Recetas»
```

## Cómo funciona

1. `yt-dlp` descarga el video (máx. 720p) junto con la descripción y los subtítulos.
2. Si hay subtítulos, se usan como transcripción; si no, Whisper transcribe el audio en el servidor.
3. Visión ([backend/app/vision.py](backend/app/vision.py)): se recortan las bandas negras, se sacan hasta 240
   fotogramas candidatos y CLIP elige los 20 más distintos y nítidos. EasyOCR lee el texto en pantalla.
4. Claude (`claude-opus-5-5`) combina descripción, transcripción, texto en pantalla y fotogramas y devuelve
   la receta en JSON. Las cantidades que no se dicen se estiman por la imagen y se marcan con «≈».
5. Todo queda en `backend/data/` (SQLite + fotogramas). Las correcciones de los usuarios se guardan aparte.
   `scripts/export_dataset.py` lo convierte en datasets para entrenar modelos propios.

## Plan de machine learning

| Fase | Estado |
|---|---|
| 1. Recoger datos (análisis + correcciones) | ✅ |
| 2. Fotogramas con CLIP + OCR | ✅ ~35 % más barato por video |
| 3. Búsqueda semántica y recomendaciones (embeddings) | ✅ e5-small, evaluado con `scripts/eval_embeddings.py` |
| 4. Clasificador propio «¿es receta?» | ✅ TF-IDF en producción — ver `notebooks/clasificador_recetas.ipynb` |
| 5. Modelo de lenguaje propio con LoRA (MLX) | ✅ experimento — Qwen2.5-1.5B + LoRA empata con el modelo base (F1 0,60 vs 0,61); no está en producción |

### Reentrenar el clasificador

```bash
cd backend
.venv/bin/python scripts/collect_classifier_data.py      # dataset de YouTube (gratis, sin Claude)
.venv/bin/python scripts/train_classifier.py             # entrena A/B/C y guarda el mejor en models/
.venv/bin/python scripts/train_classifier.py --sin-ajuste-fino   # rápido, sin la red neuronal
.venv/bin/jupyter lab notebooks/clasificador_recetas.ipynb       # la explicación paso a paso
```

Las correcciones de etiquetas revisadas a mano van en `data/classifier/label_fixes.json`.

### Modelo propio de ingredientes (fase 5)

Aprende a sacar la lista de ingredientes a partir de lo que se dice en el video. Los datos son gratis
y no usan Claude: la entrada es la transcripción y la respuesta correcta es la lista que el autor
escribió en la descripción. Necesita un Mac con Apple Silicon (MLX).

```bash
cd backend
.venv/bin/python scripts/collect_ingredients_data.py --buscar   # dataset (lento a propósito, reanudable)
.venv/bin/python scripts/train_ingredients_llm.py preparar      # train/valid/test separados por canal
.venv/bin/python scripts/train_ingredients_llm.py entrenar      # LoRA, se queda con el mejor paso en validación
.venv/bin/python scripts/train_ingredients_llm.py evaluar       # base vs ajustado en canales no vistos
.venv/bin/python scripts/extract_ingredients.py "<enlace de YouTube>"   # probarlo con un video
```

Resultados (68 videos de test, canales que el modelo no vio):

| | Precisión | Recall | F1 | Cantidad correcta |
|---|---|---|---|---|
| Qwen2.5-1.5B sin entrenar | 0,59 | 0,64 | 0,61 | 56 % |
| Con LoRA (399 ejemplos) | 0,62 | 0,59 | 0,60 | 60 % |

Lo que más mejoró fue la generación, no el entrenamiento: sin cortar los bucles de repetición los
dos modelos sacaban 20-40 «ingredientes» repetidos (F1 0,38 y 0,28). El LoRA aprendió el formato
(listas del tamaño de las del autor, cantidades algo más exactas) pero no a encontrar más
ingredientes. Siguientes pasos posibles: más ejemplos, un modelo base mayor (3B) o mirar también la imagen.

## Motor: Claude o tu propio modelo

En `backend/.env`, `RECIPE_ENGINE=claude` usa Claude y `RECIPE_ENGINE=local` usa un modelo abierto
(Qwen2.5-7B, Apache 2.0) que corre en el Mac, sin coste de API ni datos que salgan del equipo.
El modelo local solo lee texto (descripción, transcripción y OCR): no mira los fotogramas ni estima
cantidades por la imagen, y es más propenso a saltarse pasos. Para comparar los dos con el mismo video:

```bash
cd backend
.venv/bin/python -m app.cli "<enlace>" --motor local
.venv/bin/python -m app.cli "<enlace>" --motor claude
```

Para publicarla con el motor local hace falta un servidor con GPU (convertir el modelo a vLLM o llama.cpp).

## 1. Servidor

Necesitas una clave de la API de Claude: https://platform.claude.com → Settings → API Keys.

```bash
cd backend
cp .env.example .env          # y pega tu clave en ANTHROPIC_API_KEY
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Probar sin la app, desde la terminal:

```bash
cd backend
.venv/bin/python -m app.cli "https://www.youtube.com/shorts/…"
```

Si empiezas en otro ordenador: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
y además `ffmpeg` instalado (`brew install ffmpeg`).

## 2. App

**Web** (la forma más rápida de probar):

```bash
cd mobile
npx expo start --web
```

**iPhone / Android**: la opción de «Compartir» necesita código nativo, así que no funciona en Expo Go.
Hay que crear una *development build*:

```bash
cd mobile
echo "EXPO_PUBLIC_API_URL=http://<IP-de-tu-Mac>:8000" > .env   # el teléfono no entiende "localhost"
npx expo run:ios        # requiere Xcode
npx expo run:android    # requiere Android Studio
```

o compilar en la nube sin Xcode/Android Studio: `npx eas-cli@latest build --profile development`.

## Pendiente antes de publicarla

- **Nombre e identificadores**: cambiar `Recetas` y `com.recetas.app` en `mobile/app.json` por los definitivos.
- **Servidor en la nube**: los trabajos se guardan en memoria; hace falta base de datos + cola (p. ej. Postgres + Redis)
  y un servidor con `ffmpeg` (Railway, Fly.io, Render, una VM…).
- **Cuentas y límites**: cada análisis cuesta dinero (Claude + servidor). Hacen falta usuarios, límite por usuario
  y probablemente un plan de pago.
- **Instagram y Facebook** suelen bloquear descargas desde servidores en la nube; puede hacer falta
  `YTDLP_COOKIES_FILE` o un proveedor de descargas.
- **Términos de uso**: descargar videos de estas plataformas va contra sus condiciones. Conviene asesoría legal
  antes de publicar (la app no guarda ni redistribuye los videos, solo la receta y el enlace).
- **Tiendas**: cuenta de Apple Developer (99 USD/año) y de Google Play (25 USD, pago único).

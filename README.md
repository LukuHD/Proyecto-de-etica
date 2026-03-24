# Proyecto de Ética — Motor Analítico de Detección de Desinformación

Aplicación de escritorio local que actúa como el **motor analítico** de una extensión de navegador diseñada para combatir la desinformación, noticias falsas y fraudes en redes sociales como Facebook.

---

## 🎯 ¿Qué hace este sistema?

Este backend local analiza publicaciones de Facebook en tiempo real usando **Inteligencia Artificial multimodal** (texto + imagen) para detectar:

- 💰 **Fraude financiero** — Esquemas Ponzi, phishing, inversiones falsas
- 🗳️ **Desinformación política** — Deep fakes electorales, propaganda
- 🖼️ **Contenido engañoso** — Imágenes descontextualizadas, clickbait
- 😱 **Manipulación emocional** — Lenguaje de miedo, urgencia artificial
- ✅ **Publicación segura** — Contenido legítimo sin señales maliciosas

La característica más importante es la **fusión cognitiva multimodal**: cruzar el texto con la imagen para detectar descontextualización deliberada (ej: foto de 2019 presentada como noticia de hoy).

---

## 🏗️ Arquitectura del Proyecto

```
backend/
├── app/
│   ├── main.py                  # Punto de entrada FastAPI + ciclo de vida
│   ├── config.py                # Configuración centralizada
│   ├── routers/
│   │   └── analysis.py          # Controladores HTTP (endpoints)
│   ├── schemas/
│   │   └── post_schema.py       # Modelos Pydantic de validación
│   ├── utils/
│   │   ├── db_manager.py        # Gestor de base de datos JSON local
│   │   └── hasher.py            # Generación de huellas SHA-256
│   └── ai_engine/
│       ├── analyzer.py          # Orquestador multimodal (fusión cognitiva)
│       ├── text_analyzer.py     # Análisis NLP (patrones + transformers)
│       └── vision_analyzer.py   # Análisis visual (VLM moondream2)
├── scripts/
│   └── continuous_learning.py   # Extracción de corpus para fine-tuning
├── tests/
│   ├── test_hasher.py           # Tests unitarios del hasher
│   ├── test_db_manager.py       # Tests de integración de la DB
│   ├── test_text_analyzer.py    # Tests del analizador NLP
│   └── test_api.py              # Tests end-to-end de la API
├── data/                        # Base de datos JSON local (auto-generada)
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 Instalación y Puesta en Marcha

### Prerrequisitos
- Python 3.10+
- 4GB RAM mínimo (8GB recomendado para el VLM)
- GPU opcional (mejora significativamente la velocidad del VLM)

### 1. Instalar dependencias

```bash
cd backend
pip install -r requirements.txt
```

**Con soporte GPU (CUDA 11.8):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Iniciar el servidor

```bash
# Opción A: Usando uvicorn directamente
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Opción B: Usando el módulo Python
python -m app.main
```

El servidor estará disponible en `http://127.0.0.1:8000`.

### 3. Verificar que funciona

```bash
curl http://127.0.0.1:8000/health
# → {"status": "ok", "service": "Desinformación Shield", "version": "1.0.0"}
```

---

## 📡 API Reference

### POST `/api/v1/analyze` — Analizar publicación

**Cuerpo de la petición (JSON):**
```json
{
  "post_text": "¡¡GANA $5000 DIARIOS TRABAJANDO DESDE CASA!! Sin experiencia.",
  "author_name": "Juan Pérez",
  "post_timestamp": "2024-06-15T14:30:00Z",
  "image_base64": "<base64_puro_sin_prefijo>",
  "image_url": "https://example.com/imagen.jpg"
}
```
> Nota: `image_base64` e `image_url` son opcionales. Si ambos se proporcionan, `image_base64` tiene prioridad.

**Respuesta exitosa (200 OK):**
```json
{
  "post_hash": "a3f8b2c1d4e5...",
  "category": "fraude_financiero",
  "confidence": 0.93,
  "explanation": "El texto emplea tácticas clásicas de fraude financiero...",
  "text_analysis": {
    "detected_patterns": ["fraude_financiero"],
    "sentiment_score": -0.4,
    "manipulation_indicators": ["alto_numero_exclamaciones: 5"],
    "credibility_signals": [],
    "text_confidence": 0.93
  },
  "vision_analysis": {
    "image_description": "...",
    "detected_objects": [],
    "manipulation_detected": false,
    "geographic_context": null,
    "temporal_context": null,
    "vision_confidence": 0.0
  },
  "multimodal_discrepancies": [],
  "cached": false,
  "analyzed_at": "2024-06-15T14:31:05Z"
}
```

### GET `/api/v1/stats` — Estadísticas

```bash
curl http://127.0.0.1:8000/api/v1/stats
```

### GET `/api/v1/history` — Historial paginado

```bash
curl "http://127.0.0.1:8000/api/v1/history?page=1&page_size=20&category_filter=fraude_financiero"
```

### GET `/health` — Verificación de estado

```bash
curl http://127.0.0.1:8000/health
```

**Documentación interactiva (Swagger UI):** `http://127.0.0.1:8000/docs`

---

## 🧠 Modelos de IA

### Análisis de Texto (NLP)
- **Capa 1 — Análisis de Patrones (siempre activo):** Diccionarios léxicos + expresiones regulares especializadas en español para detectar fraude, manipulación y desinformación.
- **Capa 2 — Transformers (opcional):** `distilbert-base-multilingual-cased-sentiments-student` (~250MB) para análisis de sentimiento. Se activa automáticamente si `transformers` está instalado.

### Análisis Visual (VLM)
- **Modelo:** `moondream2` (vikhyatk/moondream2) — 1.87B parámetros
- **Requisitos:** ~2GB RAM, ejecutable en CPU
- Se descarga automáticamente de Hugging Face en el primer uso
- Si no está disponible, el sistema degrada graciosamente a análisis de metadatos

### Fusión Cognitiva Multimodal
Motor basado en reglas semánticas que cruza el texto con la descripción de la imagen para detectar:
- Discrepancias geográficas (texto menciona México, imagen muestra Europa)
- Discrepancias temporales (texto dice "hoy", imagen tiene señales de ser de 2015)
- Discrepancias de gravedad (texto describe catástrofe, imagen es cotidiana)
- Imagen manipulada + patrones de fraude en texto (sinergia negativa)

---

## 🗄️ Base de Datos JSON Local

El sistema persiste todos los análisis en `data/analysis_database.json` — sin necesidad de instalar ningún motor de base de datos.

**Características:**
- Indexada por hash SHA-256 (búsqueda O(1))
- Escritura atómica (rename) para prevenir corrupción
- Lock asíncrono para protección contra escrituras concurrentes
- Auto-recuperación ante archivos corruptos (crea respaldo)

---

## 🔄 Aprendizaje Continuo

Ejecutar periódicamente para extraer datos de entrenamiento del historial:

```bash
# Extracción completa con exportación
python scripts/continuous_learning.py

# Solo estadísticas, sin exportar
python scripts/continuous_learning.py --stats-only

# Con umbrales personalizados
python scripts/continuous_learning.py --min-confidence 0.5 --output-dir /ruta/corpus
```

Los archivos generados en `data/training_corpus/`:
- `training_data_YYYYMMDD.jsonl` — Corpus para fine-tuning con Hugging Face
- `multimodal_training_YYYYMMDD.jsonl` — Corpus para el motor de fusión
- `low_confidence_cases_YYYYMMDD.csv` — Casos para revisión humana
- `corpus_report_YYYYMMDD.json` — Estadísticas y recomendaciones
- `FINETUNING_INSTRUCTIONS.md` — Guía completa de fine-tuning con LoRA/QLoRA

---

## 🧪 Tests

```bash
cd backend
python -m pytest tests/ -v
# → 57 passed in 0.50s
```

---

## ⚙️ Variables de Entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Host del servidor |
| `PORT` | `8000` | Puerto del servidor |
| `FORCE_CPU` | `false` | Forzar CPU aunque haya GPU |
| `NLP_MODEL_ID` | `lxyuan/...` | Modelo NLP de sentimiento |
| `VLM_MODEL_ID` | `vikhyatk/moondream2` | Modelo VLM de visión |
| `LOW_CONFIDENCE_THRESHOLD` | `0.65` | Umbral de baja confianza |
| `HIGH_CONFIDENCE_THRESHOLD` | `0.90` | Umbral de alta confianza |


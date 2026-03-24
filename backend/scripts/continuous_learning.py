"""
continuous_learning.py — Script de Aprendizaje Continuo y Extracción de Corpus.

Este script es el corazón del sistema de mejora continua del modelo de IA.
Su responsabilidad es:
  1. Leer el historial completo de análisis del archivo JSON de base de datos.
  2. Identificar y extraer casos de alto valor para el reentrenamiento:
     a. Casos límite (baja confianza) — el modelo no está seguro → aprendizaje valioso.
     b. Patrones nuevos de fraude — categorías emergentes detectadas recientemente.
     c. Posts con múltiples discrepancias multimodales — datos de fusión valiosos.
  3. Exportar el corpus extraído en dos formatos:
     a. JSONL (JSON Lines) — formato estándar para fine-tuning en Hugging Face.
     b. CSV — para análisis humano y anotación manual de etiquetas.
  4. Generar un reporte de estadísticas del corpus extraído.

¿Por qué aprendizaje continuo?
  Los patrones de desinformación evolucionan constantemente. Los actores maliciosos
  adaptan sus tácticas para evadir los sistemas de detección actuales. Sin actualización
  periódica, cualquier modelo de IA se vuelve obsoleto en semanas o meses.

Uso:
  # Extracción programada (cron job o Task Scheduler de Windows):
  python scripts/continuous_learning.py

  # Con opciones específicas:
  python scripts/continuous_learning.py --min-confidence 0.5 --output-dir /ruta/corpus

  # Solo estadísticas, sin exportar datos:
  python scripts/continuous_learning.py --stats-only

Integración con flujo de fine-tuning de Hugging Face:
  Los archivos JSONL generados son directamente compatibles con:
    • Hugging Face Trainer API (trainer.train())
    • TRL (Transformer Reinforcement Learning) para RLHF
    • LoRA fine-tuning con PEFT library
  Ver la sección "Guía de Fine-tuning" al final de este archivo.
"""

import argparse   # Para línea de comandos
import csv        # Para exportación CSV
import json       # Para leer/escribir JSON
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Configuración del logger ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("continuous_learning")

# ── Rutas por defecto ───────────────────────────────────────────────────────
# Estas rutas son relativas al directorio `backend/` donde vive este script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent          # backend/
_DB_FILE = _PROJECT_ROOT / "data" / "analysis_database.json"
_CORPUS_OUTPUT_DIR = _PROJECT_ROOT / "data" / "training_corpus"

# ── Umbrales de selección ───────────────────────────────────────────────────
# Parámetros que controlan qué registros se consideran "casos de aprendizaje"
_DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.65  # Confianza < 65% → caso límite
_DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.90  # Confianza > 90% → ejemplo claro de entrenamiento
_DEFAULT_MIN_RECORDS_FOR_EXPORT = 5        # Mínimo de registros para generar corpus útil


def load_database(db_path: Path) -> Dict[str, Any]:
    """
    Carga el archivo JSON de base de datos completo.

    Args:
        db_path: Ruta al archivo JSON de base de datos.

    Returns:
        Diccionario con toda la base de datos.

    Raises:
        FileNotFoundError: Si el archivo no existe (el servidor nunca se ejecutó).
        json.JSONDecodeError: Si el archivo está corrupto.
    """
    if not db_path.exists():
        logger.error(
            "❌ Base de datos no encontrada en: %s\n"
            "   ¿El servidor backend se ha ejecutado al menos una vez?",
            db_path,
        )
        raise FileNotFoundError(f"Base de datos no encontrada: {db_path}")

    logger.info("📖 Cargando base de datos desde: %s", db_path)

    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_records = len(data.get("records", {}))
    logger.info("✅ Base de datos cargada — %d registros totales.", total_records)

    return data


def extract_training_cases(
    records: Dict[str, Any],
    low_confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    high_confidence_threshold: float = _DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Clasifica y extrae los registros más valiosos para el reentrenamiento.

    Estrategia de selección:
      1. CASOS LÍMITE (baja confianza): El modelo está inseguro → ejemplos donde
         necesita más aprendizaje. Son los más valiosos para reducir la incertidumbre.

      2. EJEMPLOS POSITIVOS CLAROS (alta confianza + categoría peligrosa):
         El modelo está muy seguro de que es fraude/desinformación → buenos ejemplos
         positivos para reinforcement durante el fine-tuning.

      3. EJEMPLOS NEGATIVOS CLAROS (alta confianza + publicación segura):
         El modelo está muy seguro de que es contenido legítimo → ejemplos negativos
         para enseñar al modelo qué NO debe marcar.

      4. CASOS MULTIMODALES (con discrepancias imagen-texto):
         Ejemplos donde la fusión cognitiva fue activada → datos para mejorar
         específicamente la capacidad de detección de descontextualización.

      5. PATRONES EMERGENTES (categorías con pocos ejemplos):
         Si hay muy pocos ejemplos de una categoría (p.ej., solo 2 casos de
         "desinformacion_politica"), esos son valiosos para balancear el dataset.

    Args:
        records:                    Diccionario de registros de la DB.
        low_confidence_threshold:   Umbral por debajo del cual un caso es "límite".
        high_confidence_threshold:  Umbral por encima del cual un caso es "claro".

    Returns:
        Diccionario con listas de casos por tipo de aprendizaje.
    """
    cases = {
        "low_confidence": [],      # Casos límite para revisión humana
        "high_confidence_positive": [],  # Ejemplos claros de contenido peligroso
        "high_confidence_negative": [],  # Ejemplos claros de contenido seguro
        "multimodal_discrepancy": [],   # Casos con fusión imagen-texto activada
        "emerging_patterns": [],    # Patrones con pocos ejemplos
    }

    # ── Contar distribución por categoría para detectar categorías raras ────
    category_counts: Dict[str, int] = {}
    for record in records.values():
        cat = record.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Una categoría es "rara" si tiene menos del 5% del total de registros
    total = len(records)
    rare_threshold = max(int(total * 0.05), 3)  # Mínimo 3 registros para ser "raro"

    logger.info("📊 Distribución de categorías en la DB:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        rarity_marker = " ⚠️ RARA" if count <= rare_threshold else ""
        logger.info("   • %s: %d registros (%.1f%%)%s", cat, count, count/total*100, rarity_marker)

    # ── Clasificar cada registro ────────────────────────────────────────────
    for record_hash, record in records.items():
        confidence = record.get("confidence", 0.5)
        category = record.get("category", "unknown")
        discrepancies = record.get("multimodal_discrepancies", [])

        # Caso 1: Baja confianza → caso límite para revisión
        if confidence < low_confidence_threshold:
            cases["low_confidence"].append({
                "hash": record_hash,
                "post_text": record.get("post_text", ""),
                "author": record.get("author_name", ""),
                "predicted_category": category,
                "confidence": confidence,
                "reason": "low_confidence_review_needed",
                "analyzed_at": record.get("analyzed_at", ""),
                "manipulation_indicators": record.get("manipulation_indicators", []),
            })

        # Caso 2: Alta confianza + contenido peligroso → ejemplo positivo claro
        elif confidence >= high_confidence_threshold and category != "publicacion_segura":
            cases["high_confidence_positive"].append({
                "hash": record_hash,
                "post_text": record.get("post_text", ""),
                "label": category,
                "confidence": confidence,
                "text_patterns": record.get("text_patterns", []),
                "explanation": record.get("explanation", ""),
                "training_format": _format_for_training(record, label=category),
            })

        # Caso 3: Alta confianza + contenido seguro → ejemplo negativo claro
        elif confidence >= high_confidence_threshold and category == "publicacion_segura":
            cases["high_confidence_negative"].append({
                "hash": record_hash,
                "post_text": record.get("post_text", ""),
                "label": "publicacion_segura",
                "confidence": confidence,
                "training_format": _format_for_training(record, label="publicacion_segura"),
            })

        # Caso 4: Tiene discrepancias multimodales → caso de fusión especial
        if discrepancies:
            cases["multimodal_discrepancy"].append({
                "hash": record_hash,
                "post_text": record.get("post_text", ""),
                "image_description": record.get("image_description", ""),
                "category": category,
                "confidence": confidence,
                "discrepancies": discrepancies,
                "training_format": _format_multimodal_for_training(record, discrepancies),
            })

        # Caso 5: Categoría rara → ejemplo para balanceo del dataset
        if category_counts.get(category, 0) <= rare_threshold:
            cases["emerging_patterns"].append({
                "hash": record_hash,
                "post_text": record.get("post_text", ""),
                "category": category,
                "confidence": confidence,
                "note": f"Categoría rara: solo {category_counts.get(category, 0)} ejemplos",
            })

    return cases


def _format_for_training(record: Dict[str, Any], label: str) -> Dict[str, Any]:
    """
    Formatea un registro para el entrenamiento en formato de clasificación de texto.

    El formato sigue la convención de Hugging Face para datasets de clasificación:
    {
      "text": "Texto del post a clasificar",
      "label": "categoria_asignada",
      "label_id": 0  // ID numérico para el modelo
    }

    Args:
        record: Registro de la base de datos.
        label:  Etiqueta correcta (puede diferir del campo "category" si fue
                corregida manualmente).

    Returns:
        Diccionario en formato Hugging Face para fine-tuning de clasificación.
    """
    # Mapeo de etiquetas a IDs numéricos para el modelo de clasificación
    label_to_id = {
        "publicacion_segura": 0,
        "contenido_enganoso": 1,
        "manipulacion_emocional": 2,
        "desinformacion_politica": 3,
        "fraude_financiero": 4,
    }

    return {
        "text": record.get("post_text", ""),
        "label": label,
        "label_id": label_to_id.get(label, -1),
        "metadata": {
            "author": record.get("author_name", ""),
            "confidence": record.get("confidence", 0),
            "patterns_detected": record.get("text_patterns", []),
            "analyzed_at": record.get("analyzed_at", ""),
        },
    }


def _format_multimodal_for_training(
    record: Dict[str, Any],
    discrepancies: List[str],
) -> Dict[str, Any]:
    """
    Formatea un registro con discrepancias para entrenamiento multimodal.

    Para el fine-tuning del componente de fusión cognitiva, necesitamos
    pares (texto, descripción_imagen) con las discrepancias etiquetadas
    como ground truth.

    Returns:
        Diccionario en formato para entrenamiento de detección de discrepancias.
    """
    return {
        "text_claim": record.get("post_text", ""),
        "image_description": record.get("image_description", ""),
        "discrepancies_found": discrepancies,
        "num_discrepancies": len(discrepancies),
        "is_misleading": len(discrepancies) > 0,
        "category": record.get("category", ""),
        "confidence": record.get("confidence", 0),
    }


def generate_statistics_report(
    all_records: Dict[str, Any],
    extracted_cases: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Genera un reporte estadístico completo del corpus extraído.

    Este reporte es útil para:
      • Decidir si hay suficientes datos para un ciclo de fine-tuning.
      • Identificar desequilibrios en el dataset (pocas muestras de alguna categoría).
      • Monitorear la evolución del rendimiento del modelo a lo largo del tiempo.

    Returns:
        Diccionario con estadísticas detalladas del corpus.
    """
    total_records = len(all_records)

    # Distribución de confianzas
    confidences = [r.get("confidence", 0) for r in all_records.values()]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    low_conf_count = sum(1 for c in confidences if c < _DEFAULT_LOW_CONFIDENCE_THRESHOLD)
    high_conf_count = sum(1 for c in confidences if c >= _DEFAULT_HIGH_CONFIDENCE_THRESHOLD)

    return {
        "report_generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "total_records_in_db": total_records,
        "corpus_summary": {
            "low_confidence_cases": len(extracted_cases["low_confidence"]),
            "high_confidence_positive": len(extracted_cases["high_confidence_positive"]),
            "high_confidence_negative": len(extracted_cases["high_confidence_negative"]),
            "multimodal_discrepancy_cases": len(extracted_cases["multimodal_discrepancy"]),
            "emerging_pattern_cases": len(extracted_cases["emerging_patterns"]),
        },
        "confidence_distribution": {
            "average": round(avg_confidence, 4),
            "low_confidence_count": low_conf_count,
            "high_confidence_count": high_conf_count,
            "percentage_needing_review": round(low_conf_count / total_records * 100, 1) if total_records > 0 else 0,
        },
        "recommendations": _generate_recommendations(extracted_cases, total_records),
    }


def _generate_recommendations(
    cases: Dict[str, List[Dict[str, Any]]],
    total_records: int,
) -> List[str]:
    """
    Genera recomendaciones basadas en el análisis del corpus.

    Returns:
        Lista de strings con recomendaciones accionables.
    """
    recommendations = []

    # Recomendación 1: ¿Suficientes datos para fine-tuning?
    total_training = (
        len(cases["high_confidence_positive"]) +
        len(cases["high_confidence_negative"])
    )

    if total_training < 50:
        recommendations.append(
            f"⚠️  Solo {total_training} ejemplos de alta confianza disponibles. "
            "Se recomienda al menos 100-500 ejemplos por categoría para fine-tuning efectivo. "
            "Continúa acumulando datos durante 2-4 semanas más antes del primer ciclo de entrenamiento."
        )
    elif total_training < 200:
        recommendations.append(
            f"ℹ️  {total_training} ejemplos disponibles. "
            "Suficiente para un fine-tuning ligero con LoRA (Low-Rank Adaptation). "
            "Considera usar QLoRA para reducir requerimientos de memoria."
        )
    else:
        recommendations.append(
            f"✅ {total_training} ejemplos de alta calidad disponibles. "
            "Dataset suficiente para fine-tuning estándar con Hugging Face Trainer."
        )

    # Recomendación 2: Casos límite para revisión humana
    low_conf = len(cases["low_confidence"])
    if low_conf > 10:
        recommendations.append(
            f"📋 {low_conf} casos de baja confianza requieren revisión humana. "
            "Revisa el archivo 'low_confidence_cases.csv' y etiqueta manualmente "
            "la categoría correcta para estos casos antes del próximo ciclo de entrenamiento."
        )

    # Recomendación 3: Casos multimodales
    multimodal = len(cases["multimodal_discrepancy"])
    if multimodal > 5:
        recommendations.append(
            f"🔀 {multimodal} casos con discrepancias imagen-texto detectados. "
            "Estos son especialmente valiosos para mejorar el motor de fusión multimodal. "
            "Considera fine-tuning específico del módulo de fusión cognitiva."
        )

    # Recomendación 4: Patrones emergentes
    emerging = len(cases["emerging_patterns"])
    if emerging > 0:
        recommendations.append(
            f"🆕 {emerging} casos de patrones emergentes/raros detectados. "
            "Añade más variaciones de estos patrones al diccionario léxico del text_analyzer.py "
            "para mejorar la cobertura sin necesidad de fine-tuning inmediato."
        )

    return recommendations


def export_corpus(
    extracted_cases: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    report: Dict[str, Any],
) -> None:
    """
    Exporta el corpus extraído en múltiples formatos.

    Formatos de salida:
      1. JSONL (JSON Lines) para fine-tuning directo con Hugging Face.
      2. CSV para revisión humana y anotación manual.
      3. JSON de reporte de estadísticas.

    Args:
        extracted_cases: Casos clasificados por tipo de aprendizaje.
        output_dir:      Directorio de salida para los archivos.
        report:          Reporte de estadísticas a incluir.
    """
    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Exportar JSONL para fine-tuning ──────────────────────────────────
    # Combinar todos los ejemplos de entrenamiento en un solo archivo JSONL
    all_training_examples = (
        [c["training_format"] for c in extracted_cases["high_confidence_positive"] if "training_format" in c] +
        [c["training_format"] for c in extracted_cases["high_confidence_negative"] if "training_format" in c]
    )

    if all_training_examples:
        jsonl_path = output_dir / f"training_data_{timestamp}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for example in all_training_examples:
                # JSONL: un objeto JSON por línea
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info("✅ Corpus JSONL exportado: %s (%d ejemplos)", jsonl_path, len(all_training_examples))

    # ── 2. Exportar JSONL multimodal ─────────────────────────────────────────
    multimodal_examples = [
        c["training_format"] for c in extracted_cases["multimodal_discrepancy"]
        if "training_format" in c
    ]

    if multimodal_examples:
        multimodal_path = output_dir / f"multimodal_training_{timestamp}.jsonl"
        with open(multimodal_path, "w", encoding="utf-8") as f:
            for example in multimodal_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info("✅ Corpus multimodal JSONL exportado: %s (%d ejemplos)", multimodal_path, len(multimodal_examples))

    # ── 3. Exportar CSV para revisión humana ────────────────────────────────
    low_conf_cases = extracted_cases["low_confidence"]
    if low_conf_cases:
        csv_path = output_dir / f"low_confidence_cases_{timestamp}.csv"
        fieldnames = ["hash", "post_text", "author", "predicted_category",
                      "confidence", "reason", "analyzed_at", "correct_label"]

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for case in low_conf_cases:
                # Añadir columna vacía "correct_label" para que el revisor humano la complete
                case_with_label = {**case, "correct_label": ""}
                writer.writerow(case_with_label)

        logger.info(
            "✅ CSV de casos de baja confianza exportado: %s (%d casos para revisión)",
            csv_path, len(low_conf_cases)
        )

    # ── 4. Exportar reporte de estadísticas ─────────────────────────────────
    report_path = output_dir / f"corpus_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("✅ Reporte exportado: %s", report_path)

    # ── 5. Instrucciones de fine-tuning ─────────────────────────────────────
    instructions_path = output_dir / "FINETUNING_INSTRUCTIONS.md"
    if not instructions_path.exists():
        _write_finetuning_instructions(instructions_path)


def _write_finetuning_instructions(path: Path) -> None:
    """
    Escribe un archivo de instrucciones para el proceso de fine-tuning.
    Este archivo se crea una sola vez y sirve como guía permanente.
    """
    instructions = """\
# Guía de Fine-tuning del Modelo de Detección de Desinformación

## ¿Cuándo hacer fine-tuning?

Realizar un ciclo de fine-tuning cuando:
- [ ] Tienes al menos 100 ejemplos de alta confianza por categoría
- [ ] El reporte muestra >15% de casos de baja confianza
- [ ] Han pasado 4+ semanas desde el último ciclo
- [ ] Se detectaron patrones emergentes nuevos

## Prerequisitos

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
```

## Opción A: Fine-tuning con LoRA (recomendado para hardware doméstico)

LoRA (Low-Rank Adaptation) permite fine-tuning en GPU con solo 4-8GB VRAM.

```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Cargar dataset JSONL generado por este script
dataset = load_dataset("json", data_files={"train": "training_data_XXXXXXXX.jsonl"})

# 2. Cargar modelo base
model_id = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=5)

# 3. Configurar LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,           # Rank de la adaptación (más alto = más capacidad, más memoria)
    lora_alpha=32,  # Factor de escalado
    target_modules=["q_lin", "v_lin"],  # Módulos a adaptar en DistilBERT
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# 4. Tokenizar
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset["train"].map(tokenize, batched=True)

# 5. Entrenamiento
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Ajustar según RAM disponible
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)
trainer.train()

# 6. Guardar el modelo fine-tuned
model.save_pretrained("./fine_tuned_model/final")
tokenizer.save_pretrained("./fine_tuned_model/final")
```

## Opción B: Fine-tuning con QLoRA (para hardware muy limitado)

QLoRA usa cuantización de 4-bit para reducir el uso de memoria a la mitad.

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Cargar modelo en 4-bit
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    num_labels=5,
)
# ... resto igual que Opción A
```

## Integración del modelo fine-tuned

Después del entrenamiento, actualiza el modelo en `text_analyzer.py`:

```python
# En _load_sentiment_model(), cambiar:
model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
# Por:
model="./fine_tuned_model/final"  # Ruta al modelo local fine-tuned
```

## Evaluación del modelo

Siempre evalúa el modelo en un conjunto de datos de prueba antes de reemplazar
el modelo en producción:

```python
from sklearn.metrics import classification_report

# ... hacer predicciones en test set
print(classification_report(y_true, y_pred, target_names=categories))
```

## Ciclo recomendado

1. `python scripts/continuous_learning.py` → Extrae corpus
2. Revisar manualmente `low_confidence_cases_*.csv` y añadir etiquetas correctas
3. Ejecutar fine-tuning con los datos etiquetados
4. Evaluar el nuevo modelo en un test set
5. Si mejora → reemplazar en producción
6. Volver al paso 1 en 4 semanas
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(instructions)
    logger.info("📚 Instrucciones de fine-tuning escritas en: %s", path)


def main(
    db_path: Path = _DB_FILE,
    output_dir: Path = _CORPUS_OUTPUT_DIR,
    low_confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    high_confidence_threshold: float = _DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    stats_only: bool = False,
) -> int:
    """
    Función principal del script de aprendizaje continuo.

    Args:
        db_path:                  Ruta a la base de datos JSON.
        output_dir:               Directorio de salida del corpus.
        low_confidence_threshold: Umbral de baja confianza.
        high_confidence_threshold: Umbral de alta confianza.
        stats_only:               Si True, solo muestra estadísticas sin exportar.

    Returns:
        Código de salida: 0 = éxito, 1 = error.
    """
    logger.info("=" * 70)
    logger.info("  Sistema de Aprendizaje Continuo — Motor de Detección de Desinformación")
    logger.info("=" * 70)

    # ── 1. Cargar base de datos ─────────────────────────────────────────────
    try:
        db_data = load_database(db_path)
    except FileNotFoundError:
        return 1
    except json.JSONDecodeError as e:
        logger.error("❌ Base de datos corrupta: %s", e)
        return 1

    records = db_data.get("records", {})
    total = len(records)

    if total < _DEFAULT_MIN_RECORDS_FOR_EXPORT:
        logger.warning(
            "⚠️  Solo %d registros en la base de datos. "
            "Se necesitan al menos %d para generar un corpus de entrenamiento útil. "
            "Continúa usando el sistema para acumular más datos.",
            total,
            _DEFAULT_MIN_RECORDS_FOR_EXPORT,
        )
        if not stats_only:
            return 0

    # ── 2. Extraer casos de entrenamiento ───────────────────────────────────
    logger.info("\n🔍 Analizando registros para extracción de corpus de entrenamiento…")
    extracted_cases = extract_training_cases(
        records,
        low_confidence_threshold=low_confidence_threshold,
        high_confidence_threshold=high_confidence_threshold,
    )

    # ── 3. Generar reporte estadístico ──────────────────────────────────────
    report = generate_statistics_report(records, extracted_cases)

    logger.info("\n📊 REPORTE DE CORPUS:")
    logger.info("   Casos de baja confianza (para revisión): %d",
                len(extracted_cases["low_confidence"]))
    logger.info("   Ejemplos positivos claros: %d",
                len(extracted_cases["high_confidence_positive"]))
    logger.info("   Ejemplos negativos claros: %d",
                len(extracted_cases["high_confidence_negative"]))
    logger.info("   Casos multimodales: %d",
                len(extracted_cases["multimodal_discrepancy"]))
    logger.info("   Patrones emergentes: %d",
                len(extracted_cases["emerging_patterns"]))

    logger.info("\n💡 RECOMENDACIONES:")
    for rec in report.get("recommendations", []):
        logger.info("   %s", rec)

    if stats_only:
        logger.info("\n✅ Modo --stats-only: No se exportaron archivos.")
        return 0

    # ── 4. Exportar corpus ─────────────────────────────────────────────────
    logger.info("\n💾 Exportando corpus a: %s", output_dir)
    export_corpus(extracted_cases, output_dir, report)

    logger.info("\n✅ Proceso de extracción de corpus completado exitosamente.")
    logger.info("   Archivos disponibles en: %s", output_dir)
    return 0


# ── Punto de entrada con interfaz de línea de comandos ─────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Script de aprendizaje continuo para el sistema de detección "
            "de desinformación. Extrae casos de entrenamiento del historial "
            "de análisis para fine-tuning periódico del modelo de IA."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=_DB_FILE,
        help=f"Ruta a la base de datos JSON (default: {_DB_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_CORPUS_OUTPUT_DIR,
        help=f"Directorio de salida del corpus (default: {_CORPUS_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=_DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        help=f"Umbral de baja confianza (default: {_DEFAULT_LOW_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--high-confidence",
        type=float,
        default=_DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
        help=f"Umbral de alta confianza (default: {_DEFAULT_HIGH_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Solo mostrar estadísticas sin exportar archivos",
    )

    args = parser.parse_args()

    exit_code = main(
        db_path=args.db_path,
        output_dir=args.output_dir,
        low_confidence_threshold=args.min_confidence,
        high_confidence_threshold=args.high_confidence,
        stats_only=args.stats_only,
    )
    sys.exit(exit_code)

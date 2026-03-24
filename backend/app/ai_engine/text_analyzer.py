"""
text_analyzer.py — Submódulo de Análisis de Lenguaje Natural (NLP).

Responsabilidad: Analizar el texto de una publicación para detectar patrones
semánticos que indiquen desinformación, manipulación emocional, fraude financiero
o propaganda política.

Arquitectura del análisis de texto (dos capas complementarias):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  CAPA 1: Análisis Basado en Patrones (siempre activo)               │
  │  • Rápido, determinista, no requiere GPU                            │
  │  • Diccionarios léxicos especializados en español                   │
  │  • Expresiones regulares para patrones de fraude                    │
  │  • Análisis de rasgos lingüísticos (signos de exclamación, caps)    │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────┐
  │  CAPA 2: Modelo Transformers (activado si disponible)               │
  │  • Clasificación de sentimiento con modelo ligero BETO/DistilBERT   │
  │  • Carga lazy — solo se inicializa en el primer uso                 │
  │  • Fallback graceful si el modelo no está disponible                │
  └─────────────────────────────────────────────────────────────────────┘

Decisión arquitectónica — Carga lazy de modelos:
  Los modelos de transformers pueden tardar varios segundos en cargarse.
  Usamos inicialización lazy (se cargan al primer uso, no al importar el módulo)
  para que el servidor arranque instantáneamente y el primer análisis real
  pague el costo de carga. Los análisis subsiguientes usan el modelo en caché en RAM.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Intento de importación de transformers (dependencia opcional) ───────────
# Si transformers no está instalado, el análisis se degrada graciosamente
# a solo el análisis basado en patrones, que sigue siendo muy efectivo.
try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Librería 'transformers' disponible para análisis NLP avanzado.")
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "⚠️  'transformers' no instalado. El análisis NLP usará solo patrones léxicos. "
        "Para activar el análisis avanzado: pip install transformers torch"
    )

# ── Estado global del modelo de sentimiento (carga lazy) ───────────────────
# None = aún no cargado; se inicializa en el primer análisis.
_sentiment_pipeline = None


def _load_sentiment_model() -> Optional[Any]:
    """
    Carga el modelo de análisis de sentimiento de forma lazy.

    Modelo seleccionado: 'lxyuan/distilbert-base-multilingual-cased-sentiments-student'
    Justificación:
      • Multilingüe: funciona en español, inglés y otros idiomas.
      • DistilBERT es 40% más rápido y 60% más ligero que BERT base.
      • ~250MB de descarga — razonable para uso doméstico.
      • Alternativa si hay problemas de memoria: 'cardiffnlp/twitter-xlm-roberta-base-sentiment'

    Returns:
        Pipeline de HuggingFace listo para inferencia, o None si falla la carga.
    """
    global _sentiment_pipeline

    if not _TRANSFORMERS_AVAILABLE:
        return None

    if _sentiment_pipeline is not None:
        return _sentiment_pipeline  # Ya está cargado — no volver a cargar

    try:
        logger.info("🔄 Cargando modelo de análisis de sentimiento…")
        _sentiment_pipeline = hf_pipeline(
            task="text-classification",
            # Modelo multilingüe ligero optimizado para redes sociales
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            # Usar CPU por defecto para garantizar compatibilidad universal.
            # Cambiar a device=0 para usar GPU CUDA si está disponible.
            device=-1,  # -1 = CPU, 0 = primera GPU CUDA
            # Retornar scores de todas las etiquetas (positivo/neutro/negativo)
            return_all_scores=True,
            truncation=True,       # Truncar textos largos al límite del modelo
            max_length=512,        # Máximo de tokens soportado por DistilBERT
        )
        logger.info("✅ Modelo de sentimiento cargado exitosamente.")
        return _sentiment_pipeline
    except Exception as e:
        logger.error("❌ No se pudo cargar el modelo de sentimiento: %s", e)
        return None


# ── Diccionarios léxicos especializados ────────────────────────────────────
# Estos diccionarios están diseñados específicamente para el contexto
# hispanohablante y los patrones de fraude/desinformación en Facebook.

# Patrones de fraude financiero (esquemas Ponzi, phishing, inversiones falsas)
_FINANCIAL_FRAUD_PATTERNS = [
    r"gana[s]?\s+\$?\d+",               # "ganas $500", "gana 1000"
    r"trabaja[r]?\s+desde\s+casa",       # "trabajar desde casa"
    r"ingresos?\s+pasivos?",             # "ingreso pasivo"
    r"invierte?\s+(solo|tan solo)",      # "invierte solo $50"
    r"retiro\s+garantizado",             # promesa de retiro asegurado
    r"criptomoneda[s]?\s+(gratis|free)", # cripto gratis
    r"multiplica\s+tu\s+(dinero|capital)",
    r"sin\s+experiencia\s+previa",
    r"únete\s+(ahora|hoy)",
    r"plazas?\s+limitadas?",             # urgencia artificial
    r"oferta\s+(por tiempo|limitada)",
    r"haz\s+clic\s+aquí",               # call-to-action sospechoso
    r"contáctame\s+por\s+whatsapp",      # desvío a canal privado
    r"negocio\s+del\s+siglo",
    r"\d+%\s+de\s+ganancia",            # porcentajes de ganancia irreales
]

# Patrones de manipulación emocional
_EMOTIONAL_MANIPULATION_PATTERNS = [
    r"urgente[!]+",
    r"¡+alerta[!]*",
    r"no\s+te\s+lo\s+pierdas",
    r"comparte\s+antes\s+de\s+que\s+(lo\s+)?borren",  # conspiración de censura
    r"el\s+gobierno\s+(no\s+quiere|oculta|esconde)",
    r"te\s+están\s+mintiendo",
    r"la\s+(verdad\s+)?(que\s+)?(ocultan|no\s+dicen)",
    r"wake\s+up",
    r"¡+cuidado[!]*",
    r"peligro\s+inminente",
    r"esto\s+es\s+real",
    r"no\s+lo\s+verás\s+en\s+(los\s+medios|televisión|tv)",  # antiestablishment
    r"reenvía\s+a\s+(todos|tus\s+contactos)",                 # cadena de mensajes
]

# Patrones de desinformación política
_POLITICAL_DISINFO_PATTERNS = [
    r"fraude\s+electoral",
    r"voto\s+(robado|trampa|manipulado)",
    r"dictadura",
    r"gobierno\s+(corrupt[ao]|fascist[ao]|comunist[ao])",
    r"conspiración\s+(global|mundial|sionista|masónica)",
    r"nuevo\s+orden\s+mundial",
    r"chip\s+(en\s+la\s+vacuna|5g)",
    r"vacuna\s+(mata|enferma|arn)",
    r"noticias\s+falsas\s+de\s+los\s+medios",
]

# Indicadores lingüísticos de baja credibilidad
_LOW_CREDIBILITY_INDICATORS = [
    r"!{3,}",              # Tres o más signos de exclamación seguidos
    r"¡{3,}",
    r"\?{3,}",             # Tres o más signos de interrogación
    r"[A-ZÁÉÍÓÚÑ]{10,}",  # Diez o más letras mayúsculas consecutivas (GRITANDO)
    r"\.{4,}",             # Cuatro o más puntos suspensivos
]

# Señales de credibilidad (el post podría ser legítimo)
_CREDIBILITY_SIGNALS = [
    r"según\s+(el\s+estudio|la\s+investigación|fuentes?|reuters|ap\s+news)",
    r"fuente[s]?:\s+https?://",          # Cita con URL
    r"publicado\s+en\s+(nature|science|lancet|pubmed)",
    r"de\s+acuerdo\s+con\s+(la\s+oms|who|cdc|onu|un\b)",
    r"verificado\s+por",
    r"fact.?check",
]


def _run_pattern_analysis(text: str) -> Dict[str, Any]:
    """
    Ejecuta el análisis de patrones léxicos sobre el texto.

    Este análisis es determinista, rápido y no requiere GPU.
    Usa expresiones regulares compiladas para máxima eficiencia.

    Args:
        text: Texto de la publicación (ya normalizado a minúsculas).

    Returns:
        Diccionario con todos los patrones detectados y clasificados.
    """
    text_lower = text.lower()

    detected_patterns: List[str] = []
    manipulation_indicators: List[str] = []
    credibility_signals: List[str] = []

    # ── Búsqueda de patrones de fraude financiero ───────────────────────────
    for pattern in _FINANCIAL_FRAUD_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            detected_patterns.append("fraude_financiero")
            break  # Basta con encontrar uno para clasificar esta categoría

    # ── Búsqueda de patrones de manipulación emocional ─────────────────────
    emotional_hits = []
    for pattern in _EMOTIONAL_MANIPULATION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            emotional_hits.append(pattern)

    if emotional_hits:
        detected_patterns.append("manipulacion_emocional")
        manipulation_indicators.extend([f"patron: {p}" for p in emotional_hits[:3]])

    # ── Búsqueda de patrones de desinformación política ─────────────────────
    for pattern in _POLITICAL_DISINFO_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            detected_patterns.append("desinformacion_politica")
            break

    # ── Indicadores de baja credibilidad (estilo de escritura) ─────────────
    for pattern in _LOW_CREDIBILITY_INDICATORS:
        if re.search(pattern, text):  # Usar texto original para detectar CAPS
            manipulation_indicators.append(f"estilo_alarmista: {pattern}")

    # ── Señales de credibilidad ─────────────────────────────────────────────
    for pattern in _CREDIBILITY_SIGNALS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            credibility_signals.append(f"fuente_verificable: {pattern}")

    # ── Análisis de densidad de signos de exclamación ──────────────────────
    # Más de 3 signos de exclamación en un texto corto es una señal de alerta
    exclamation_count = text.count("!") + text.count("¡")
    if exclamation_count > 3:
        manipulation_indicators.append(
            f"alto_numero_exclamaciones: {exclamation_count}"
        )

    # ── Análisis de proporción de MAYÚSCULAS ───────────────────────────────
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if caps_ratio > 0.4:  # Más del 40% en mayúsculas
            manipulation_indicators.append(
                f"exceso_mayusculas: {caps_ratio:.1%}"
            )

    return {
        "detected_patterns": list(set(detected_patterns)),  # Eliminar duplicados
        "manipulation_indicators": manipulation_indicators,
        "credibility_signals": credibility_signals,
        "exclamation_count": exclamation_count,
    }


def _run_sentiment_analysis(text: str) -> Tuple[float, str]:
    """
    Ejecuta el análisis de sentimiento usando el modelo de transformers.

    Mapeo de etiquetas del modelo a puntuación normalizada:
      positive  →  +1.0 (contenido positivo/neutro)
      neutral   →   0.0
      negative  → -1.0 (contenido negativo — no implica desinformación per se)

    La negatividad extrema combinada con patrones de manipulación SÍ es señal
    de alerta (miedo inducido artificialmente, pánico informativo).

    Args:
        text: Texto de la publicación.

    Returns:
        Tupla (sentiment_score, label) donde score ∈ [-1.0, 1.0].
    """
    model = _load_sentiment_model()

    if model is None:
        # Fallback: estimación heurística basada en palabras clave
        return _heuristic_sentiment(text)

    try:
        # Truncar texto largo para evitar OOM en modelos con ventana de 512 tokens
        truncated_text = text[:1000]

        # Ejecutar inferencia — retorna lista de dicts con label y score
        results = model(truncated_text)

        # El pipeline con return_all_scores=True retorna [[{label, score}, ...]]
        if results and isinstance(results[0], list):
            scores_dict = {item["label"]: item["score"] for item in results[0]}
        elif results and isinstance(results[0], dict):
            scores_dict = {results[0]["label"]: results[0]["score"]}
        else:
            return _heuristic_sentiment(text)

        # Normalizar a escala [-1, +1]
        positive = scores_dict.get("positive", scores_dict.get("LABEL_1", 0.0))
        negative = scores_dict.get("negative", scores_dict.get("LABEL_0", 0.0))
        neutral = scores_dict.get("neutral", scores_dict.get("LABEL_2", 0.0))

        # Sentimiento neto: positivo - negativo (neutral actúa como moderador)
        sentiment_score = positive - negative

        dominant_label = max(scores_dict, key=lambda k: scores_dict[k])
        return sentiment_score, dominant_label

    except Exception as e:
        logger.warning("⚠️  Error en análisis de sentimiento transformer: %s. Usando heurística.", e)
        return _heuristic_sentiment(text)


def _heuristic_sentiment(text: str) -> Tuple[float, str]:
    """
    Estimación heurística de sentimiento basada en léxico cuando el
    modelo transformer no está disponible.

    Usa listas de palabras positivas/negativas comunes en español para
    estimar el tono general del texto.
    """
    text_lower = text.lower()

    positive_words = [
        "excelente", "bueno", "increíble", "feliz", "alegre", "amor",
        "bendición", "esperanza", "éxito", "logro", "hermoso", "positivo",
    ]
    negative_words = [
        "malo", "terrible", "horrible", "miedo", "peligro", "muerto",
        "catástrofe", "desastre", "robo", "fraude", "mentira", "engaño",
        "urgente", "alerta", "cuidado", "trampa",
    ]

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count + neg_count == 0:
        return 0.0, "neutral"

    score = (pos_count - neg_count) / (pos_count + neg_count)
    label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
    return score, label


async def analyze_text(text: str) -> Dict[str, Any]:
    """
    Función principal del submódulo de análisis de texto.

    Orquesta el análisis basado en patrones y el análisis de sentimiento
    con transformers, combinando sus resultados en un único diccionario
    estructurado.

    Es async para integrarse naturalmente con el sistema FastAPI, aunque
    el análisis de texto en sí no requiere I/O asíncrono.

    Args:
        text: Texto completo de la publicación a analizar.

    Returns:
        Diccionario compatible con el esquema TextAnalysisResult de Pydantic.
    """
    logger.debug("🔤 Iniciando análisis de texto (%d caracteres)…", len(text))

    # ── Capa 1: Análisis de patrones léxicos ───────────────────────────────
    pattern_results = _run_pattern_analysis(text)

    # ── Capa 2: Análisis de sentimiento ────────────────────────────────────
    sentiment_score, sentiment_label = _run_sentiment_analysis(text)

    # ── Cálculo de confianza del análisis de texto ─────────────────────────
    # La confianza se basa en la cantidad y calidad de señales encontradas.
    # Más señales detectadas = mayor certeza en la clasificación.
    pattern_count = len(pattern_results["detected_patterns"])
    manipulation_count = len(pattern_results["manipulation_indicators"])
    credibility_count = len(pattern_results["credibility_signals"])

    # Fórmula de confianza: señales negativas aumentan la certeza de ser fake,
    # señales positivas la disminuyen (el post podría ser legítimo)
    base_confidence = min(0.5 + (pattern_count * 0.15) + (manipulation_count * 0.05), 0.95)

    # Si hay señales de credibilidad, reducimos la confianza en la clasificación
    # negativa (el post tiene elementos que lo hacen potencialmente legítimo)
    confidence = max(base_confidence - (credibility_count * 0.10), 0.05)

    # Si el sentimiento es extremadamente negativo y hay patrones de manipulación,
    # aumentamos la confianza
    if sentiment_score < -0.6 and manipulation_count > 0:
        confidence = min(confidence + 0.10, 0.95)

    logger.debug(
        "🔤 Análisis de texto completado — Patrones: %s | Sentimiento: %.2f | Confianza: %.2f",
        pattern_results["detected_patterns"],
        sentiment_score,
        confidence,
    )

    return {
        "detected_patterns": pattern_results["detected_patterns"],
        "sentiment_score": round(sentiment_score, 4),
        "manipulation_indicators": pattern_results["manipulation_indicators"],
        "credibility_signals": pattern_results["credibility_signals"],
        "text_confidence": round(confidence, 4),
        # Campos adicionales para el corpus de entrenamiento
        "_sentiment_label": sentiment_label,
        "_exclamation_count": pattern_results["exclamation_count"],
    }

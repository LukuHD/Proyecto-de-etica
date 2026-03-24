"""
analyzer.py — Orquestador Principal del Motor de Inteligencia Artificial Multimodal.

Este módulo es el núcleo cognitivo del sistema. Su responsabilidad es:
  1. Recibir los datos limpios de una publicación (texto + imagen).
  2. Orquestar los análisis de texto y visión de forma concurrente.
  3. Ejecutar la FUSIÓN COGNITIVA MULTIMODAL: cruzar los resultados de
     ambos análisis para detectar discrepancias deliberadas entre imagen y texto.
  4. Producir un veredicto final unificado con categoría, confianza y explicación.

La fusión cognitiva multimodal es la funcionalidad más valiosa del sistema:
  Ejemplo crítico: Un texto afirma "incendio en Ciudad de México hoy 2024"
  pero la imagen muestra arquitectura europea y tiene metadatos de 2019.
  Ningún análisis individual detectaría esto de forma aislada — solo la
  fusión revela la intención maliciosa de la desinformación.

Ejecución concurrente:
  Los análisis de texto y visión son independientes entre sí, por lo que
  se ejecutan en PARALELO usando asyncio.gather(). Esto reduce el tiempo
  total de análisis al máximo de (tiempo_texto, tiempo_vision) en lugar
  de la suma de ambos.

Decisión de diseño — Motor de fusión basado en reglas vs. modelo:
  Implementamos la fusión como un motor basado en reglas semánticas porque:
    • No requiere datos de entrenamiento adicionales para la fusión misma.
    • Es explicable: podemos generar explicaciones detalladas de cada discrepancia.
    • Es determinista: facilita la depuración y auditoría del sistema.
    • Un modelo de fusión neuronal requeriría datos etiquetados de pares
      texto-imagen contradictorios, que son costosos de obtener.
  En el futuro, este motor puede ser reemplazado por un modelo de fusión
  entrenado con los datos acumulados en la base de datos JSON.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.ai_engine.text_analyzer import analyze_text
from app.ai_engine.vision_analyzer import analyze_image

logger = logging.getLogger(__name__)


# ── Mapeo de categorías y sus pesos ────────────────────────────────────────
# Cada categoría tiene un peso que determina su prioridad en el veredicto final
# cuando múltiples categorías son detectadas simultáneamente.
_CATEGORY_PRIORITY = {
    "fraude_financiero": 5,       # Máxima prioridad — riesgo económico directo
    "desinformacion_politica": 4, # Alta prioridad — impacto social
    "manipulacion_emocional": 3,  # Media prioridad — manipulación psicológica
    "contenido_enganoso": 2,      # Media prioridad — imagen descontextualizada
    "publicacion_segura": 1,      # Mínima prioridad — sin amenaza detectada
}

# ── Plantillas de explicación ──────────────────────────────────────────────
# Plantillas de lenguaje natural para generar explicaciones comprensibles.
_EXPLANATION_TEMPLATES = {
    "fraude_financiero": (
        "Esta publicación muestra múltiples señales características de fraude financiero: "
        "{patterns}. Los esquemas que prometen ganancias fáciles o retornos garantizados "
        "son tácticas clásicas de estafa."
    ),
    "desinformacion_politica": (
        "Se detectaron patrones de desinformación política en el contenido: {patterns}. "
        "El texto emplea técnicas retóricas diseñadas para generar desconfianza o "
        "impulsar agendas específicas sin evidencia verificable."
    ),
    "manipulacion_emocional": (
        "El contenido usa técnicas de manipulación emocional: {patterns}. "
        "El lenguaje alarmista, las afirmaciones no verificadas y la urgencia artificial "
        "son estrategias diseñadas para evadir el pensamiento crítico del lector."
    ),
    "contenido_enganoso": (
        "Se detectaron discrepancias significativas entre la imagen y el texto: {discrepancies}. "
        "Esta combinación sugiere que el contenido visual está siendo usado fuera de su "
        "contexto original para apoyar afirmaciones falsas o engañosas."
    ),
    "publicacion_segura": (
        "No se detectaron señales significativas de desinformación, fraude o manipulación. "
        "El contenido parece ser una publicación orgánica sin intención maliciosa identificable. "
        "Recuerda verificar siempre información importante en fuentes primarias."
    ),
    "multimodal_discrepancy": (
        "ALERTA MULTIMODAL: {discrepancies}. Esta discrepancia entre la imagen y el texto "
        "es una señal fuerte de que el contenido visual está siendo descontextualizado "
        "deliberadamente para engañar."
    ),
}


def _fuse_categories(
    text_patterns: List[str],
    manipulation_detected: bool,
    multimodal_discrepancies: List[str],
    text_confidence: float,
    vision_confidence: float,
) -> Tuple[str, float]:
    """
    Determina la categoría final y la confianza combinada del análisis multimodal.

    Lógica de fusión:
      1. Si hay discrepancias multimodales (imagen vs texto), el contenido
         es engañoso independientemente de otros análisis.
      2. Si hay patrones de fraude financiero, tienen máxima prioridad.
      3. Si hay manipulación emocional + imagen manipulada, aumentar la certeza.
      4. Si no hay señales negativas, clasificar como publicación segura.

    Args:
        text_patterns:           Patrones detectados en el texto.
        manipulation_detected:   Si la imagen fue detectada como manipulada.
        multimodal_discrepancies: Lista de discrepancias imagen-texto.
        text_confidence:         Confianza del análisis de texto [0-1].
        vision_confidence:       Confianza del análisis visual [0-1].

    Returns:
        Tupla (category_string, combined_confidence).
    """
    # ── Caso 1: Discrepancias multimodales (señal más fuerte) ───────────────
    if multimodal_discrepancies:
        # Las discrepancias imagen-texto son la señal más confiable de desinformación
        confidence = min(0.70 + (len(multimodal_discrepancies) * 0.08), 0.95)
        return "contenido_enganoso", confidence

    # ── Caso 2: Determinar categoría dominante del texto ────────────────────
    if not text_patterns:
        # Sin patrones detectados → publicación segura
        # La confianza refleja qué tan seguro estamos de que es segura
        confidence = max(0.60 - text_confidence * 0.3, 0.30)
        return "publicacion_segura", confidence

    # Seleccionar la categoría de mayor prioridad
    best_category = max(
        text_patterns,
        key=lambda p: _CATEGORY_PRIORITY.get(p, 0),
    )

    # ── Calcular confianza combinada ────────────────────────────────────────
    # Ponderación: texto tiene 60% de peso, visión 40% (cuando disponible)
    if vision_confidence > 0:
        combined = (text_confidence * 0.6) + (vision_confidence * 0.4)
    else:
        combined = text_confidence

    # Bonus de confianza si la imagen también muestra manipulación
    if manipulation_detected and best_category != "publicacion_segura":
        combined = min(combined + 0.12, 0.95)

    # La confianza mínima es 0.35 para cualquier clasificación con evidencia
    combined = max(combined, 0.35)

    return best_category, round(combined, 4)


def _detect_multimodal_discrepancies(
    post_text: str,
    text_result: Dict[str, Any],
    vision_result: Dict[str, Any],
) -> List[str]:
    """
    Motor de fusión cognitiva: detecta discrepancias entre el texto y la imagen.

    Esta es la función más importante del sistema. Compara las afirmaciones
    del texto con el contenido visual detectado para identificar manipulación
    de contexto — el método de desinformación más sofisticado y común.

    Casos detectados:
      1. Discrepancia geográfica: El texto menciona un lugar A pero la imagen
         parece mostrar un lugar B completamente diferente.
      2. Discrepancia temporal: El texto afirma que el evento es "actual" pero
         la imagen tiene señales de ser antigua o de otro período.
      3. Discrepancia de gravedad: El texto describe una catástrofe pero la
         imagen muestra una escena cotidiana o viceversa.
      4. Imagen manipulada + texto de denuncia: Sinergia que confirma intención engañosa.

    Args:
        post_text:    Texto original de la publicación.
        text_result:  Resultado del análisis de texto.
        vision_result: Resultado del análisis de imagen.

    Returns:
        Lista de strings describiendo cada discrepancia detectada.
    """
    discrepancies: List[str] = []
    text_lower = post_text.lower()

    # ── Verificar si hay análisis visual disponible ─────────────────────────
    image_description = vision_result.get("image_description", "").lower()
    geographic_context = (vision_result.get("geographic_context") or "").lower()
    temporal_context = (vision_result.get("temporal_context") or "").lower()
    manipulation_detected = vision_result.get("manipulation_detected", False)

    # Sin descripción visual útil, no podemos hacer la fusión
    if not image_description or "no se proporcionó imagen" in image_description:
        return discrepancies

    # ── Discrepancia geográfica ─────────────────────────────────────────────
    # Países/regiones de América Latina mencionados en el texto
    latam_countries = {
        "méxico": ["mexico", "aztec", "latin america", "central america"],
        "colombia": ["colombia", "latin america", "south america"],
        "venezuela": ["venezuela", "latin america", "south america"],
        "argentina": ["argentina", "latin america", "south america"],
        "perú": ["peru", "latin america", "south america"],
        "chile": ["chile", "latin america", "south america"],
        "brasil": ["brazil", "latin america", "south america"],
        "españa": ["spain", "europe", "iberian"],
    }

    # Contextos geográficos que contradirían las menciones anteriores
    conflicting_regions = {
        "europe": ["europa", "europeo"],
        "asia": ["asia", "asiático", "china", "india"],
        "middle east": ["medio oriente", "árabe"],
        "north america": ["estados unidos", "usa", "eeuu"],
        "africa": ["áfrica", "africano"],
    }

    for country_es, country_en_variants in latam_countries.items():
        # Si el texto menciona un país latinoamericano
        if country_es in text_lower:
            # Y la imagen muestra un contexto geográfico muy diferente
            for region, region_es in conflicting_regions.items():
                if region in geographic_context and not any(v in geographic_context for v in country_en_variants):
                    discrepancies.append(
                        f"Discrepancia geográfica: el texto menciona '{country_es}' "
                        f"pero la imagen parece mostrar un contexto de '{region}'. "
                        f"Posible reutilización de imagen fuera de contexto."
                    )
                    break

    # ── Discrepancia temporal ───────────────────────────────────────────────
    # Si el texto usa palabras de actualidad pero la imagen parece antigua
    recency_words = ["hoy", "ahora", "esta mañana", "esta tarde", "en este momento",
                     "recientemente", "esta semana", "este mes", "2024", "2025"]
    text_claims_recent = any(word in text_lower for word in recency_words)

    historical_signals = ["old", "vintage", "historical", "1990", "1980", "1970",
                           "2010", "2011", "2012", "2013", "2014", "2015", "2016",
                           "decades ago", "years ago", "archival"]
    image_seems_old = any(signal in temporal_context for signal in historical_signals)

    if text_claims_recent and image_seems_old:
        discrepancies.append(
            "Discrepancia temporal crítica: el texto afirma que el evento es reciente "
            f"(palabras clave: {[w for w in recency_words if w in text_lower][:3]}), "
            f"pero la imagen muestra indicadores de ser de un período anterior. "
            "Esta es una táctica clásica de desinformación: reciclar imágenes antiguas "
            "para ilustrar eventos supuestamente actuales."
        )

    # ── Discrepancia de gravedad (texto catastrófico + imagen cotidiana) ────
    disaster_words = ["catástrofe", "desastre", "terremoto", "inundación", "incendio",
                      "explosion", "ataque", "bombardeo", "masacre", "colapso",
                      "emergencia", "víctimas", "muertos", "heridos"]
    text_claims_disaster = any(word in text_lower for word in disaster_words)

    peaceful_signals = ["calm", "peaceful", "normal", "everyday", "ordinary",
                        "street", "shopping", "park", "family", "smiling"]
    image_seems_peaceful = any(signal in image_description for signal in peaceful_signals)

    if text_claims_disaster and image_seems_peaceful:
        discrepancies.append(
            "Discrepancia de contenido: el texto describe una situación de emergencia o desastre, "
            "pero la imagen parece mostrar una escena cotidiana sin signos de catástrofe. "
            "Posible uso de imagen no relacionada para generar impacto emocional falso."
        )

    # ── Señal combinada: imagen manipulada + texto de denuncia ─────────────
    if manipulation_detected and ("fraude_financiero" in text_result.get("detected_patterns", [])
                                  or "desinformacion_politica" in text_result.get("detected_patterns", [])):
        discrepancies.append(
            "Sinergia negativa: se detectó manipulación digital en la imagen "
            "JUNTO CON patrones de fraude o desinformación en el texto. "
            "Esta combinación sugiere una campaña de desinformación elaborada y deliberada."
        )

    return discrepancies


def _generate_explanation(
    category: str,
    text_result: Dict[str, Any],
    vision_result: Dict[str, Any],
    discrepancies: List[str],
) -> str:
    """
    Genera una explicación en lenguaje natural del veredicto final.

    La explicación debe ser:
      • Comprensible por cualquier usuario sin conocimientos técnicos.
      • Específica (mencionar exactamente qué se detectó, no generalidades).
      • Accionable (dar al usuario información para tomar decisiones).
      • Concisa (máximo 2-3 oraciones).

    Args:
        category:     Categoría asignada al contenido.
        text_result:  Resultado del análisis de texto.
        vision_result: Resultado del análisis visual.
        discrepancies: Lista de discrepancias detectadas.

    Returns:
        String de explicación en lenguaje natural.
    """
    # ── Caso con discrepancias multimodales (máxima prioridad explicativa) ──
    if discrepancies:
        primary_discrepancy = discrepancies[0][:200]  # Limitar longitud
        additional = f" Se detectaron {len(discrepancies)} discrepancias en total." if len(discrepancies) > 1 else ""
        return (
            f"⚠️ ALERTA DE DESINFORMACIÓN: {primary_discrepancy}{additional} "
            "Verifica esta información en fuentes independientes antes de compartirla."
        )

    # ── Explicación basada en la categoría ─────────────────────────────────
    patterns = text_result.get("detected_patterns", [])
    manipulation_indicators = text_result.get("manipulation_indicators", [])

    if category == "fraude_financiero":
        pattern_detail = ", ".join(manipulation_indicators[:2]) if manipulation_indicators else "promesas financieras irreales"
        return _EXPLANATION_TEMPLATES["fraude_financiero"].format(patterns=pattern_detail)

    elif category == "desinformacion_politica":
        pattern_detail = ", ".join(patterns[:2]) if patterns else "narrativas sin verificar"
        return _EXPLANATION_TEMPLATES["desinformacion_politica"].format(patterns=pattern_detail)

    elif category == "manipulacion_emocional":
        indicator_detail = ", ".join(manipulation_indicators[:2]) if manipulation_indicators else "lenguaje alarmista"
        return _EXPLANATION_TEMPLATES["manipulacion_emocional"].format(patterns=indicator_detail)

    elif category == "contenido_enganoso":
        disc_detail = discrepancies[0][:100] if discrepancies else "inconsistencias entre imagen y texto"
        return _EXPLANATION_TEMPLATES["contenido_enganoso"].format(discrepancies=disc_detail)

    else:  # publicacion_segura
        return _EXPLANATION_TEMPLATES["publicacion_segura"]


async def run_full_analysis(
    post_text: str,
    author_name: str,
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Función principal del motor de IA: orquesta el análisis multimodal completo.

    Flujo de ejecución:
      ┌─────────────────────────────────────────────────────────┐
      │  1. Análisis de texto (NLP)    ─┐                       │
      │                                  ├→ asyncio.gather()    │
      │  2. Análisis de imagen (VLM)   ─┘  (paralelo)          │
      │                                                          │
      │  3. Fusión cognitiva multimodal                         │
      │     └→ Detección de discrepancias imagen-texto          │
      │                                                          │
      │  4. Determinación de categoría y confianza final        │
      │                                                          │
      │  5. Generación de explicación en lenguaje natural       │
      └─────────────────────────────────────────────────────────┘

    Args:
        post_text:    Texto completo de la publicación.
        author_name:  Nombre del autor.
        image_base64: Imagen en Base64 (opcional).
        image_url:    URL de la imagen (alternativa).

    Returns:
        Diccionario completo compatible con el esquema AnalysisResponse de Pydantic.
    """
    logger.info(
        "🧠 Iniciando análisis multimodal para publicación de '%s' (%d chars de texto)…",
        author_name,
        len(post_text),
    )

    # ── Paso 1 & 2: Análisis PARALELO de texto e imagen ────────────────────
    # asyncio.gather() ejecuta ambas coroutines concurrentemente.
    # El tiempo total es max(tiempo_texto, tiempo_imagen), no su suma.
    text_result, vision_result = await asyncio.gather(
        analyze_text(post_text),
        analyze_image(image_base64, image_url),
    )

    logger.debug("✅ Análisis paralelo completado.")
    logger.debug("   Texto — Patrones: %s | Confianza: %.2f",
                 text_result.get("detected_patterns"), text_result.get("text_confidence"))
    logger.debug("   Visión — Manipulación: %s | Confianza: %.2f",
                 vision_result.get("manipulation_detected"), vision_result.get("vision_confidence"))

    # ── Paso 3: Fusión cognitiva multimodal ────────────────────────────────
    multimodal_discrepancies = _detect_multimodal_discrepancies(
        post_text, text_result, vision_result
    )

    if multimodal_discrepancies:
        logger.info(
            "⚡ Fusión multimodal detectó %d discrepancia(s) imagen-texto.",
            len(multimodal_discrepancies),
        )

    # ── Paso 4: Determinar categoría y confianza final ─────────────────────
    final_category, final_confidence = _fuse_categories(
        text_patterns=text_result.get("detected_patterns", []),
        manipulation_detected=vision_result.get("manipulation_detected", False),
        multimodal_discrepancies=multimodal_discrepancies,
        text_confidence=text_result.get("text_confidence", 0.0),
        vision_confidence=vision_result.get("vision_confidence", 0.0),
    )

    # ── Paso 5: Generar explicación en lenguaje natural ─────────────────────
    explanation = _generate_explanation(
        category=final_category,
        text_result=text_result,
        vision_result=vision_result,
        discrepancies=multimodal_discrepancies,
    )

    logger.info(
        "🏁 Análisis completado — Categoría: %s | Confianza: %.1f%% | Discrepancias: %d",
        final_category,
        final_confidence * 100,
        len(multimodal_discrepancies),
    )

    # ── Construir respuesta final ───────────────────────────────────────────
    return {
        "category": final_category,
        "confidence": final_confidence,
        "explanation": explanation,
        "multimodal_discrepancies": multimodal_discrepancies,

        # Resultados parciales (para desglose detallado en la respuesta)
        "text_analysis": {
            "detected_patterns": text_result.get("detected_patterns", []),
            "sentiment_score": text_result.get("sentiment_score", 0.0),
            "manipulation_indicators": text_result.get("manipulation_indicators", []),
            "credibility_signals": text_result.get("credibility_signals", []),
            "text_confidence": text_result.get("text_confidence", 0.0),
        },
        "vision_analysis": {
            "image_description": vision_result.get("image_description", "No analizada."),
            "detected_objects": vision_result.get("detected_objects", []),
            "manipulation_detected": vision_result.get("manipulation_detected", False),
            "geographic_context": vision_result.get("geographic_context"),
            "temporal_context": vision_result.get("temporal_context"),
            "vision_confidence": vision_result.get("vision_confidence", 0.0),
        },

        # Metadatos de la sesión de análisis
        "analyzed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

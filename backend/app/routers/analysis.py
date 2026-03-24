"""
analysis.py — Router de análisis de publicaciones.

Este módulo contiene los controladores (handlers) de todas las rutas HTTP
relacionadas con el análisis de contenido. Es el punto de integración entre:
  • La capa de transporte HTTP (FastAPI + Pydantic)
  • El sistema de caché (db_manager)
  • El motor de IA (analyzer)

Endpoints expuestos:
  POST /api/v1/analyze      → Análisis de una publicación (principal)
  GET  /api/v1/stats        → Estadísticas de la base de datos
  GET  /api/v1/history      → Historial de análisis (paginado)

Flujo del endpoint principal POST /analyze:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  1. FastAPI valida el payload con Pydantic (PostAnalysisRequest)     │
  │  2. Calcular hash SHA-256 del post (texto + imagen)                  │
  │  3. Consultar caché JSON → Si existe: devolver resultado inmediato   │
  │  4. Si es nuevo: invocar motor IA (run_full_analysis)                │
  │  5. Guardar resultado en base de datos JSON                          │
  │  6. Construir y devolver AnalysisResponse enriquecido                │
  └──────────────────────────────────────────────────────────────────────┘

Decisión arquitectónica — Separación de responsabilidades:
  Este router NO contiene lógica de negocio. Solo coordina el flujo de datos
  entre los módulos especializados. Si la lógica de análisis cambia, este
  archivo no necesita modificarse.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.ai_engine.analyzer import run_full_analysis
from app.schemas.post_schema import AnalysisCategory, AnalysisResponse, PostAnalysisRequest, TextAnalysisResult, VisionAnalysisResult
from app.utils.db_manager import check_cache, get_all_records, get_statistics, save_analysis_result
from app.utils.hasher import compute_post_hash

logger = logging.getLogger(__name__)

# ── Instancia del router ────────────────────────────────────────────────────
# El router se monta en app.main con el prefijo /api/v1
router = APIRouter()


# ── Endpoint principal: POST /analyze ──────────────────────────────────────
@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analizar una publicación de red social",
    description=(
        "Recibe el contenido de una publicación (texto + imagen opcional) "
        "y retorna un análisis detallado indicando si contiene desinformación, "
        "fraude u otro contenido malicioso. Usa caché para posts ya analizados."
    ),
)
async def analyze_post(request: PostAnalysisRequest) -> AnalysisResponse:
    """
    Handler principal del sistema de detección de desinformación.

    Este endpoint es el único punto de entrada para la extensión del navegador.
    FastAPI valida automáticamente el cuerpo de la petición contra el esquema
    PostAnalysisRequest antes de llamar a esta función, rechazando con HTTP 422
    cualquier payload malformado.

    Args:
        request: Cuerpo de la petición validado por Pydantic.

    Returns:
        AnalysisResponse con el veredicto completo del análisis.

    Raises:
        HTTPException 500: Si ocurre un error interno en el motor de IA.
    """
    logger.info(
        "📨 Petición de análisis recibida — Autor: '%s' | Texto: %d chars | "
        "Imagen: %s",
        request.author_name,
        len(request.post_text),
        "Base64" if request.image_base64 else ("URL" if request.image_url else "No"),
    )

    # ── Paso 1: Calcular la huella digital SHA-256 ──────────────────────────
    # La imagen puede estar en Base64 o como URL — usamos la que esté disponible
    image_data = request.image_base64 or request.image_url
    post_hash = compute_post_hash(
        post_text=request.post_text,
        image_data=image_data,
        author_name=request.author_name,
    )

    # ── Paso 2: Consultar la caché JSON ─────────────────────────────────────
    # La consulta de caché es O(1) porque el JSON está indexado por hash
    cached_record: Optional[Dict[str, Any]] = await check_cache(post_hash)

    if cached_record:
        # ── Camino rápido: post ya analizado ───────────────────────────────
        logger.info("💾 Devolviendo resultado desde caché para hash: %s…", post_hash[:16])

        # Reconstruir el objeto de respuesta desde los datos del caché
        return _build_response_from_cache(post_hash, cached_record)

    # ── Paso 3: Análisis completo con el motor de IA ────────────────────────
    # Solo llegamos aquí si el post es genuinamente nuevo
    logger.info("🧠 Post nuevo — Iniciando análisis IA para hash: %s…", post_hash[:16])

    try:
        analysis_result = await run_full_analysis(
            post_text=request.post_text,
            author_name=request.author_name,
            image_base64=request.image_base64,
            image_url=request.image_url,
        )
    except Exception as e:
        logger.error("❌ Error crítico en el motor de IA: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Error interno en el motor de análisis: {str(e)}. "
                "Por favor verifica los logs del servidor para más detalles."
            ),
        )

    # ── Paso 4: Persistir el resultado en la base de datos JSON ─────────────
    # Esta operación es asíncrona y protegida por el Lock de escritura
    try:
        await save_analysis_result(
            post_hash=post_hash,
            post_text=request.post_text,
            author_name=request.author_name,
            post_timestamp=request.post_timestamp.isoformat(),
            analysis_result=analysis_result,
        )
    except Exception as e:
        # No falla la respuesta al cliente si la escritura en DB falla,
        # pero sí lo registramos como error crítico para monitoreo.
        logger.error(
            "❌ Error al persistir resultado en DB (hash: %s…): %s",
            post_hash[:16], e,
        )

    # ── Paso 5: Construir y devolver la respuesta enriquecida ───────────────
    return _build_response_from_analysis(post_hash, analysis_result, cached=False)


# ── Endpoint de estadísticas: GET /stats ───────────────────────────────────
@router.get(
    "/stats",
    summary="Estadísticas de la base de datos de análisis",
    description="Retorna estadísticas resumidas de todos los análisis realizados.",
)
async def get_analysis_stats() -> Dict[str, Any]:
    """
    Retorna estadísticas de uso del sistema de detección.

    Útil para el panel de control de la extensión del navegador,
    mostrando cuántos posts han sido analizados y su distribución por categoría.
    """
    try:
        stats = await get_statistics()
        return stats
    except Exception as e:
        logger.error("Error obteniendo estadísticas: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas: {str(e)}",
        )


# ── Endpoint de historial: GET /history ────────────────────────────────────
@router.get(
    "/history",
    summary="Historial de análisis realizados",
    description=(
        "Retorna el historial de publicaciones analizadas, "
        "paginado para evitar respuestas demasiado grandes."
    ),
)
async def get_analysis_history(
    page: int = Query(default=1, ge=1, description="Número de página (empieza en 1)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Registros por página (máx. 100)"),
    category_filter: Optional[str] = Query(
        default=None,
        description="Filtrar por categoría (fraude_financiero, desinformacion_politica, etc.)",
    ),
) -> Dict[str, Any]:
    """
    Retorna el historial paginado de análisis, opcionalmente filtrado por categoría.

    La paginación es esencial aquí porque después de semanas de uso, la base
    de datos puede contener miles de registros y devolver todos sería ineficiente.
    """
    try:
        db_data = await get_all_records()
        records = db_data.get("records", {})

        # Convertir dict a lista para paginación
        records_list = list(records.values())

        # Aplicar filtro de categoría si se especifica
        if category_filter:
            records_list = [
                r for r in records_list
                if r.get("category") == category_filter
            ]

        # Ordenar por fecha de análisis (más reciente primero)
        records_list.sort(key=lambda r: r.get("analyzed_at", ""), reverse=True)

        # Calcular paginación
        total = len(records_list)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_records = records_list[start_idx:end_idx]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "records": page_records,
        }

    except Exception as e:
        logger.error("Error obteniendo historial: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener historial: {str(e)}",
        )


# ── Funciones auxiliares de construcción de respuestas ─────────────────────
def _build_response_from_analysis(
    post_hash: str,
    analysis_result: Dict[str, Any],
    cached: bool,
) -> AnalysisResponse:
    """
    Construye el objeto AnalysisResponse a partir del resultado del motor IA.

    Centralizar la construcción de la respuesta en una función auxiliar
    garantiza consistencia entre el resultado directo del análisis y el
    resultado reconstruido desde la caché.

    Args:
        post_hash:       Hash SHA-256 del post.
        analysis_result: Diccionario retornado por run_full_analysis().
        cached:          True si el resultado viene de la caché.

    Returns:
        Instancia de AnalysisResponse validada por Pydantic.
    """
    text_data = analysis_result.get("text_analysis", {})
    vision_data = analysis_result.get("vision_analysis", {})

    # Construir el objeto TextAnalysisResult
    text_analysis = TextAnalysisResult(
        detected_patterns=text_data.get("detected_patterns", []),
        sentiment_score=text_data.get("sentiment_score", 0.0),
        manipulation_indicators=text_data.get("manipulation_indicators", []),
        credibility_signals=text_data.get("credibility_signals", []),
        text_confidence=text_data.get("text_confidence", 0.0),
    )

    # Construir el objeto VisionAnalysisResult
    vision_analysis = VisionAnalysisResult(
        image_description=vision_data.get("image_description", "No analizada."),
        detected_objects=vision_data.get("detected_objects", []),
        manipulation_detected=vision_data.get("manipulation_detected", False),
        geographic_context=vision_data.get("geographic_context"),
        temporal_context=vision_data.get("temporal_context"),
        vision_confidence=vision_data.get("vision_confidence", 0.0),
    )

    # Construir la respuesta final
    return AnalysisResponse(
        post_hash=post_hash,
        category=AnalysisCategory(analysis_result.get("category", "publicacion_segura")),
        confidence=analysis_result.get("confidence", 0.5),
        explanation=analysis_result.get("explanation", "Análisis completado."),
        text_analysis=text_analysis,
        vision_analysis=vision_analysis,
        multimodal_discrepancies=analysis_result.get("multimodal_discrepancies", []),
        cached=cached,
        analyzed_at=datetime.now(timezone.utc),
    )


def _build_response_from_cache(
    post_hash: str,
    cached_record: Dict[str, Any],
) -> AnalysisResponse:
    """
    Reconstruye un AnalysisResponse a partir de un registro del caché JSON.

    El registro del caché almacena los campos clave del análisis, pero no
    el objeto completo. Esta función los reconstituye con los valores
    disponibles y defaults razonables para los campos no almacenados.

    Args:
        post_hash:     Hash SHA-256 del post.
        cached_record: Registro del JSON de base de datos.

    Returns:
        Instancia de AnalysisResponse marcada como cached=True.
    """
    # Reconstruir TextAnalysisResult desde los campos almacenados en DB
    text_analysis = TextAnalysisResult(
        detected_patterns=cached_record.get("text_patterns", []),
        sentiment_score=cached_record.get("sentiment_score", 0.0),
        manipulation_indicators=cached_record.get("manipulation_indicators", []),
        credibility_signals=[],
        text_confidence=cached_record.get("confidence", 0.5),
    )

    # Reconstruir VisionAnalysisResult desde los campos almacenados en DB
    vision_analysis = VisionAnalysisResult(
        image_description=cached_record.get("image_description", "Recuperado de caché."),
        detected_objects=[],
        manipulation_detected=False,
        geographic_context=None,
        temporal_context=None,
        vision_confidence=0.0,
    )

    # Parsear la fecha de análisis del registro (puede ser string ISO o None)
    analyzed_at_str = cached_record.get("analyzed_at")
    try:
        analyzed_at = datetime.fromisoformat(
            analyzed_at_str.replace("Z", "+00:00")
        ) if analyzed_at_str else datetime.now(timezone.utc)
    except (ValueError, AttributeError):
        analyzed_at = datetime.now(timezone.utc)

    return AnalysisResponse(
        post_hash=post_hash,
        category=AnalysisCategory(cached_record.get("category", "publicacion_segura")),
        confidence=cached_record.get("confidence", 0.5),
        explanation=cached_record.get("explanation", "Resultado recuperado de caché."),
        text_analysis=text_analysis,
        vision_analysis=vision_analysis,
        multimodal_discrepancies=cached_record.get("multimodal_discrepancies", []),
        cached=True,  # Marcar como resultado de caché
        analyzed_at=analyzed_at,
    )

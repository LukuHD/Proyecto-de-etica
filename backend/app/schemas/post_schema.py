"""
post_schema.py — Esquemas de validación de datos con Pydantic v2.

Este módulo centraliza la definición de todos los modelos de datos que
atraviesan la frontera HTTP de la aplicación:

  • PostAnalysisRequest  → Carga útil (payload) enviada por la extensión
  • AnalysisCategory     → Enumeración de categorías posibles de clasificación
  • TextAnalysisResult   → Resultado parcial del análisis de texto (NLP)
  • VisionAnalysisResult → Resultado parcial del análisis de imagen (visión)
  • AnalysisResponse     → Respuesta enriquecida enviada de vuelta a la extensión

Decisión arquitectónica: Separar los esquemas en su propio módulo garantiza
que la lógica de validación sea reutilizable desde los tests, otros routers
y scripts de mantenimiento, sin acoplar el modelo de datos a ninguna capa
específica de la aplicación.

Pydantic v2 realiza la validación en el momento de instanciar el modelo,
lanzando `ValidationError` (que FastAPI captura automáticamente y convierte
en una respuesta HTTP 422 Unprocessable Entity) si algún campo no cumple
las restricciones declaradas.
"""

from __future__ import annotations  # Permite type hints con referencias hacia adelante

from datetime import datetime        # Para el campo de marca de tiempo (timestamp)
from enum import Enum                # Para definir enumeraciones de categorías
from typing import Optional          # Para campos opcionales

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator  # Pydantic v2


# ── Enumeración de categorías de clasificación ──────────────────────────────
class AnalysisCategory(str, Enum):
    """
    Enumeración estricta de las posibles categorías de clasificación.

    Usar un Enum en lugar de strings libres garantiza:
      1. Consistencia en todos los registros del JSON.
      2. Autocompletado en el cliente de la extensión.
      3. Validación automática por Pydantic (rechaza valores no listados).

    Las categorías están alineadas con las amenazas más comunes
    identificadas en el análisis de desinformación en Facebook:
    """
    FRAUDE_FINANCIERO = "fraude_financiero"
    """Promesas de ganancias irreales, esquemas Ponzi, phishing económico."""

    DESINFORMACION_POLITICA = "desinformacion_politica"
    """Contenido político manipulado, deep fakes electorales, propaganda."""

    CONTENIDO_ENGANOSO = "contenido_enganoso"
    """Imágenes o videos fuera de contexto, clickbait agresivo, pseudociencia."""

    MANIPULACION_EMOCIONAL = "manipulacion_emocional"
    """Lenguaje diseñado para provocar miedo, ira o urgencia artificial."""

    PUBLICACION_SEGURA = "publicacion_segura"
    """No se detectaron señales significativas de desinformación o fraude."""


# ── Esquema de la petición entrante ────────────────────────────────────────
class PostAnalysisRequest(BaseModel):
    """
    Modelo de la carga útil (payload) que la extensión del navegador envía
    al endpoint POST /api/v1/analyze.

    Campos obligatorios:
      • post_text     — Contenido textual completo de la publicación.
      • author_name   — Nombre del autor tal como aparece en la red social.
      • post_timestamp — Marca de tiempo ISO 8601 de la publicación original.

    Campos opcionales (al menos uno debe estar presente para el análisis visual):
      • image_base64  — Imagen adjunta codificada en Base64 (sin prefijo data:URI).
      • image_url     — URL directa de la imagen si la extensión no puede codificarla.

    Nota: La extensión debe enviar `image_base64` O `image_url`, no necesariamente
    ambos. Si ninguno se proporciona, el análisis se realiza solo sobre el texto.
    """

    post_text: str = Field(
        ...,                          # Campo requerido (... = obligatorio en Pydantic)
        min_length=1,                 # No aceptar strings vacíos
        max_length=10_000,            # Límite razonable para evitar ataques DoS
        description="Texto completo de la publicación extraído por la extensión.",
        examples=["¡GANA $5000 DIARIOS TRABAJANDO DESDE CASA! Haz clic aquí 👇"],
    )

    author_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Nombre del autor de la publicación en la red social.",
        examples=["Juan Pérez"],
    )

    post_timestamp: datetime = Field(
        ...,
        description=(
            "Marca de tiempo de creación de la publicación en formato ISO 8601. "
            "La extensión debe convertir la fecha del DOM al formato estándar."
        ),
        examples=["2024-06-15T14:30:00Z"],
    )

    image_base64: Optional[str] = Field(
        default=None,
        description=(
            "Imagen adjunta codificada en Base64 puro (sin el prefijo data:image/...;base64,). "
            "Mutuamente excluyente con image_url; si ambos se proporcionan, "
            "image_base64 tiene prioridad."
        ),
    )

    image_url: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="URL directa de la imagen adjunta, usada si no se puede codificar en Base64.",
        examples=["https://example.com/image.jpg"],
    )

    @field_validator("post_text")
    @classmethod
    def text_must_not_be_whitespace_only(cls, v: str) -> str:
        """
        Validador personalizado: rechaza textos que sean solo espacios en blanco.
        Un post de 1000 espacios pasaría min_length=1 pero no tendría contenido útil.
        """
        if v.strip() == "":
            raise ValueError("El texto de la publicación no puede contener solo espacios en blanco.")
        return v.strip()  # También eliminamos espacios sobrantes en los extremos


# ── Esquemas de resultados parciales del motor IA ──────────────────────────
class TextAnalysisResult(BaseModel):
    """
    Resultado del submódulo de análisis de lenguaje natural (NLP).
    Encapsula todas las señales detectadas en el texto de la publicación.
    """

    detected_patterns: list[str] = Field(
        default_factory=list,
        description="Lista de patrones semánticos sospechosos detectados en el texto.",
        examples=[["lenguaje_sensacionalista", "promesa_financiera_irreal"]],
    )

    sentiment_score: float = Field(
        ...,
        ge=-1.0,    # Mayor o igual a -1.0 (muy negativo)
        le=1.0,     # Menor o igual a 1.0 (muy positivo)
        description="Puntuación de sentimiento: -1.0 (muy negativo) a 1.0 (muy positivo).",
    )

    manipulation_indicators: list[str] = Field(
        default_factory=list,
        description="Indicadores específicos de manipulación emocional detectados.",
    )

    credibility_signals: list[str] = Field(
        default_factory=list,
        description="Señales que podrían indicar contenido creíble (fuentes citadas, etc.).",
    )

    text_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confianza del análisis de texto como proporción [0.0 - 1.0].",
    )


class VisionAnalysisResult(BaseModel):
    """
    Resultado del submódulo de análisis de visión por computadora.
    Encapsula toda la información extraída de la imagen adjunta.
    """

    image_description: str = Field(
        default="No se proporcionó imagen.",
        description="Descripción en lenguaje natural del contenido visual de la imagen.",
    )

    detected_objects: list[str] = Field(
        default_factory=list,
        description="Objetos y entidades visualmente identificados en la imagen.",
    )

    manipulation_detected: bool = Field(
        default=False,
        description="Indica si se detectaron signos de manipulación digital (edición, deepfake).",
    )

    geographic_context: Optional[str] = Field(
        default=None,
        description="Contexto geográfico inferido de la imagen, si es detectable.",
    )

    temporal_context: Optional[str] = Field(
        default=None,
        description="Indicios sobre la época o fecha de la imagen (metadatos, elementos visuales).",
    )

    vision_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confianza del análisis visual como proporción [0.0 - 1.0].",
    )


# ── Esquema de la respuesta completa ───────────────────────────────────────
class AnalysisResponse(BaseModel):
    """
    Respuesta enriquecida que el backend devuelve a la extensión del navegador
    tras completar el análisis multimodal completo.

    Este objeto contiene TODA la información necesaria para que la extensión
    construya la alerta visual en el DOM de Facebook sin necesidad de realizar
    ningún procesamiento adicional en el lado del cliente.
    """

    post_hash: str = Field(
        ...,
        description="Huella digital SHA-256 única de la publicación analizada.",
    )

    category: AnalysisCategory = Field(
        ...,
        description="Categoría de clasificación asignada por el motor de IA.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Nivel de confianza global del veredicto como proporción [0.0 - 1.0]. "
            "Ejemplo: 0.87 = 87% de certeza en la clasificación asignada."
        ),
    )

    explanation: str = Field(
        ...,
        description=(
            "Justificación en lenguaje natural del veredicto, comprensible para "
            "el usuario final sin conocimientos técnicos. "
            "Ejemplo: 'La imagen parece estar descontextualizada geográficamente; "
            "fue tomada en Europa del Este, pero el texto afirma que muestra eventos recientes en México.'"
        ),
    )

    text_analysis: TextAnalysisResult = Field(
        ...,
        description="Desglose detallado del análisis de texto (NLP).",
    )

    vision_analysis: VisionAnalysisResult = Field(
        ...,
        description="Desglose detallado del análisis de imagen (visión por computadora).",
    )

    multimodal_discrepancies: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de discrepancias detectadas entre el contenido del texto y la imagen. "
            "Este campo es el corazón de la fusión cognitiva multimodal."
        ),
        examples=[[
            "El texto menciona un evento en 2024, pero la imagen muestra metadatos de 2019.",
            "La imagen muestra una zona tropical, pero el texto describe una ciudad ártica.",
        ]],
    )

    cached: bool = Field(
        default=False,
        description="True si el resultado fue recuperado de la caché JSON local (sin re-análisis).",
    )

    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Marca de tiempo UTC del momento en que se realizó (o recuperó) el análisis.",
    )

    model_config = ConfigDict()

    @field_serializer("analyzed_at")
    def serialize_analyzed_at(self, v: datetime) -> str:
        """Serializa el campo analyzed_at como string ISO 8601 con sufijo Z (UTC)."""
        return v.isoformat().replace("+00:00", "Z")

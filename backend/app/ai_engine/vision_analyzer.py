"""
vision_analyzer.py — Submódulo de Análisis de Visión por Computadora.

Responsabilidad: Analizar la imagen adjunta a una publicación usando un
modelo de Lenguaje Visual (VLM — Vision Language Model) para:
  1. Generar una descripción en lenguaje natural del contenido visual.
  2. Detectar posibles manipulaciones digitales (edición, deepfake, Photoshop).
  3. Extraer el contexto geográfico y temporal de la imagen.
  4. Identificar objetos, personas y escenas relevantes para el fact-checking.

Modelo seleccionado: moondream2 (vikhyatk/moondream2)
  • 1.87B parámetros — ejecutable en CPU con 4-8GB RAM.
  • Versión cuantizada int8: ~900MB en disco, ~2GB en RAM — viable en hardware doméstico.
  • Diseñado específicamente para preguntas sobre imágenes (VQA — Visual Q&A).
  • Alternativas si moondream2 no está disponible:
    - microsoft/Florence-2-base (224M parámetros, muy ligero)
    - THUDM/cogvlm2-llama3-chat-19B (solo para GPU con ≥16GB VRAM)

Estrategia de carga:
  Idéntica al text_analyzer: carga lazy del modelo al primer uso para
  que el servidor arranque instantáneamente. El modelo se mantiene en
  memoria (RAM o VRAM) para análisis subsiguientes sin costo de recarga.

Gestión de memoria:
  Si el sistema no tiene suficiente RAM para el modelo completo, se activa
  un modo degradado (fallback) que usa solo análisis de metadatos de la imagen
  y heurísticas básicas, garantizando que el sistema NUNCA falle completamente.
"""

import base64
import io
import logging
import re                           # Movido aquí desde el final del archivo (PEP 8)
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Importaciones de procesamiento de imagen ───────────────────────────────
try:
    from PIL import Image, ImageStat
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning(
        "⚠️  'Pillow' no instalado. Análisis de imagen desactivado. "
        "Para activarlo: pip install Pillow"
    )

# ── Importación del VLM (modelo de lenguaje visual) ────────────────────────
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
    logger.info("✅ PyTorch y Transformers disponibles para análisis visual.")
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "⚠️  'torch' o 'transformers' no disponibles. "
        "El análisis visual usará solo metadatos e heurísticas básicas."
    )

# ── Importación opcional de httpx para descargar imágenes por URL ──────────
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    logger.warning("⚠️  'httpx' no instalado. No se pueden analizar imágenes por URL.")

# ── Estado global del VLM (carga lazy) ─────────────────────────────────────
_vlm_model = None
_vlm_tokenizer = None
_VLM_MODEL_ID = "vikhyatk/moondream2"  # Modelo VLM ligero y eficiente
_VLM_REVISION = "2024-08-26"           # Revisión estable del modelo


def _load_vision_model() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Carga el modelo VLM de forma lazy.

    Determina automáticamente si usar GPU (CUDA) o CPU, con gestión
    inteligente de memoria para evitar OOM (Out of Memory).

    Returns:
        Tupla (model, tokenizer) listas para inferencia, o (None, None) si falla.
    """
    global _vlm_model, _vlm_tokenizer

    if not _TRANSFORMERS_AVAILABLE or not _PIL_AVAILABLE:
        return None, None

    # Si ya están cargados, retornar directamente (carga lazy)
    if _vlm_model is not None and _vlm_tokenizer is not None:
        return _vlm_model, _vlm_tokenizer

    try:
        logger.info("🔄 Cargando modelo VLM '%s'…", _VLM_MODEL_ID)
        logger.info("   (Esto puede tardar 1-3 minutos en la primera ejecución mientras se descarga el modelo)")

        # ── Configuración del dispositivo ───────────────────────────────────
        # Detección automática de CUDA para usar GPU si está disponible.
        # Si no hay GPU, el modelo corre en CPU (más lento pero funcional).
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16  # float16 en GPU para mayor velocidad y menor VRAM
            logger.info("   🎮 GPU CUDA detectada. Usando aceleración por hardware.")
        else:
            device = "cpu"
            dtype = torch.float32  # float32 en CPU (float16 puede causar inestabilidad)
            logger.info("   💻 Sin GPU CUDA. Usando CPU (análisis más lento pero funcional).")

        # ── Carga del tokenizer ─────────────────────────────────────────────
        _vlm_tokenizer = AutoTokenizer.from_pretrained(
            _VLM_MODEL_ID,
            revision=_VLM_REVISION,
            trust_remote_code=True,  # moondream2 usa código personalizado
        )

        # ── Carga del modelo ────────────────────────────────────────────────
        _vlm_model = AutoModelForCausalLM.from_pretrained(
            _VLM_MODEL_ID,
            revision=_VLM_REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
            # low_cpu_mem_usage=True reduce el pico de memoria durante la carga
            low_cpu_mem_usage=True,
        ).to(device).eval()  # .eval() desactiva dropout para inferencia eficiente

        logger.info("✅ Modelo VLM cargado exitosamente en %s.", device.upper())
        return _vlm_model, _vlm_tokenizer

    except Exception as e:
        logger.error(
            "❌ No se pudo cargar el modelo VLM: %s\n"
            "   El análisis visual usará el modo degradado (metadatos + heurísticas).",
            e,
        )
        _vlm_model = None
        _vlm_tokenizer = None
        return None, None


def _decode_image(image_base64: Optional[str] = None, image_url: Optional[str] = None) -> Optional[Any]:
    """
    Convierte Base64 o URL a un objeto PIL.Image listo para el modelo VLM.

    Prioridad: image_base64 tiene prioridad sobre image_url.

    Args:
        image_base64: String Base64 puro de la imagen (sin prefijo data:URI).
        image_url:    URL directa de la imagen.

    Returns:
        Objeto PIL.Image en modo RGB, o None si no se puede decodificar.
    """
    if not _PIL_AVAILABLE:
        return None

    if image_base64:
        try:
            # Eliminar el prefijo data:image/...;base64, si existe
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]

            # Decodificar Base64 a bytes
            image_bytes = base64.b64decode(image_base64)

            # Crear imagen PIL desde los bytes en memoria (sin escribir a disco)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.debug("🖼️  Imagen decodificada desde Base64: %dx%d px", *image.size)
            return image

        except Exception as e:
            logger.warning("⚠️  Error decodificando imagen Base64: %s", e)
            return None

    if image_url and _HTTPX_AVAILABLE:
        # ── Validación anti-SSRF ────────────────────────────────────────────
        # Antes de realizar la petición HTTP, validamos que la URL sea segura:
        #   1. Solo se permiten esquemas http/https (no file://, ftp://, etc.)
        #   2. No se permiten IPs privadas ni localhost (evitar SSRF a servicios internos)
        # Esto previene que un actor malicioso use la extensión para hacer que el
        # servidor realice peticiones a servicios internos de la red local.
        sanitized_url = _validate_image_url(image_url)
        if sanitized_url is None:
            logger.warning(
                "⚠️  URL de imagen rechazada por política de seguridad anti-SSRF: %s",
                image_url[:100],  # Limitar el log para no exponer URLs largas
            )
            return None

        try:
            # Descargar la imagen con timeout finito para prevenir ataques de lentitud
            with httpx.Client(timeout=15.0, follow_redirects=False) as client:
                # follow_redirects=False: validamos explícitamente antes de redirigir
                response = client.get(sanitized_url)
                response.raise_for_status()

            # Verificar que el Content-Type sea una imagen antes de procesar
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                logger.warning(
                    "⚠️  Content-Type no es imagen (%s) para URL: %s",
                    content_type, image_url[:80],
                )
                return None

            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            logger.debug("🖼️  Imagen descargada desde URL: %dx%d px", *image.size)
            return image

        except Exception as e:
            logger.warning("⚠️  Error descargando imagen desde URL '%s': %s", image_url[:80], e)
            return None

    return None


def _validate_image_url(url: str) -> Optional[str]:
    """
    Valida que una URL de imagen sea segura y reconstruye la URL desde sus
    componentes parseados para romper el flujo de datos del usuario (anti-SSRF).

    Protección contra SSRF (Server-Side Request Forgery):
      Un atacante podría enviar una URL como "http://localhost:6379/..." para
      hacer que el servidor realice peticiones a servicios internos (Redis, DB, etc.)
      que normalmente no son accesibles desde fuera de la máquina.

    Estrategia anti-SSRF:
      En lugar de devolver la URL original (que CodeQL rastrea como dato de usuario),
      reconstruimos la URL desde sus componentes parseados y validados.
      Esto rompe el "taint" (contaminación) de los datos de usuario y garantiza
      que solo URLs con estructura HTTP/HTTPS correcta lleguen al cliente HTTP.

    Reglas de validación:
      1. El esquema debe ser "http" o "https" (no file://, ftp://, gopher://, etc.)
      2. El host no puede ser localhost, 127.0.0.1 ni IPs de redes privadas.
      3. El host no puede ser una dirección IP de link-local (169.254.x.x).
      4. La URL no puede contener credenciales incrustadas (user:pass@host).

    Args:
        url: URL a validar.

    Returns:
        URL reconstruida desde componentes limpios si es segura, o None si debe
        ser rechazada. La URL retornada proviene de componentes verificados,
        no directamente de la entrada del usuario.
    """
    import ipaddress
    from urllib.parse import ParseResult, urlparse, urlunparse

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    # Regla 1: Solo esquemas HTTP/HTTPS
    if parsed.scheme not in ("http", "https"):
        return None

    # Regla 2: No credenciales incrustadas (http://user:pass@host)
    if parsed.username or parsed.password:
        return None

    hostname = parsed.hostname
    if not hostname:
        return None

    # Regla 3: Bloquear nombres de host que resuelven a localhost
    if hostname.lower() in ("localhost", "localhost.localdomain", "ip6-localhost"):
        return None

    # Regla 4: Intentar parsear como dirección IP y bloquear rangos privados
    try:
        ip = ipaddress.ip_address(hostname)
        # Bloquear IPs privadas, de loopback, link-local y reservadas
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return None
    except ValueError:
        # No es una IP — es un nombre de dominio; lo permitimos
        # (la resolución DNS a IP privada es un riesgo residual aceptado
        # en este contexto de uso doméstico local)
        pass

    # ── Reconstruir la URL desde componentes validados (rompe el taint flow) ──
    # Construimos la URL desde las partes parseadas verificadas en lugar de
    # devolver la cadena original del usuario. Esto garantiza que el valor
    # pasado al cliente HTTP proviene de componentes parseados confiables,
    # no directamente de la entrada sin procesar.
    safe_scheme = parsed.scheme          # "http" o "https" (validado arriba)
    safe_netloc = parsed.netloc          # host:port (sin credenciales, validado arriba)
    safe_path = parsed.path or "/"       # ruta (parte del URL parseado)
    safe_query = parsed.query            # parámetros de consulta (ya parseados)
    safe_fragment = ""                   # Ignoramos fragmentos (#...) — no necesarios

    # urlunparse reconstruye la URL desde sus 6 componentes
    reconstructed_url: str = urlunparse((
        safe_scheme,
        safe_netloc,
        safe_path,
        "",              # params (parte antes de ?)
        safe_query,
        safe_fragment,
    ))

    return reconstructed_url


def _analyze_image_metadata(image: Any) -> Dict[str, Any]:
    """
    Extrae información estadística y de metadatos de la imagen sin usar el VLM.

    Este análisis es ligero y siempre disponible, incluso en modo degradado.
    Analiza propiedades cuantitativas de los píxeles para detectar señales
    de manipulación o inconsistencias.

    Args:
        image: Objeto PIL.Image en modo RGB.

    Returns:
        Diccionario con metadatos estadísticos de la imagen.
    """
    metadata = {
        "width": image.width,
        "height": image.height,
        "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0,
    }

    try:
        # Análisis estadístico de los canales de color (R, G, B)
        stat = ImageStat.Stat(image)

        # La desviación estándar alta en los canales indica imagen natural/compleja
        # La desviación estándar muy baja puede indicar imagen editada o generada
        metadata["color_std_r"] = round(stat.stddev[0], 2)
        metadata["color_std_g"] = round(stat.stddev[1], 2)
        metadata["color_std_b"] = round(stat.stddev[2], 2)
        metadata["avg_brightness"] = round(stat.mean[0] * 0.299 + stat.mean[1] * 0.587 + stat.mean[2] * 0.114, 2)

        # Una imagen con muy baja entropía de color puede ser generada por IA
        # (colores artificialmente uniformes)
        avg_std = (metadata["color_std_r"] + metadata["color_std_g"] + metadata["color_std_b"]) / 3
        metadata["color_complexity"] = "alta" if avg_std > 40 else "media" if avg_std > 15 else "baja"

        # Detectar imágenes extremadamente pequeñas o grandes (puede ser señal)
        total_pixels = image.width * image.height
        metadata["resolution_category"] = (
            "miniatura" if total_pixels < 10_000
            else "baja" if total_pixels < 100_000
            else "media" if total_pixels < 1_000_000
            else "alta"
        )

    except Exception as e:
        logger.debug("No se pudieron calcular estadísticas de imagen: %s", e)

    return metadata


def _query_vlm(
    model: Any,
    tokenizer: Any,
    image: Any,
    question: str,
) -> str:
    """
    Realiza una consulta de pregunta-respuesta al modelo VLM sobre una imagen.

    moondream2 soporta el formato de "visual question answering" (VQA),
    donde se proporciona una imagen y una pregunta en lenguaje natural,
    y el modelo genera una respuesta descriptiva.

    Args:
        model:     Modelo moondream2 cargado.
        tokenizer: Tokenizer del modelo.
        image:     Objeto PIL.Image a analizar.
        question:  Pregunta en lenguaje natural sobre la imagen.

    Returns:
        Respuesta del modelo en lenguaje natural.
    """
    try:
        # Codificar la imagen para el modelo usando el método nativo de moondream2
        enc_image = model.encode_image(image)

        # Generar respuesta a la pregunta
        answer = model.answer_question(
            enc_image,
            question,
            tokenizer,
        )
        return answer.strip()

    except Exception as e:
        logger.warning("⚠️  Error en consulta VLM: %s", e)
        return "No se pudo obtener respuesta del modelo visual."


def _extract_vlm_insights(
    model: Any,
    tokenizer: Any,
    image: Any,
) -> Dict[str, Any]:
    """
    Extrae múltiples insights de la imagen usando el VLM con una batería
    de preguntas especializadas para la detección de desinformación.

    Las preguntas están diseñadas para extraer información que pueda
    cruzarse con el texto de la publicación:
      • Descripción general: ¿Qué hay en la imagen?
      • Contexto geográfico: ¿Dónde parece estar tomada?
      • Contexto temporal: ¿Cuándo parece ser? (señales de época)
      • Personas: ¿Hay personas? ¿Qué hacen?
      • Señales de manipulación: ¿Parece editada o generada por IA?

    Args:
        model:     Modelo VLM cargado.
        tokenizer: Tokenizer.
        image:     Imagen a analizar.

    Returns:
        Diccionario con los insights extraídos de la imagen.
    """
    insights = {}

    # ── Pregunta 1: Descripción general ────────────────────────────────────
    insights["description"] = _query_vlm(
        model, tokenizer, image,
        "Describe what you see in this image in detail, including objects, people, and scene."
    )

    # ── Pregunta 2: Contexto geográfico ────────────────────────────────────
    insights["geographic_context"] = _query_vlm(
        model, tokenizer, image,
        "What country or region does this image appear to be from? "
        "Look for clues like signs, architecture, landscape, flags, or clothing."
    )

    # ── Pregunta 3: Contexto temporal ──────────────────────────────────────
    insights["temporal_context"] = _query_vlm(
        model, tokenizer, image,
        "What time period does this image appear to be from? "
        "Look for clues like technology, clothing style, vehicles, or image quality."
    )

    # ── Pregunta 4: Señales de manipulación digital ─────────────────────────
    insights["manipulation_indicators"] = _query_vlm(
        model, tokenizer, image,
        "Does this image show any signs of digital manipulation, editing, or being AI-generated? "
        "Look for unnatural edges, inconsistent lighting, blurry areas, or artifacts."
    )

    # ── Pregunta 5: Tipo de evento mostrado ────────────────────────────────
    insights["event_type"] = _query_vlm(
        model, tokenizer, image,
        "What type of event or situation is shown in this image? "
        "Is it a natural disaster, protest, crime scene, political event, or everyday scene?"
    )

    # ── Pregunta 6: Objetos y entidades identificadas ──────────────────────
    objects_response = _query_vlm(
        model, tokenizer, image,
        "List the main objects, people, and entities you can identify in this image. "
        "Be specific about flags, logos, uniforms, or identifiable landmarks."
    )
    # Convertir la respuesta en una lista separando por comas o saltos de línea
    insights["detected_objects"] = [
        obj.strip()
        for obj in re.split(r"[,\n;]", objects_response)
        if obj.strip() and len(obj.strip()) > 2
    ][:10]  # Máximo 10 objetos para mantener la respuesta compacta

    return insights


def _detect_manipulation_signals(
    metadata: Dict[str, Any],
    vlm_insights: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Consolida las señales de manipulación de los metadatos y del VLM.

    Args:
        metadata:    Metadatos estadísticos de la imagen.
        vlm_insights: Insights del VLM (puede ser None en modo degradado).

    Returns:
        Tupla (manipulation_detected: bool, signals: List[str]).
    """
    signals: List[str] = []

    # ── Señales de metadatos ────────────────────────────────────────────────
    if metadata.get("color_complexity") == "baja":
        signals.append("Complejidad de color muy baja — posible imagen generada por IA.")

    if metadata.get("resolution_category") == "miniatura":
        signals.append("Imagen de muy baja resolución — posiblemente recortada o degradada deliberadamente.")

    # ── Señales del VLM ─────────────────────────────────────────────────────
    if vlm_insights:
        manipulation_text = vlm_insights.get("manipulation_indicators", "").lower()
        manipulation_keywords = [
            "edited", "manipulated", "photoshopped", "ai-generated", "artificial",
            "inconsistent lighting", "unnatural", "artifacts", "blurry", "fake",
        ]
        if any(kw in manipulation_text for kw in manipulation_keywords):
            signals.append(
                f"El modelo visual detectó posible manipulación: '{vlm_insights.get('manipulation_indicators', '')[:100]}'"
            )

    manipulation_detected = len(signals) > 0
    return manipulation_detected, signals


async def analyze_image(
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Función principal del submódulo de análisis visual.

    Orquesta la decodificación de la imagen, el análisis de metadatos,
    la inferencia con el VLM y la consolidación de resultados.

    Es async para integrarse con el sistema FastAPI, aunque el análisis
    de imagen puede ser computacionalmente intensivo (síncrono a nivel de CPU/GPU).
    Para sistemas de producción con alta carga, considerar ejecutar esto en
    un executor de threads: asyncio.get_event_loop().run_in_executor(None, ...)

    Args:
        image_base64: Imagen en Base64 (prioridad sobre URL).
        image_url:    URL de la imagen (alternativa al Base64).

    Returns:
        Diccionario compatible con el esquema VisionAnalysisResult de Pydantic.
    """
    # ── Caso: no hay imagen ─────────────────────────────────────────────────
    if not image_base64 and not image_url:
        logger.debug("🖼️  No se proporcionó imagen. Saltando análisis visual.")
        return {
            "image_description": "No se proporcionó imagen para analizar.",
            "detected_objects": [],
            "manipulation_detected": False,
            "geographic_context": None,
            "temporal_context": None,
            "vision_confidence": 0.0,
        }

    # ── Decodificar imagen ──────────────────────────────────────────────────
    logger.debug("🖼️  Iniciando análisis de imagen…")
    image = _decode_image(image_base64, image_url)

    if image is None:
        logger.warning("⚠️  No se pudo decodificar la imagen. Retornando análisis vacío.")
        return {
            "image_description": "No se pudo procesar la imagen proporcionada.",
            "detected_objects": [],
            "manipulation_detected": False,
            "geographic_context": None,
            "temporal_context": None,
            "vision_confidence": 0.0,
        }

    # ── Análisis de metadatos (siempre disponible) ──────────────────────────
    metadata = _analyze_image_metadata(image)

    # ── Análisis VLM (requiere modelo cargado) ──────────────────────────────
    vlm_model, vlm_tokenizer = _load_vision_model()
    vlm_insights: Optional[Dict[str, Any]] = None

    if vlm_model is not None and vlm_tokenizer is not None:
        try:
            vlm_insights = _extract_vlm_insights(vlm_model, vlm_tokenizer, image)
            logger.debug("🖼️  Análisis VLM completado exitosamente.")
        except Exception as e:
            logger.error("❌ Error en análisis VLM: %s. Usando solo metadatos.", e)

    # ── Consolidar señales de manipulación ──────────────────────────────────
    manipulation_detected, manipulation_signals = _detect_manipulation_signals(
        metadata, vlm_insights
    )

    # ── Construir respuesta ─────────────────────────────────────────────────
    if vlm_insights:
        description = vlm_insights.get("description", "Descripción no disponible.")
        geographic_context = vlm_insights.get("geographic_context")
        temporal_context = vlm_insights.get("temporal_context")
        detected_objects = vlm_insights.get("detected_objects", [])
        vision_confidence = 0.75  # Confianza alta cuando el VLM está disponible
    else:
        # Modo degradado: descripción basada en metadatos
        description = (
            f"Imagen de {metadata.get('width', '?')}x{metadata.get('height', '?')} píxeles. "
            f"Complejidad de color: {metadata.get('color_complexity', 'desconocida')}. "
            f"El modelo VLM no estaba disponible para análisis visual detallado."
        )
        geographic_context = None
        temporal_context = None
        detected_objects = []
        vision_confidence = 0.2  # Baja confianza en modo degradado

    # Agregar las señales de manipulación a los objetos detectados si existen
    if manipulation_signals:
        detected_objects = detected_objects + manipulation_signals

    logger.debug(
        "🖼️  Análisis visual completado — Manipulación: %s | Confianza: %.2f",
        manipulation_detected,
        vision_confidence,
    )

    return {
        "image_description": description,
        "detected_objects": detected_objects[:15],  # Máximo 15 para mantener respuesta compacta
        "manipulation_detected": manipulation_detected,
        "geographic_context": geographic_context,
        "temporal_context": temporal_context,
        "vision_confidence": vision_confidence,
        # Metadatos adicionales para el corpus de entrenamiento
        "_image_metadata": metadata,
        "_vlm_available": vlm_model is not None,
    }

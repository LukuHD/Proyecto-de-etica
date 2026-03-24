"""
db_manager.py — Gestor de la base de datos JSON local.

Este módulo es el único punto de acceso al archivo de persistencia de datos.
Implementa todas las operaciones de lectura, escritura y consulta sobre el
archivo JSON que actúa como base de datos sin requerir ningún motor externo
(sin PostgreSQL, sin SQLite, sin Redis — solo archivos del sistema operativo).

Decisión arquitectónica — ¿Por qué JSON en lugar de SQLite?
  El requisito explícito del sistema prohíbe dependencias de motores de base
  de datos externos. JSON puro cumple este requisito y además:
    • Es legible por humanos (facilita la depuración y auditoría).
    • Es nativamente soportado por Python (módulo json estándar).
    • Permite exportar el corpus de entrenamiento con un simple script.
    • No requiere instalación de controladores ni servidores.

Consideraciones de concurrencia:
  FastAPI es asíncrono por naturaleza y puede recibir varias peticiones
  simultáneas. Para proteger las escrituras al archivo JSON contra condiciones
  de carrera (race conditions), usamos un asyncio.Lock() que garantiza que
  solo una coroutine escribe al archivo en un momento dado.
  Las LECTURAS se pueden realizar sin lock porque Python garantiza atomicidad
  en operaciones de lectura de archivos en sistemas Unix (POSIX).
  Sin embargo, por consistencia de diseño, también protegemos las lecturas
  que ocurren durante una secuencia de lectura-modificación-escritura.

Estructura del archivo JSON de base de datos:
  {
    "schema_version": "1.0",
    "created_at": "2024-06-15T12:00:00Z",
    "last_updated": "2024-06-15T18:30:00Z",
    "total_analyses": 42,
    "records": {
      "<sha256_hash_64chars>": {
        "hash": "<sha256_hash>",
        "post_text": "texto original...",
        "author_name": "Juan Pérez",
        "post_timestamp": "2024-06-15T14:30:00Z",
        "analyzed_at": "2024-06-15T14:31:05Z",
        "category": "fraude_financiero",
        "confidence": 0.93,
        "explanation": "El texto emplea tácticas de ingeniería social…",
        "multimodal_discrepancies": ["…"],
        "text_patterns": ["promesa_financiera_irreal"],
        "has_image": true,
        "cached_hits": 2
      }
    }
  }

  Nota: Usamos un objeto (dict) indexado por hash para el campo "records"
  en lugar de una lista, lo que permite búsquedas en O(1) por hash sin
  iterar sobre todos los registros.
"""

import asyncio     # Para el mecanismo de lock asíncrono
import json        # Módulo estándar de serialización JSON
import logging
import os          # Para verificar existencia de archivos y rutas del sistema
from datetime import datetime, timezone
from pathlib import Path       # API moderna de rutas del sistema de archivos
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Configuración de rutas ──────────────────────────────────────────────────
# La base de datos JSON se almacena en el directorio `data/` relativo al
# directorio raíz del proyecto backend. Usamos Path para compatibilidad
# multiplataforma (Windows usa \, Unix usa /).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/
_DATA_DIR = _PROJECT_ROOT / "data"
_DB_FILE_PATH = _DATA_DIR / "analysis_database.json"

# ── Esquema inicial de la base de datos ────────────────────────────────────
# Este diccionario define la estructura exacta que tendrá el archivo JSON
# cuando se crea por primera vez. Actúa como un "CREATE TABLE" en SQL.
_INITIAL_DB_SCHEMA: Dict[str, Any] = {
    "schema_version": "1.0",
    "description": (
        "Base de datos local del sistema de detección de desinformación. "
        "No editar manualmente — gestionado por db_manager.py"
    ),
    "created_at": "",       # Se rellenará con la fecha real en initialize_database()
    "last_updated": "",     # Se actualizará en cada escritura
    "total_analyses": 0,    # Contador global de análisis realizados
    "records": {},          # Diccionario indexado por hash SHA-256
}

# ── Lock global para escrituras concurrentes ───────────────────────────────
# asyncio.Lock() garantiza exclusión mutua en operaciones de escritura.
# Se debe crear en el hilo del event loop principal de asyncio.
# Declaramos como None y lo inicializamos en initialize_database() para
# evitar problemas con loops de asyncio en entornos de testing.
_db_write_lock: Optional[asyncio.Lock] = None


def _get_write_lock() -> asyncio.Lock:
    """
    Retorna el lock global de escritura, creándolo si aún no existe.

    Decisión de diseño: la inicialización lazy del Lock garantiza que siempre
    se cree en el event loop correcto, lo cual es crucial si la aplicación
    se usa en contextos de testing con loops aislados.
    """
    global _db_write_lock
    if _db_write_lock is None:
        _db_write_lock = asyncio.Lock()
    return _db_write_lock


def _now_iso() -> str:
    """Retorna la fecha y hora UTC actual en formato ISO 8601."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_db_sync() -> Dict[str, Any]:
    """
    Lee y parsea el archivo JSON de forma SINCRÓNICA.

    Esta función interna se usa dentro de contextos ya protegidos por el Lock
    asíncrono. No se llama directamente desde los routers (que son asíncronos).

    Returns:
        Diccionario Python con el contenido completo de la base de datos.

    Raises:
        json.JSONDecodeError: Si el archivo está corrupto (no es JSON válido).
        FileNotFoundError:    Si el archivo no existe (no debería ocurrir después
                              de initialize_database(), pero se maneja por seguridad).
    """
    if not _DB_FILE_PATH.exists():
        logger.warning("⚠️  Archivo de base de datos no encontrado. Reinicializando…")
        _write_db_sync(_INITIAL_DB_SCHEMA.copy())
        return _INITIAL_DB_SCHEMA.copy()

    with open(_DB_FILE_PATH, "r", encoding="utf-8") as f:
        # json.load() parsea el archivo completo en memoria.
        # Para millones de registros, consideraríamos streaming (ijson),
        # pero para uso doméstico personal, esto es perfectamente eficiente.
        data = json.load(f)

    return data


def _write_db_sync(data: Dict[str, Any]) -> None:
    """
    Escribe el diccionario completo al archivo JSON de forma SINCRÓNICA.

    Estrategia de escritura segura (atomic write):
      1. Escribimos a un archivo temporal en el mismo directorio.
      2. Una vez completada la escritura, renombramos el temporal al nombre
         definitivo (operación atómica en sistemas POSIX).
      Esto garantiza que si el proceso se interrumpe durante la escritura
      (corte de luz, error de disco), el archivo original permanece intacto
      y no queda en un estado inconsistente (JSON parcialmente escrito = corrupto).

    Args:
        data: Diccionario con el contenido completo a persistir.
    """
    # Actualizar timestamp de última modificación
    data["last_updated"] = _now_iso()

    # Ruta del archivo temporal (en el mismo directorio para garantizar
    # que rename() sea una operación en el mismo volumen de disco)
    tmp_path = _DB_FILE_PATH.with_suffix(".json.tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            # indent=2 hace el JSON legible por humanos sin ocupar demasiado espacio.
            # ensure_ascii=False preserva caracteres UTF-8 (tildes, ñ, emojis).
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Rename atómico: sustituye el archivo definitivo con el temporal
        os.replace(tmp_path, _DB_FILE_PATH)

    except Exception as e:
        logger.error("❌ Error al escribir la base de datos JSON: %s", e)
        # Limpiar el archivo temporal si quedó a medias
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise  # Re-lanzamos la excepción para que el router la maneje


async def initialize_database() -> None:
    """
    Inicializa el archivo JSON de base de datos si no existe.

    Esta función es llamada por el evento de startup de FastAPI (lifespan).
    Garantiza que cuando el primer request llegue, el archivo ya exista y
    tenga una estructura válida.

    Flujo:
      1. Crea el directorio `data/` si no existe.
      2. Si el archivo de DB no existe → crea uno nuevo con el esquema vacío.
      3. Si el archivo existe → lo valida para detectar posibles corrupciones.

    Es async para poder ser awaited en el contexto del lifespan de FastAPI,
    aunque la operación de archivo en sí sea sincrónica (el I/O de archivos
    locales es tan rápido que no justifica aiofiles en el startup).
    """
    # Crear directorio de datos si no existe (equivalente a mkdir -p)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not _DB_FILE_PATH.exists():
        # Primera ejecución: crear la base de datos con el esquema inicial
        logger.info("📂 Creando base de datos JSON en: %s", _DB_FILE_PATH)

        initial_data = _INITIAL_DB_SCHEMA.copy()
        initial_data["created_at"] = _now_iso()
        initial_data["last_updated"] = _now_iso()

        _write_db_sync(initial_data)
        logger.info("✅ Base de datos creada exitosamente.")
    else:
        # La base de datos ya existe: validar que sea JSON válido y tenga
        # la estructura esperada (defensa contra archivos corruptos).
        try:
            data = _read_db_sync()

            # Validación mínima de estructura
            required_keys = {"schema_version", "records", "total_analyses"}
            missing_keys = required_keys - set(data.keys())

            if missing_keys:
                logger.warning(
                    "⚠️  Base de datos con estructura incompleta. "
                    "Claves faltantes: %s. Reparando…",
                    missing_keys,
                )
                # Agregar claves faltantes preservando los datos existentes
                for key in missing_keys:
                    data[key] = _INITIAL_DB_SCHEMA.get(key, {})
                _write_db_sync(data)

            record_count = len(data.get("records", {}))
            logger.info(
                "✅ Base de datos encontrada con %d registros previos.", record_count
            )

        except json.JSONDecodeError as e:
            logger.error(
                "❌ Base de datos JSON corrupta: %s. Creando respaldo y reiniciando…", e
            )
            # Respaldar el archivo corrupto antes de reemplazarlo
            backup_path = _DB_FILE_PATH.with_suffix(f".json.bak_{_now_iso().replace(':', '-')}")
            _DB_FILE_PATH.rename(backup_path)
            logger.warning("Respaldo creado en: %s", backup_path)

            initial_data = _INITIAL_DB_SCHEMA.copy()
            initial_data["created_at"] = _now_iso()
            _write_db_sync(initial_data)


async def check_cache(post_hash: str) -> Optional[Dict[str, Any]]:
    """
    Verifica si una publicación ya fue analizada previamente (consulta de caché).

    Busca el hash SHA-256 de la publicación en el índice del archivo JSON.
    La búsqueda es O(1) gracias a que `records` es un diccionario indexado
    por hash (equivalente a un SELECT por clave primaria en SQL).

    Args:
        post_hash: Hash SHA-256 de 64 caracteres de la publicación.

    Returns:
        El registro completo del análisis previo si existe en caché,
        o None si la publicación es nueva y nunca ha sido analizada.
    """
    # Leemos la base de datos sin lock para la consulta de caché.
    # En sistemas Unix, read() de un archivo que no está siendo escrito
    # concurrentemente es inherentemente seguro.
    data = _read_db_sync()
    records: Dict[str, Any] = data.get("records", {})

    if post_hash in records:
        logger.info("💾 Cache HIT — Hash encontrado: %s…", post_hash[:16])

        # Incrementar el contador de hits para el registro (escritura con lock)
        async with _get_write_lock():
            fresh_data = _read_db_sync()  # Re-leer para evitar race conditions
            if post_hash in fresh_data["records"]:
                fresh_data["records"][post_hash]["cached_hits"] = (
                    fresh_data["records"][post_hash].get("cached_hits", 0) + 1
                )
                _write_db_sync(fresh_data)

        return records[post_hash]

    logger.info("🔍 Cache MISS — Hash nuevo: %s…", post_hash[:16])
    return None  # La publicación es nueva, requiere análisis completo


async def save_analysis_result(
    post_hash: str,
    post_text: str,
    author_name: str,
    post_timestamp: str,
    analysis_result: Dict[str, Any],
) -> None:
    """
    Persiste el resultado de un nuevo análisis en la base de datos JSON.

    Esta función se ejecuta DESPUÉS de que el motor de IA completa el análisis,
    guardando el resultado para futuras consultas de caché y para el corpus
    de entrenamiento continuo.

    Flujo con protección de concurrencia:
      1. Adquirir el Lock de escritura (bloquea otras coroutines de escritura).
      2. Leer el estado más reciente del archivo (puede haber cambiado desde
         la última lectura de esta coroutine).
      3. Agregar el nuevo registro.
      4. Actualizar contadores globales.
      5. Escribir el archivo de forma atómica.
      6. Liberar el Lock automáticamente al salir del bloque `async with`.

    Args:
        post_hash:       Hash SHA-256 de la publicación.
        post_text:       Texto original de la publicación (para el corpus).
        author_name:     Nombre del autor.
        post_timestamp:  Timestamp original del post en ISO 8601.
        analysis_result: Diccionario con el resultado completo del análisis IA.
    """
    async with _get_write_lock():
        # Re-leer el archivo dentro del lock para obtener el estado más reciente
        # (otra coroutine pudo haber escrito mientras esperábamos el lock)
        data = _read_db_sync()

        # Construir el registro completo que se almacenará en el JSON
        new_record: Dict[str, Any] = {
            "hash": post_hash,
            "post_text": post_text,
            "author_name": author_name,
            "post_timestamp": post_timestamp,
            "analyzed_at": _now_iso(),

            # Resultado del análisis (extraemos los campos clave para indexación rápida)
            "category": analysis_result.get("category"),
            "confidence": analysis_result.get("confidence"),
            "explanation": analysis_result.get("explanation"),
            "multimodal_discrepancies": analysis_result.get("multimodal_discrepancies", []),

            # Señales del análisis de texto (útiles para el corpus de entrenamiento)
            "text_patterns": analysis_result.get("text_analysis", {}).get("detected_patterns", []),
            "manipulation_indicators": analysis_result.get("text_analysis", {}).get("manipulation_indicators", []),
            "sentiment_score": analysis_result.get("text_analysis", {}).get("sentiment_score", 0.0),

            # Señales del análisis visual
            "image_description": analysis_result.get("vision_analysis", {}).get("image_description", ""),
            "has_image": analysis_result.get("vision_analysis", {}).get("vision_confidence", 0) > 0,

            # Metadatos de uso
            "cached_hits": 0,   # Número de veces que este resultado fue devuelto desde caché
        }

        # Insertar el nuevo registro en el índice
        data["records"][post_hash] = new_record

        # Actualizar el contador global de análisis realizados
        data["total_analyses"] = len(data["records"])

        # Escribir el archivo actualizado de forma atómica
        _write_db_sync(data)

        logger.info(
            "💾 Análisis guardado en DB — Hash: %s… | Categoría: %s | Confianza: %.1f%%",
            post_hash[:16],
            analysis_result.get("category", "desconocida"),
            analysis_result.get("confidence", 0.0) * 100,
        )


async def get_all_records() -> Dict[str, Any]:
    """
    Retorna todos los registros de la base de datos.

    Utilizado principalmente por el script de aprendizaje continuo para
    extraer el corpus completo de análisis previos.

    Returns:
        El diccionario completo de la base de datos incluyendo metadatos y registros.
    """
    return _read_db_sync()


async def get_statistics() -> Dict[str, Any]:
    """
    Calcula y retorna estadísticas resumidas de la base de datos.

    Útil para el endpoint de monitoreo y para el dashboard de la extensión.
    """
    data = _read_db_sync()
    records = data.get("records", {})

    # Contar por categoría
    category_counts: Dict[str, int] = {}
    confidence_values: list[float] = []
    low_confidence_count = 0

    for record in records.values():
        # Contador por categoría
        cat = record.get("category", "desconocida")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        # Recopilar valores de confianza para estadísticas
        conf = record.get("confidence", 0.0)
        confidence_values.append(conf)

        # Contar registros con baja confianza (candidatos para revisión humana)
        if conf < 0.6:
            low_confidence_count += 1

    avg_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return {
        "total_records": len(records),
        "total_analyses": data.get("total_analyses", 0),
        "category_distribution": category_counts,
        "average_confidence": round(avg_confidence, 4),
        "low_confidence_records": low_confidence_count,
        "database_created_at": data.get("created_at"),
        "last_updated": data.get("last_updated"),
    }

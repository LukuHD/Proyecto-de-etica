"""
config.py — Configuración centralizada de la aplicación.

Centraliza todos los parámetros configurables del sistema en un único lugar,
usando variables de entorno cuando están disponibles (para despliegue) o
valores por defecto razonables para el uso doméstico local.

Uso:
  from app.config import settings
  print(settings.HOST)  # "127.0.0.1"
"""

import os
from pathlib import Path


class Settings:
    """
    Clase de configuración de la aplicación.

    Todos los parámetros pueden ser sobreescritos con variables de entorno.
    Ejemplo: HOST=0.0.0.0 python -m app.main (para aceptar conexiones externas)
    """

    # ── Red ──────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ── Base de datos ────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DB_FILE: Path = DATA_DIR / "analysis_database.json"

    # ── IA — Modelos ─────────────────────────────────────────────────────────
    # Modelo de sentimiento NLP (texto)
    NLP_MODEL_ID: str = os.getenv(
        "NLP_MODEL_ID",
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

    # Modelo VLM para análisis de imágenes
    VLM_MODEL_ID: str = os.getenv("VLM_MODEL_ID", "vikhyatk/moondream2")
    VLM_REVISION: str = os.getenv("VLM_REVISION", "2024-08-26")

    # ── IA — Hardware ────────────────────────────────────────────────────────
    # Forzar CPU aunque haya GPU disponible (útil para máquinas con poca VRAM)
    FORCE_CPU: bool = os.getenv("FORCE_CPU", "false").lower() == "true"

    # ── Límites de seguridad ─────────────────────────────────────────────────
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
    MAX_IMAGE_SIZE_MB: float = float(os.getenv("MAX_IMAGE_SIZE_MB", "10.0"))
    HTTP_TIMEOUT_SECONDS: int = int(os.getenv("HTTP_TIMEOUT_SECONDS", "15"))

    # ── Aprendizaje continuo ─────────────────────────────────────────────────
    LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.65"))
    HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.90"))


# Instancia global de configuración — importar desde aquí
settings = Settings()

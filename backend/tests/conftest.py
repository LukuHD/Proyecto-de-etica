"""
conftest.py — Configuración compartida de pytest para todos los tests.

Define fixtures reutilizables que evitan duplicación de código entre
los distintos módulos de test.
"""

import sys
from pathlib import Path

import pytest

# Asegurar que el directorio `backend/` está en el PYTHONPATH para imports
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


@pytest.fixture
def temp_db_dir(tmp_path: Path) -> Path:
    """
    Fixture que proporciona un directorio temporal para la base de datos de tests.
    Se crea un directorio fresco para cada test que lo solicite, garantizando
    aislamiento completo entre tests.
    """
    db_dir = tmp_path / "test_data"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


@pytest.fixture
def sample_post_text() -> str:
    """Texto de ejemplo de publicación fraudulenta para tests."""
    return "¡¡¡GANA $5000 DIARIOS TRABAJANDO DESDE CASA!!! Ingreso pasivo garantizado. Sin experiencia. ÚNETE AHORA. Haz clic aquí 👇"


@pytest.fixture
def sample_safe_text() -> str:
    """Texto de ejemplo de publicación segura para tests."""
    return "Hoy tuve un buen día con mi familia. Fuimos al parque y disfrutamos del sol."


@pytest.fixture
def sample_political_text() -> str:
    """Texto de ejemplo con desinformación política para tests."""
    return "¡El gobierno oculta la verdad! Fraude electoral comprobado. La prensa no lo dirá. Comparte antes de que lo borren."

"""
test_db_manager.py — Tests de integración para el gestor de base de datos JSON.

Valida el ciclo completo de inicialización, escritura, lectura y caché
del sistema de persistencia local basado en archivos JSON.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Asegurar que el directorio backend está en el path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


# ── Fixtures específicos para este módulo ──────────────────────────────────
@pytest.fixture(autouse=True)
def patch_db_path(tmp_path: Path, monkeypatch):
    """
    Parchea las rutas del módulo db_manager para usar un directorio temporal.
    Esto garantiza que los tests no toquen la base de datos real del proyecto.
    """
    import app.utils.db_manager as db_mod

    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    test_db_file = test_data_dir / "test_analysis_database.json"

    # Monkeypatch las rutas y resetear el lock global
    monkeypatch.setattr(db_mod, "_DATA_DIR", test_data_dir)
    monkeypatch.setattr(db_mod, "_DB_FILE_PATH", test_db_file)
    monkeypatch.setattr(db_mod, "_db_write_lock", None)  # Resetear lock entre tests

    yield test_db_file


class TestInitializeDatabase:
    """Tests para la función de inicialización de la base de datos."""

    @pytest.mark.asyncio
    async def test_creates_db_file_if_not_exists(self, patch_db_path):
        """Debe crear el archivo JSON si no existe."""
        from app.utils.db_manager import initialize_database

        assert not patch_db_path.exists()

        await initialize_database()

        assert patch_db_path.exists()

    @pytest.mark.asyncio
    async def test_db_has_correct_schema(self, patch_db_path):
        """El archivo creado debe tener el esquema correcto."""
        from app.utils.db_manager import initialize_database

        await initialize_database()

        with open(patch_db_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verificar campos obligatorios del esquema
        assert "schema_version" in data
        assert "records" in data
        assert "total_analyses" in data
        assert "created_at" in data
        assert "last_updated" in data

        # Los registros deben empezar vacíos
        assert isinstance(data["records"], dict)
        assert len(data["records"]) == 0
        assert data["total_analyses"] == 0

    @pytest.mark.asyncio
    async def test_does_not_overwrite_existing_db(self, patch_db_path):
        """Si la DB ya existe con datos, no debe sobreescribirla."""
        from app.utils.db_manager import initialize_database

        # Crear una DB con un registro simulado
        existing_data = {
            "schema_version": "1.0",
            "created_at": "2024-01-01T00:00:00Z",
            "last_updated": "2024-01-01T00:00:00Z",
            "total_analyses": 1,
            "records": {
                "abc123": {"hash": "abc123", "post_text": "test", "category": "publicacion_segura"}
            },
        }
        patch_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(patch_db_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f)

        await initialize_database()

        # Los datos existentes deben preservarse
        with open(patch_db_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "abc123" in data["records"]
        assert data["total_analyses"] == 1

    @pytest.mark.asyncio
    async def test_recovers_from_corrupted_db(self, patch_db_path, tmp_path):
        """Si la DB está corrupta, debe crear un respaldo y reiniciar."""
        from app.utils.db_manager import initialize_database

        # Escribir JSON inválido
        patch_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(patch_db_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json content {{{}}")

        await initialize_database()

        # Debe haberse creado un archivo de respaldo
        backup_files = list(patch_db_path.parent.glob("*.bak*"))
        assert len(backup_files) > 0

        # La DB debe ser válida ahora
        with open(patch_db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "records" in data


class TestCacheOperations:
    """Tests para las operaciones de caché (check_cache y save_analysis_result)."""

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none_for_new_hash(self, patch_db_path):
        """Un hash nuevo debe retornar None (cache miss)."""
        from app.utils.db_manager import check_cache, initialize_database

        await initialize_database()

        result = await check_cache("hashquenoexiste" * 4)  # 64 chars
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_after_saving_result(self, patch_db_path):
        """Después de guardar un análisis, el mismo hash debe retornar el resultado."""
        from app.utils.db_manager import check_cache, initialize_database, save_analysis_result

        await initialize_database()

        test_hash = "a" * 64  # Hash de prueba de 64 caracteres

        analysis = {
            "category": "fraude_financiero",
            "confidence": 0.92,
            "explanation": "Fraude detectado en el texto.",
            "multimodal_discrepancies": [],
            "text_analysis": {
                "detected_patterns": ["fraude_financiero"],
                "sentiment_score": -0.5,
                "manipulation_indicators": [],
                "credibility_signals": [],
                "text_confidence": 0.92,
            },
            "vision_analysis": {
                "image_description": "No hay imagen.",
                "vision_confidence": 0.0,
            },
        }

        # Guardar el análisis
        await save_analysis_result(
            post_hash=test_hash,
            post_text="GANA DINERO FÁCIL",
            author_name="Spammer",
            post_timestamp="2024-06-15T14:30:00Z",
            analysis_result=analysis,
        )

        # Verificar cache hit
        cached = await check_cache(test_hash)

        assert cached is not None
        assert cached["hash"] == test_hash
        assert cached["category"] == "fraude_financiero"
        assert cached["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_cache_increments_hit_counter(self, patch_db_path):
        """Cada consulta de caché exitosa debe incrementar el contador de hits."""
        from app.utils.db_manager import check_cache, initialize_database, save_analysis_result

        await initialize_database()

        test_hash = "b" * 64

        await save_analysis_result(
            post_hash=test_hash,
            post_text="texto de prueba",
            author_name="autor",
            post_timestamp="2024-01-01T00:00:00Z",
            analysis_result={
                "category": "publicacion_segura",
                "confidence": 0.85,
                "explanation": "Contenido seguro.",
                "multimodal_discrepancies": [],
                "text_analysis": {"detected_patterns": [], "sentiment_score": 0.1,
                                   "manipulation_indicators": [], "credibility_signals": [],
                                   "text_confidence": 0.85},
                "vision_analysis": {"image_description": "", "vision_confidence": 0.0},
            },
        )

        # Primera consulta
        await check_cache(test_hash)
        # Segunda consulta
        cached = await check_cache(test_hash)

        # El contador de hits debe ser > 0 (el valor exacto depende de las escrituras)
        assert cached is not None
        assert cached.get("cached_hits", 0) >= 1


class TestStatistics:
    """Tests para las estadísticas de la base de datos."""

    @pytest.mark.asyncio
    async def test_empty_db_statistics(self, patch_db_path):
        """La DB vacía debe retornar estadísticas con ceros."""
        from app.utils.db_manager import get_statistics, initialize_database

        await initialize_database()

        stats = await get_statistics()

        assert stats["total_records"] == 0
        assert stats["average_confidence"] == 0.0
        assert stats["low_confidence_records"] == 0

    @pytest.mark.asyncio
    async def test_statistics_after_adding_records(self, patch_db_path):
        """Las estadísticas deben reflejar correctamente los registros añadidos."""
        from app.utils.db_manager import get_statistics, initialize_database, save_analysis_result

        await initialize_database()

        # Añadir dos registros
        for i, (cat, conf) in enumerate([("fraude_financiero", 0.9), ("publicacion_segura", 0.8)]):
            await save_analysis_result(
                post_hash="c" * 63 + str(i),  # Hash único de 64 chars
                post_text=f"texto {i}",
                author_name="test",
                post_timestamp="2024-01-01T00:00:00Z",
                analysis_result={
                    "category": cat,
                    "confidence": conf,
                    "explanation": "test",
                    "multimodal_discrepancies": [],
                    "text_analysis": {"detected_patterns": [], "sentiment_score": 0.0,
                                       "manipulation_indicators": [], "credibility_signals": [],
                                       "text_confidence": conf},
                    "vision_analysis": {"image_description": "", "vision_confidence": 0.0},
                },
            )

        stats = await get_statistics()

        assert stats["total_records"] == 2
        assert "fraude_financiero" in stats["category_distribution"]
        assert "publicacion_segura" in stats["category_distribution"]
        assert stats["average_confidence"] == pytest.approx(0.85, abs=0.01)

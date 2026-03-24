"""
test_api.py — Tests de integración para los endpoints HTTP de FastAPI.

Usa el TestClient asíncrono de httpx para simular peticiones HTTP reales
al servidor, validando el comportamiento end-to-end del sistema.

Los tests parchean el motor de IA para evitar cargar modelos en el CI/CD,
concentrándose en validar la lógica del router, caché y serialización.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


# ── Mock del resultado del motor IA ────────────────────────────────────────
_MOCK_AI_RESULT: Dict[str, Any] = {
    "category": "fraude_financiero",
    "confidence": 0.93,
    "explanation": "El texto emplea tácticas clásicas de fraude financiero.",
    "multimodal_discrepancies": [],
    "text_analysis": {
        "detected_patterns": ["fraude_financiero"],
        "sentiment_score": -0.4,
        "manipulation_indicators": ["alto_numero_exclamaciones: 5"],
        "credibility_signals": [],
        "text_confidence": 0.93,
    },
    "vision_analysis": {
        "image_description": "No se proporcionó imagen para analizar.",
        "detected_objects": [],
        "manipulation_detected": False,
        "geographic_context": None,
        "temporal_context": None,
        "vision_confidence": 0.0,
    },
    "analyzed_at": "2024-06-15T14:31:05Z",
}

_MOCK_SAFE_RESULT: Dict[str, Any] = {
    "category": "publicacion_segura",
    "confidence": 0.80,
    "explanation": "No se detectaron señales de desinformación.",
    "multimodal_discrepancies": [],
    "text_analysis": {
        "detected_patterns": [],
        "sentiment_score": 0.3,
        "manipulation_indicators": [],
        "credibility_signals": [],
        "text_confidence": 0.30,
    },
    "vision_analysis": {
        "image_description": "No se proporcionó imagen.",
        "detected_objects": [],
        "manipulation_detected": False,
        "geographic_context": None,
        "temporal_context": None,
        "vision_confidence": 0.0,
    },
    "analyzed_at": "2024-06-15T14:31:10Z",
}


@pytest.fixture
def test_client(tmp_path: Path):
    """
    Fixture que crea un TestClient con la DB parchada al directorio temporal.
    El motor de IA es mockeado para evitar cargar modelos pesados en los tests.
    """
    import app.utils.db_manager as db_mod

    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    test_db_file = test_data_dir / "test_analysis_database.json"

    # Parchear rutas de DB
    original_data_dir = db_mod._DATA_DIR
    original_db_file = db_mod._DB_FILE_PATH
    original_lock = db_mod._db_write_lock

    db_mod._DATA_DIR = test_data_dir
    db_mod._DB_FILE_PATH = test_db_file
    db_mod._db_write_lock = None

    from app.main import app
    client = TestClient(app)

    yield client

    # Restaurar rutas originales
    db_mod._DATA_DIR = original_data_dir
    db_mod._DB_FILE_PATH = original_db_file
    db_mod._db_write_lock = original_lock


class TestHealthEndpoint:
    """Tests para el endpoint de verificación de salud."""

    def test_health_returns_200(self, test_client):
        """El endpoint /health debe retornar HTTP 200."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, test_client):
        """El endpoint /health debe retornar {"status": "ok"}."""
        response = test_client.get("/health")
        data = response.json()

        assert data["status"] == "ok"
        assert "service" in data
        assert "version" in data


class TestAnalyzeEndpoint:
    """Tests para el endpoint principal POST /api/v1/analyze."""

    def _build_request_payload(
        self,
        text: str = "texto de prueba",
        author: str = "Test Author",
        timestamp: str = "2024-06-15T14:30:00Z",
        image_base64: str = None,
        image_url: str = None,
    ) -> dict:
        """Construye un payload válido para el endpoint de análisis."""
        payload = {
            "post_text": text,
            "author_name": author,
            "post_timestamp": timestamp,
        }
        if image_base64:
            payload["image_base64"] = image_base64
        if image_url:
            payload["image_url"] = image_url
        return payload

    def test_analyze_returns_200_with_valid_payload(self, test_client):
        """Debe retornar HTTP 200 con un payload válido."""
        with patch(
            "app.routers.analysis.run_full_analysis",
            new=AsyncMock(return_value=_MOCK_AI_RESULT),
        ):
            payload = self._build_request_payload(
                text="¡¡¡GANA $5000 DIARIOS!!!"
            )
            response = test_client.post("/api/v1/analyze", json=payload)

        assert response.status_code == 200

    def test_analyze_returns_correct_schema(self, test_client):
        """La respuesta debe contener todos los campos del esquema AnalysisResponse."""
        with patch(
            "app.routers.analysis.run_full_analysis",
            new=AsyncMock(return_value=_MOCK_AI_RESULT),
        ):
            payload = self._build_request_payload(text="¡¡¡GANA $5000!!!")
            response = test_client.post("/api/v1/analyze", json=payload)

        data = response.json()

        required_fields = [
            "post_hash", "category", "confidence", "explanation",
            "text_analysis", "vision_analysis", "multimodal_discrepancies",
            "cached", "analyzed_at",
        ]
        for field in required_fields:
            assert field in data, f"Campo requerido faltante en respuesta: {field}"

    def test_analyze_returns_valid_category(self, test_client):
        """La categoría retornada debe ser una de las categorías válidas."""
        valid_categories = {
            "fraude_financiero", "desinformacion_politica",
            "contenido_enganoso", "manipulacion_emocional", "publicacion_segura",
        }
        with patch(
            "app.routers.analysis.run_full_analysis",
            new=AsyncMock(return_value=_MOCK_AI_RESULT),
        ):
            payload = self._build_request_payload()
            response = test_client.post("/api/v1/analyze", json=payload)

        data = response.json()
        assert data["category"] in valid_categories

    def test_analyze_returns_confidence_in_range(self, test_client):
        """La confianza debe estar en el rango [0.0, 1.0]."""
        with patch(
            "app.routers.analysis.run_full_analysis",
            new=AsyncMock(return_value=_MOCK_AI_RESULT),
        ):
            payload = self._build_request_payload()
            response = test_client.post("/api/v1/analyze", json=payload)

        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_analyze_returns_post_hash_64_chars(self, test_client):
        """El post_hash en la respuesta debe ser un SHA-256 de 64 caracteres."""
        with patch(
            "app.routers.analysis.run_full_analysis",
            new=AsyncMock(return_value=_MOCK_AI_RESULT),
        ):
            payload = self._build_request_payload()
            response = test_client.post("/api/v1/analyze", json=payload)

        data = response.json()
        assert len(data["post_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in data["post_hash"])

    def test_second_request_same_post_returns_cached(self, test_client):
        """El segundo análisis del mismo post debe retornar cached=True."""
        with patch(
            "app.routers.analysis.run_full_analysis",
            new=AsyncMock(return_value=_MOCK_AI_RESULT),
        ) as mock_ai:
            payload = self._build_request_payload(text="texto único para cache test 12345")

            # Primera petición — debe llamar al motor de IA
            response1 = test_client.post("/api/v1/analyze", json=payload)
            assert response1.status_code == 200
            assert response1.json()["cached"] is False

            # Segunda petición con el MISMO contenido — debe usar caché
            response2 = test_client.post("/api/v1/analyze", json=payload)
            assert response2.status_code == 200
            assert response2.json()["cached"] is True

        # El motor de IA solo debió ser llamado una vez (en la primera petición)
        assert mock_ai.call_count == 1

    def test_analyze_returns_422_for_empty_text(self, test_client):
        """Un texto vacío debe retornar HTTP 422 (validación Pydantic)."""
        payload = {
            "post_text": "",
            "author_name": "Test",
            "post_timestamp": "2024-06-15T14:30:00Z",
        }
        response = test_client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_returns_422_for_whitespace_only_text(self, test_client):
        """Un texto solo con espacios debe retornar HTTP 422."""
        payload = {
            "post_text": "   \n\t  ",
            "author_name": "Test",
            "post_timestamp": "2024-06-15T14:30:00Z",
        }
        response = test_client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_returns_422_for_missing_required_fields(self, test_client):
        """Faltar campos requeridos debe retornar HTTP 422."""
        # Falta author_name y post_timestamp
        payload = {"post_text": "solo el texto"}
        response = test_client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 422

    def test_different_texts_produce_different_hashes(self, test_client):
        """Dos publicaciones distintas deben producir hashes distintos."""
        with patch(
            "app.routers.analysis.run_full_analysis",
            new=AsyncMock(return_value=_MOCK_SAFE_RESULT),
        ):
            payload1 = self._build_request_payload(text="Texto número uno completamente único")
            payload2 = self._build_request_payload(text="Texto número dos completamente diferente")

            response1 = test_client.post("/api/v1/analyze", json=payload1)
            response2 = test_client.post("/api/v1/analyze", json=payload2)

        assert response1.json()["post_hash"] != response2.json()["post_hash"]


class TestStatsEndpoint:
    """Tests para el endpoint GET /api/v1/stats."""

    def test_stats_returns_200(self, test_client):
        """El endpoint de estadísticas debe retornar HTTP 200."""
        response = test_client.get("/api/v1/stats")
        assert response.status_code == 200

    def test_stats_returns_correct_structure(self, test_client):
        """Las estadísticas deben tener la estructura esperada."""
        response = test_client.get("/api/v1/stats")
        data = response.json()

        required_fields = [
            "total_records", "total_analyses", "category_distribution",
            "average_confidence", "low_confidence_records",
        ]
        for field in required_fields:
            assert field in data, f"Campo faltante en estadísticas: {field}"

    def test_stats_empty_db_has_zero_records(self, test_client):
        """Una DB vacía debe reportar 0 registros."""
        response = test_client.get("/api/v1/stats")
        data = response.json()

        assert data["total_records"] == 0


class TestHistoryEndpoint:
    """Tests para el endpoint GET /api/v1/history."""

    def test_history_returns_200(self, test_client):
        """El endpoint de historial debe retornar HTTP 200."""
        response = test_client.get("/api/v1/history")
        assert response.status_code == 200

    def test_history_returns_paginated_structure(self, test_client):
        """El historial debe retornar estructura paginada."""
        response = test_client.get("/api/v1/history")
        data = response.json()

        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert "records" in data
        assert isinstance(data["records"], list)

    def test_history_respects_page_size(self, test_client):
        """El parámetro page_size debe limitar los registros retornados."""
        response = test_client.get("/api/v1/history?page_size=5")
        data = response.json()

        assert data["page_size"] == 5
        assert len(data["records"]) <= 5

    def test_history_invalid_page_returns_422(self, test_client):
        """page=0 (inválido) debe retornar 422."""
        response = test_client.get("/api/v1/history?page=0")
        assert response.status_code == 422

"""
test_text_analyzer.py — Tests unitarios para el analizador de texto NLP.

Valida que los patrones léxicos sean detectados correctamente
y que el análisis produzca resultados coherentes.
"""

import asyncio
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ai_engine.text_analyzer import _run_pattern_analysis, analyze_text


class TestPatternAnalysis:
    """Tests para el análisis de patrones léxicos (sin modelo transformer)."""

    def test_detects_financial_fraud_patterns(self, sample_post_text):
        """Debe detectar patrones de fraude financiero en el texto de ejemplo."""
        result = _run_pattern_analysis(sample_post_text)

        assert "fraude_financiero" in result["detected_patterns"]

    def test_detects_emotional_manipulation(self):
        """Debe detectar patrones de manipulación emocional."""
        text = "¡URGENTE! El gobierno oculta esto. Comparte antes de que lo borren."
        result = _run_pattern_analysis(text)

        assert "manipulacion_emocional" in result["detected_patterns"]

    def test_detects_political_disinformation(self, sample_political_text):
        """Debe detectar patrones de desinformación política."""
        result = _run_pattern_analysis(sample_political_text)

        assert "desinformacion_politica" in result["detected_patterns"]

    def test_detects_credibility_signals(self):
        """Debe detectar señales de credibilidad cuando hay fuentes citadas."""
        text = "Según el estudio publicado en Nature, los resultados muestran que..."
        result = _run_pattern_analysis(text)

        assert len(result["credibility_signals"]) > 0

    def test_detects_excessive_exclamations(self):
        """Debe contar exclamaciones excesivas como indicador de manipulación."""
        text = "¡¡¡ATENCIÓN!!! ¡INCREÍBLE! ¡No te lo pierdas!"
        result = _run_pattern_analysis(text)

        assert result["exclamation_count"] > 3
        # Debe aparecer en manipulation_indicators
        exclamation_indicators = [i for i in result["manipulation_indicators"] if "exclamacion" in i]
        assert len(exclamation_indicators) > 0

    def test_safe_content_has_no_patterns(self, sample_safe_text):
        """Contenido seguro no debe tener patrones maliciosos."""
        result = _run_pattern_analysis(sample_safe_text)

        # El texto seguro no debe tener patrones peligrosos
        dangerous_patterns = {"fraude_financiero", "desinformacion_politica", "manipulacion_emocional"}
        detected_dangerous = set(result["detected_patterns"]) & dangerous_patterns
        assert len(detected_dangerous) == 0

    def test_caps_lock_detected_as_manipulation(self):
        """Exceso de mayúsculas debe detectarse como indicador de manipulación."""
        text = "GANA DINERO FÁCIL HOY MISMO HAZLO AHORA MISMO"  # Más del 40% en CAPS
        result = _run_pattern_analysis(text)

        caps_indicators = [i for i in result["manipulation_indicators"] if "mayuscula" in i]
        assert len(caps_indicators) > 0

    def test_empty_patterns_for_normal_text(self):
        """Texto completamente normal no debe tener patrones o indicadores."""
        text = "Hoy aprendí algo nuevo sobre historia. Fue una lectura interesante."
        result = _run_pattern_analysis(text)

        assert len(result["detected_patterns"]) == 0
        # Sin mayúsculas excesivas ni exclamaciones
        assert result["exclamation_count"] == 0


class TestAnalyzeTextAsync:
    """Tests para la función asíncrona principal analyze_text."""

    @pytest.mark.asyncio
    async def test_returns_required_fields(self, sample_post_text):
        """El resultado debe contener todos los campos requeridos por el esquema."""
        result = await analyze_text(sample_post_text)

        required_fields = [
            "detected_patterns",
            "sentiment_score",
            "manipulation_indicators",
            "credibility_signals",
            "text_confidence",
        ]
        for field in required_fields:
            assert field in result, f"Campo requerido faltante: {field}"

    @pytest.mark.asyncio
    async def test_confidence_is_normalized(self, sample_post_text):
        """La confianza debe estar siempre en el rango [0.0, 1.0]."""
        result = await analyze_text(sample_post_text)

        assert 0.0 <= result["text_confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_sentiment_score_is_normalized(self, sample_post_text):
        """El score de sentimiento debe estar en el rango [-1.0, 1.0]."""
        result = await analyze_text(sample_post_text)

        assert -1.0 <= result["sentiment_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_fraud_text_has_higher_confidence_than_safe(
        self, sample_post_text, sample_safe_text
    ):
        """El texto fraudulento debe tener mayor confianza de clasificación que el texto seguro."""
        fraud_result = await analyze_text(sample_post_text)
        safe_result = await analyze_text(sample_safe_text)

        # El fraude debe tener confianza más alta que el contenido seguro
        assert fraud_result["text_confidence"] > safe_result["text_confidence"]

    @pytest.mark.asyncio
    async def test_detected_patterns_is_list(self, sample_post_text):
        """Los patrones detectados deben ser una lista."""
        result = await analyze_text(sample_post_text)

        assert isinstance(result["detected_patterns"], list)

    @pytest.mark.asyncio
    async def test_fraud_detected_in_fraud_text(self, sample_post_text):
        """El análisis debe detectar fraude en el texto de ejemplo fraudulento."""
        result = await analyze_text(sample_post_text)

        assert "fraude_financiero" in result["detected_patterns"]

    @pytest.mark.asyncio
    async def test_unicode_text_does_not_crash(self):
        """El texto con emojis y caracteres especiales no debe causar errores."""
        text = "🚨 ALERTA 🚨 ¡¡¡GANA DINERO HOY!!! 💰💰💰 Precio: $999.99 → ¡Oferta por tiempo limitado!"
        result = await analyze_text(text)

        assert isinstance(result, dict)
        assert "text_confidence" in result

    @pytest.mark.asyncio
    async def test_very_long_text_does_not_crash(self):
        """Textos muy largos (cerca del límite) no deben causar errores."""
        long_text = "Esta es una publicación normal. " * 300  # ~9600 caracteres
        result = await analyze_text(long_text)

        assert isinstance(result, dict)

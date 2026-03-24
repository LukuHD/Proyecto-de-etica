"""
test_hasher.py — Tests unitarios para el módulo de hashing SHA-256.

Valida el comportamiento determinista, consistente y correcto de
la función de generación de huellas digitales.
"""

import pytest
import sys
from pathlib import Path

# Asegurar que el directorio backend está en el path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.utils.hasher import compute_post_hash, compute_text_only_hash


class TestComputePostHash:
    """Tests para la función principal de hashing."""

    def test_returns_64_char_hex_string(self):
        """El hash SHA-256 debe ser siempre una cadena hexadecimal de 64 caracteres."""
        result = compute_post_hash("texto de prueba", author_name="test")
        assert isinstance(result, str)
        assert len(result) == 64
        # Solo caracteres hexadecimales válidos
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_same_input_same_hash(self):
        """El mismo texto + imagen + autor siempre debe producir el mismo hash."""
        text = "publicación de prueba"
        author = "Juan Test"
        image = "dGVzdGltYWdl"  # Base64 de "testimage"

        hash1 = compute_post_hash(text, image, author)
        hash2 = compute_post_hash(text, image, author)

        assert hash1 == hash2

    def test_different_text_different_hash(self):
        """Textos distintos deben producir hashes distintos."""
        hash1 = compute_post_hash("texto uno")
        hash2 = compute_post_hash("texto dos")

        assert hash1 != hash2

    def test_different_authors_different_hash(self):
        """El mismo texto con autores distintos debe producir hashes distintos."""
        text = "mismo texto"
        hash1 = compute_post_hash(text, author_name="autor_a")
        hash2 = compute_post_hash(text, author_name="autor_b")

        assert hash1 != hash2

    def test_case_insensitive_text(self):
        """El texto en mayúsculas/minúsculas debe producir el mismo hash (normalización)."""
        hash1 = compute_post_hash("GANA DINERO")
        hash2 = compute_post_hash("gana dinero")
        hash3 = compute_post_hash("Gana Dinero")

        assert hash1 == hash2 == hash3

    def test_with_and_without_image_different_hash(self):
        """Con y sin imagen debe producir hashes distintos."""
        text = "texto de prueba"
        hash_no_image = compute_post_hash(text)
        hash_with_image = compute_post_hash(text, image_data="algunaimagen")

        assert hash_no_image != hash_with_image

    def test_url_vs_base64_same_content(self):
        """Una URL y una cadena Base64 producen hashes distintos (son datos distintos)."""
        hash_url = compute_post_hash("texto", image_data="https://example.com/img.jpg")
        hash_b64 = compute_post_hash("texto", image_data="dGVzdA==")

        assert hash_url != hash_b64

    def test_none_image_handled_gracefully(self):
        """image_data=None no debe lanzar excepción y debe producir un hash válido."""
        result = compute_post_hash("texto", image_data=None)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_invalid_type_raises_type_error(self):
        """Pasar un no-string como post_text debe lanzar TypeError."""
        with pytest.raises(TypeError):
            compute_post_hash(12345)  # type: ignore

    def test_strips_whitespace_in_text(self):
        """Los espacios sobrantes al inicio/final del texto deben ser ignorados."""
        hash1 = compute_post_hash("  texto con espacios  ")
        hash2 = compute_post_hash("texto con espacios")

        assert hash1 == hash2

    def test_unicode_text_handled_correctly(self):
        """Texto con caracteres UTF-8 (tildes, ñ, emojis) debe hashear correctamente."""
        text_with_unicode = "¡Hola España! ¿Cómo están? 🇲🇽"
        result = compute_post_hash(text_with_unicode)

        assert isinstance(result, str)
        assert len(result) == 64


class TestComputeTextOnlyHash:
    """Tests para la función de hash solo de texto."""

    def test_returns_valid_hash(self):
        """Debe retornar un hash SHA-256 válido de 64 caracteres."""
        result = compute_text_only_hash("texto de prueba")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_consistent_with_post_hash_no_image(self):
        """
        El hash de solo texto NO debe ser igual al hash del post sin imagen,
        porque compute_post_hash incluye el autor y el marcador de imagen.
        """
        text = "texto de prueba"
        text_hash = compute_text_only_hash(text)
        post_hash = compute_post_hash(text)  # Incluye "unknown_author" y "NO_IMAGE_ATTACHED"

        # Son distintos por diseño — el hash del post siempre incluye contexto adicional
        assert text_hash != post_hash

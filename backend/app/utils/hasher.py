"""
hasher.py — Módulo de generación de huellas criptográficas SHA-256.

Responsabilidad única: dada una publicación (texto + representación de imagen),
calcular una firma digital inmutable y exclusiva que permita al sistema de
caché identificar posts ya analizados sin necesidad de comparar el contenido
completo, lo que sería costoso en memoria y tiempo de CPU.

¿Por qué SHA-256?
  • Es resistente a colisiones: la probabilidad de que dos publicaciones distintas
    generen el mismo hash es astronomicamente pequeña (2^-256).
  • Es determinista: el mismo contenido siempre produce el mismo hash.
  • Es rápido para tamaños de datos pequeños (texto + URL/base64 comprimida).
  • Es parte de la biblioteca estándar de Python (hashlib), sin dependencias externas.

Decisión arquitectónica: Combinar texto E imagen en el hash garantiza que un
post con el mismo texto pero diferente imagen (o viceversa) genere una huella
diferente, evitando falsos positivos en la caché.
"""

import hashlib  # Módulo estándar de Python para funciones hash criptográficas
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_post_hash(
    post_text: str,
    image_data: Optional[str] = None,
    author_name: str = "",
) -> str:
    """
    Calcula el hash SHA-256 único de una publicación combinando todos sus
    componentes identificativos.

    Estrategia de composición del hash:
      Se construye una cadena canónica concatenando los componentes con
      delimitadores únicos (chr(0) = byte nulo) que no pueden aparecer en
      texto UTF-8 legible, evitando ataques de extensión de longitud o
      colisiones por concatenación accidental.

      Ejemplo:
        "GANA DINERO\x00juan_perez\x00<base64_de_imagen>"

    Args:
        post_text:   Texto completo de la publicación (normalizado a lowercase
                     para que "FRAUDE" y "fraude" produzcan el mismo hash).
        image_data:  String de imagen en Base64 puro O la URL directa de la imagen.
                     Si es None, se usa un placeholder constante para mantener
                     la estructura del hash independientemente de si hay imagen.
        author_name: Nombre del autor (incluido para diferenciar publicaciones
                     con texto idéntico publicadas por distintos autores).

    Returns:
        String hexadecimal de 64 caracteres representando el hash SHA-256.

    Raises:
        TypeError: Si post_text no es un string (validación defensiva).
    """
    if not isinstance(post_text, str):
        # Defensa programática: Pydantic ya valida esto, pero por si acaso.
        raise TypeError(f"post_text debe ser str, recibido: {type(post_text)}")

    # ── Normalización del texto ─────────────────────────────────────────────
    # Convertimos a minúsculas para que variaciones de casing no generen
    # hashes distintos para el mismo contenido semántico.
    # También eliminamos espacios sobrantes al inicio y final.
    normalized_text = post_text.lower().strip()

    # ── Normalización del autor ─────────────────────────────────────────────
    normalized_author = author_name.lower().strip() if author_name else "unknown_author"

    # ── Manejo de imagen ────────────────────────────────────────────────────
    # Si no hay imagen, usamos un marcador constante para que el hash sea
    # reproducible y no dependa de un valor None (que en Python no es hasheable
    # de la misma forma en diferentes versiones).
    image_component = image_data if image_data else "NO_IMAGE_ATTACHED"

    # ── Construcción de la cadena canónica ──────────────────────────────────
    # chr(0) es el byte nulo — un separador que nunca aparece en texto UTF-8
    # legible, lo que previene ataques de concatenación.
    canonical_string = chr(0).join([
        normalized_text,
        normalized_author,
        image_component,
    ])

    # ── Cálculo del hash ────────────────────────────────────────────────────
    # .encode("utf-8") convierte el string Python a bytes, que es lo que
    # hashlib necesita como entrada.
    # .hexdigest() devuelve el hash como string hexadecimal de 64 caracteres.
    post_hash = hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()

    logger.debug(
        "Hash calculado para publicación de '%s': %s…",
        normalized_author,
        post_hash[:16],  # Solo mostramos los primeros 16 chars en el log por brevedad
    )

    return post_hash


def compute_text_only_hash(text: str) -> str:
    """
    Calcula el hash SHA-256 únicamente del texto.

    Función auxiliar usada en el script de aprendizaje continuo para
    agrupar publicaciones con texto idéntico independientemente de la imagen.

    Args:
        text: Texto a hashear.

    Returns:
        String hexadecimal de 64 caracteres.
    """
    return hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()

"""
Extraccion de caracteristicas por pixel para el clasificador de madurez.

Esta logica se comparte entre entrenamiento e inferencia, asi que debe ser
identica en ambos. Si se cambia, incrementar VERSION_CARACTERISTICAS: los
modelos guardados con otra version se rechazan y deben reentrenarse.
"""
from typing import Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

VERSION_CARACTERISTICAS = 2

# Indice de clase = valor en las mascaras de entrenamiento.
CLASES = ("FONDO", "VERDE", "PINTON", "MADURO", "SOBREMADURO")

# Lado mayor (px) al que se normaliza toda imagen antes de clasificar.
# Las caracteristicas locales (suavizado, textura) dependen de la escala en
# pixeles, por eso entrenamiento e inferencia usan el mismo tamano de trabajo.
LADO_TRABAJO = 320

NOMBRES_CARACTERISTICAS = (
    "r", "g", "b", "tono_sen", "tono_cos", "saturacion", "brillo",
    "g_menos_r", "r_menos_b", "exceso_verde", "croma_r", "croma_g",
    "brillo_relativo", "r_suave", "g_suave", "b_suave", "g_menos_r_suave",
    "textura_5", "textura_9",
    "g_menos_r_ctx", "brillo_ctx", "textura_15", "borde_brillo", "borde_brillo_ctx",
)


def redimensionar_para_trabajo(
    imagen: Image.Image, lado_max: int = LADO_TRABAJO
) -> Tuple[np.ndarray, float]:
    """Devuelve (array RGB uint8, escala) con escala = lado_trabajo / lado_original (<= 1)."""
    imagen = imagen.convert("RGB")
    ancho, alto = imagen.size
    escala = min(1.0, lado_max / float(max(ancho, alto)))
    if escala < 1.0:
        nuevo = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
        imagen = imagen.resize(nuevo, Image.LANCZOS)
    return np.asarray(imagen, dtype=np.uint8), escala


def _desviacion_local(canal: np.ndarray, tam: int) -> np.ndarray:
    media = ndi.uniform_filter(canal, size=tam, mode="reflect")
    media_cuad = ndi.uniform_filter(canal * canal, size=tam, mode="reflect")
    return np.sqrt(np.maximum(media_cuad - media * media, 0.0))


def extraer_caracteristicas(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: array (alto, ancho, 3) uint8.
    Devuelve una matriz (alto*ancho, n_caracteristicas) float32.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Se esperaba un array RGB de forma (alto, ancho, 3).")
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32) / 255.0
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    tono = hsv[..., 0] * 2.0 * np.pi  # el tono es circular: el rojo esta en 0 y en 2*pi
    sat, val = hsv[..., 1], hsv[..., 2]

    borde = ndi.gaussian_gradient_magnitude(val, sigma=1.0)
    suma = r + g + b + 1e-6
    r_s, g_s, b_s = (ndi.gaussian_filter(c, sigma=1.5) for c in (r, g, b))

    capas = [
        r, g, b,
        np.sin(tono), np.cos(tono), sat, val,
        g - r, r - b, 2.0 * g - r - b,
        r / suma, g / suma,
        val / (val.mean() + 1e-6),  # brillo relativo: robusto a la iluminacion global
        r_s, g_s, b_s, g_s - r_s,
        _desviacion_local(val, 5), _desviacion_local(val, 9),
        # Contexto a mayor escala y bordes: ayudan a separar frutos (lisos, contorno
        # nitido) de hojas (nervaduras, bordes irregulares).
        ndi.gaussian_filter(g - r, sigma=4.0), ndi.gaussian_filter(val, sigma=4.0),
        _desviacion_local(val, 15),
        borde, ndi.gaussian_filter(borde, sigma=3.0),
    ]
    return np.stack(capas, axis=-1).reshape(-1, len(capas)).astype(np.float32)

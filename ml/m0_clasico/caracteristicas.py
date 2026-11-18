"""
Características de M0 sobre recortes de cerezas (interfaz usada por ml/m0_clasico/entrenar.py).

El descriptor (color HSV/Lab + LBP + HOG, 442 dimensiones) está especificado e implementado en
ml/m0_clasico/descriptor.py (CPU, numpy) y ml/m0_clasico/descriptor_cuda.py (GPU, CUDA). Ambas
versiones ejecutan el mismo algoritmo; la elección de dispositivo solo cambia el tiempo.

Historial: la versión 1 usaba scikit-image (rgb2hsv, local_binary_pattern, hog); se reemplazó por la
versión 2, de especificación propia, para poder implementarla idéntica en CUDA (scikit-image no
publica el código de LBP/HOG en Python y su algoritmo no se puede replicar con garantía).
"""

from typing import Iterable

import numpy as np

from .descriptor import DIMENSION, VERSION_DESCRIPTOR, descriptor_lote_numpy, preparar_recorte

__all__ = ["DIMENSION", "VERSION_DESCRIPTOR", "caracteristicas_recorte", "descriptor_lote", "extraer_lote", "preparar_recorte", "recortar"]


def recortar(imagen: np.ndarray, caja_xyxy, margen: float = 0.10) -> np.ndarray:
    """Recorte de la caja con un margen relativo, limitado a la imagen."""
    alto, ancho = imagen.shape[:2]
    x0, y0, x1, y1 = caja_xyxy
    mx, my = (x1 - x0) * margen, (y1 - y0) * margen
    x0, y0 = int(max(0, np.floor(x0 - mx))), int(max(0, np.floor(y0 - my)))
    x1, y1 = int(min(ancho, np.ceil(x1 + mx))), int(min(alto, np.ceil(y1 + my)))
    return imagen[y0:y1, x0:x1]


def descriptor_lote(imgs48: np.ndarray, dispositivo: str = "cpu", tiempos: dict = None) -> np.ndarray:
    """imgs48 uint8 (N, 48, 48, 3) ya reescalados -> float32 (N, 442)."""
    if dispositivo == "gpu":
        from .descriptor_cuda import descriptor_lote_cuda

        return descriptor_lote_cuda(imgs48, tiempos=tiempos).astype(np.float32)
    if dispositivo != "cpu":
        raise ValueError(f"dispositivo desconocido: {dispositivo!r} (cpu o gpu)")
    return descriptor_lote_numpy(imgs48).astype(np.float32)


def caracteristicas_recorte(recorte_rgb: np.ndarray) -> np.ndarray:
    return descriptor_lote(preparar_recorte(recorte_rgb)[None])[0]


def extraer_lote(recortes: Iterable[np.ndarray], dispositivo: str = "cpu") -> np.ndarray:
    recortes = list(recortes)
    if not recortes:
        return np.zeros((0, DIMENSION), np.float32)
    return descriptor_lote(np.stack([preparar_recorte(r) for r in recortes]), dispositivo)

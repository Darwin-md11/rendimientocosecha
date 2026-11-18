"""
Descriptor de M0 (línea base clásica): color + textura de un recorte de cereza.

Bloques (todos sobre el recorte reescalado a LADO x LADO px):
  - Histograma HSV: 16 bins por canal (48)
  - Histograma Lab: 16 bins por canal (48)
  - Media y desviación de H, S, V, L, a, b (12)
  - LBP uniforme (P=8, R=1) sobre la escala de grises: 10 bins (10)
  - HOG: 9 orientaciones, celdas de 12 px, bloques 2x2 (324)
Total: 442 dimensiones.

Esta función es deliberadamente una pieza aislada y versionada: es el paso costoso de M0
(un recorte por cereza) y el candidato natural para acelerarse con CUDA o repartirse con MPI.
Cualquier implementación alternativa debe dar el mismo vector (ver pruebas).
"""

from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab
from skimage.feature import hog, local_binary_pattern

VERSION_DESCRIPTOR = 1
LADO = 48
BINS = 16
NOMBRES_BLOQUES: List[Tuple[str, int]] = [("hsv_hist", 48), ("lab_hist", 48), ("media_desv", 12), ("lbp", 10), ("hog", 324)]
DIMENSION = sum(n for _, n in NOMBRES_BLOQUES)

# Rangos de cada canal para los histogramas
_RANGO_HSV = [(0.0, 1.0)] * 3
_RANGO_LAB = [(0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0)]


def _histogramas(canales: np.ndarray, rangos) -> np.ndarray:
    n = canales.shape[0] * canales.shape[1]
    return np.concatenate([np.histogram(canales[..., c], bins=BINS, range=rangos[c])[0] / n for c in range(3)]).astype(np.float32)


def caracteristicas_recorte(recorte_rgb: np.ndarray) -> np.ndarray:
    """recorte_rgb: uint8 (alto, ancho, 3). Devuelve un vector float32 de DIMENSION."""
    img = Image.fromarray(np.ascontiguousarray(recorte_rgb, dtype=np.uint8)).resize((LADO, LADO), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    hsv = rgb2hsv(rgb)
    lab = rgb2lab(rgb)
    gris = rgb @ np.array([0.299, 0.587, 0.114])
    lbp = local_binary_pattern((gris * 255).astype(np.uint8), P=8, R=1, method="uniform")
    hist_lbp = np.histogram(lbp, bins=10, range=(0, 10))[0] / lbp.size
    vec = np.concatenate(
        [
            _histogramas(hsv, _RANGO_HSV),
            _histogramas(lab, _RANGO_LAB),
            np.concatenate([hsv.reshape(-1, 3).mean(0), hsv.reshape(-1, 3).std(0), lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)]),
            hist_lbp,
            hog(gris, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(2, 2), feature_vector=True),
        ]
    ).astype(np.float32)
    assert vec.shape == (DIMENSION,)
    return vec


def recortar(imagen: np.ndarray, caja_xyxy, margen: float = 0.10) -> np.ndarray:
    """Recorte de la caja con un margen relativo, limitado a la imagen."""
    alto, ancho = imagen.shape[:2]
    x0, y0, x1, y1 = caja_xyxy
    mx, my = (x1 - x0) * margen, (y1 - y0) * margen
    x0, y0 = int(max(0, np.floor(x0 - mx))), int(max(0, np.floor(y0 - my)))
    x1, y1 = int(min(ancho, np.ceil(x1 + mx))), int(min(alto, np.ceil(y1 + my)))
    return imagen[y0:y1, x0:x1]


def extraer_lote(recortes: Iterable[np.ndarray]) -> np.ndarray:
    recortes = list(recortes)
    return np.stack([caracteristicas_recorte(r) for r in recortes]) if recortes else np.zeros((0, DIMENSION), np.float32)

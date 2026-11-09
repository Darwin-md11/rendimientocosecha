"""
ML Fruit Detector Adapter

Detector de cerezas de cafe basado en un clasificador de pixeles entrenado
(scikit-learn). Pasos:
  1. Reduce la imagen al tamano de trabajo.
  2. Clasifica cada pixel: FONDO / VERDE / PINTON / MADURO / SOBREMADURO.
  3. Limpia la mascara de primer plano (quita ruido y huecos pequenos).
  4. Separa frutos que se tocan con watershed sobre la transformada de distancia.
  5. Cada region es una deteccion: estado = clase con mayor probabilidad media
     en la region, confianza = esa probabilidad media.
"""
from typing import List

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

from ....domain.models.coffee_cherry import BoundingBox, DeteccionFruto, EstadoMadurez
from ....domain.ports.fruit_detector_port import FruitDetectorPort
from ...ml.features import CLASES, LADO_TRABAJO, redimensionar_para_trabajo
from ...ml.modelo import ModeloMadurezPixel

_ESTADOS_POR_CLASE = {
    1: EstadoMadurez.VERDE,
    2: EstadoMadurez.PINTON,
    3: EstadoMadurez.MADURO,
    4: EstadoMadurez.SOBREMADURO,
}


def _componentes_mayores(mascara: np.ndarray, area_max_descartada: int) -> np.ndarray:
    """Conserva solo los componentes conectados con mas de `area_max_descartada` pixeles."""
    etiquetas, _ = ndi.label(mascara)
    areas = np.bincount(etiquetas.ravel())
    conservar = areas > area_max_descartada
    conservar[0] = False
    return conservar[etiquetas]


class MLFruitDetectorAdapter(FruitDetectorPort):
    def __init__(
        self,
        modelo: ModeloMadurezPixel,
        lado_max: int = LADO_TRABAJO,
        area_min_px: int = 90,
        radio_min_px: float = 2.5,
        distancia_min_picos: int = 4,
        excentricidad_max: float = 0.9,
        solidez_min: float = 0.8,
    ):
        """
        area_min_px / radio_min_px / distancia_min_picos estan en pixeles de la
        imagen de trabajo (lado mayor = lado_max).
        excentricidad_max / solidez_min descartan regiones alargadas o
        irregulares (hojas) que no se parecen a una cereza.
        """
        self.modelo = modelo
        self.lado_max = lado_max
        self.area_min_px = area_min_px
        self.radio_min_px = radio_min_px
        self.distancia_min_picos = distancia_min_picos
        self.excentricidad_max = excentricidad_max
        self.solidez_min = solidez_min

    def detectar_frutos(self, imagen: Image.Image, umbral_confianza: float = 0.40) -> List[DeteccionFruto]:
        ancho_orig, alto_orig = imagen.size
        rgb, escala = redimensionar_para_trabajo(imagen, self.lado_max)

        proba = self.modelo.predecir_proba(rgb)
        etiquetas = proba.argmax(axis=-1)

        primer_plano = etiquetas > 0
        primer_plano = ndi.binary_opening(primer_plano, structure=np.ones((3, 3), dtype=bool))
        primer_plano = _componentes_mayores(primer_plano, self.area_min_px - 1)
        huecos_pequenos = ~primer_plano & ~_componentes_mayores(~primer_plano, 20)
        primer_plano = primer_plano | huecos_pequenos
        if not primer_plano.any():
            return []

        distancia = ndi.gaussian_filter(ndi.distance_transform_edt(primer_plano), 1.0)
        componentes, _ = ndi.label(primer_plano)
        picos = peak_local_max(
            distancia,
            min_distance=self.distancia_min_picos,
            threshold_abs=self.radio_min_px,
            labels=componentes,
            exclude_border=False,
        )
        if len(picos) == 0:
            return []
        marcadores = np.zeros(componentes.shape, dtype=np.int32)
        marcadores[tuple(picos.T)] = np.arange(1, len(picos) + 1)
        regiones = watershed(-distancia, marcadores, mask=primer_plano)

        detecciones: List[DeteccionFruto] = []
        for id_region, corte in enumerate(ndi.find_objects(regiones), start=1):
            if corte is None:
                continue
            mascara = regiones[corte] == id_region
            if int(mascara.sum()) < self.area_min_px:
                continue
            forma = regionprops(mascara.astype(np.uint8))[0]
            if forma.eccentricity > self.excentricidad_max or forma.solidity < self.solidez_min:
                continue

            medias = proba[corte][mascara][:, 1:].mean(axis=0)  # solo los 4 estados de fruto
            clase = int(medias.argmax()) + 1
            confianza = float(medias[clase - 1])
            if confianza < umbral_confianza:
                continue

            y0, y1 = corte[0].start, corte[0].stop
            x0, x1 = corte[1].start, corte[1].stop
            x_min = min(max(0.0, x0 / escala), ancho_orig - 1.0)
            y_min = min(max(0.0, y0 / escala), alto_orig - 1.0)
            x_max = min(float(ancho_orig), max(x1 / escala, x_min + 1.0))
            y_max = min(float(alto_orig), max(y1 / escala, y_min + 1.0))
            detecciones.append(
                DeteccionFruto(
                    id_deteccion=f"F{len(detecciones) + 1:03d}",
                    estado=_ESTADOS_POR_CLASE[clase],
                    confianza=min(1.0, confianza),
                    bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                )
            )
        return detecciones

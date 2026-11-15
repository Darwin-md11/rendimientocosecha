"""
Mock Fruit Detector Adapter

Detector SIMULADO: ignora el contenido de la imagen y genera cajas aleatorias.
Sirve solo para pruebas de la arquitectura; para estimaciones reales usar
MLFruitDetectorAdapter.
"""
from typing import List

import numpy as np
from PIL import Image

from ....domain.models.coffee_cherry import BoundingBox, DeteccionFruto, EstadoMadurez
from ....domain.models.modelo_info import FamiliaModelo, ModeloInfo
from ....domain.ports.fruit_detector_port import FruitDetectorPort


class MockFruitDetectorAdapter(FruitDetectorPort):
    def __init__(self, num_simulated_fruits: int = 15, semilla: int = 42):
        self.num_simulated_fruits = num_simulated_fruits
        self.semilla = semilla

    def info(self) -> ModeloInfo:
        return ModeloInfo(
            nombre="Detector simulado (cajas aleatorias)",
            familia=FamiliaModelo.MOCK,
            version="1",
            datos_entrenamiento="ninguno: ignora el contenido de la imagen",
            validado_con_datos_reales=False,
        )

    def detectar_frutos(self, imagen: Image.Image, umbral_confianza: float = 0.40) -> List[DeteccionFruto]:
        width, height = imagen.size
        detecciones = []
        # Generador local: no altera el estado global de numpy.random (antes se
        # llamaba np.random.seed(42), lo que afectaba a todo el proceso).
        rng = np.random.default_rng(self.semilla)
        estados = [EstadoMadurez.VERDE, EstadoMadurez.PINTON, EstadoMadurez.MADURO, EstadoMadurez.SOBREMADURO]
        probabilidades = [0.25, 0.20, 0.45, 0.10]

        for i in range(self.num_simulated_fruits):
            conf = float(rng.uniform(0.35, 0.98))
            if conf < umbral_confianza:
                continue
            est = estados[int(rng.choice(len(estados), p=probabilidades))]
            w_box = min(float(rng.uniform(30, 60)), width - 2.0)
            h_box = min(float(rng.uniform(30, 60)), height - 2.0)
            if w_box <= 0 or h_box <= 0:
                continue  # imagen demasiado pequena para alojar una caja
            # La caja siempre queda dentro de la imagen (antes podia salirse en imagenes chicas).
            x_min = float(rng.uniform(0, max(0.0, width - w_box - 1)))
            y_min = float(rng.uniform(0, max(0.0, height - h_box - 1)))

            bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_min + w_box, y_max=y_min + h_box)
            detecciones.append(
                DeteccionFruto(id_deteccion=f"M{len(detecciones) + 1:03d}", estado=est, confianza=conf, bbox=bbox)
            )

        return detecciones

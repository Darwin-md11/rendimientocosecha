"""
Mock Fruit Detector Adapter
"""
from typing import List
import uuid
import numpy as np
from PIL import Image
from ....domain.ports.fruit_detector_port import FruitDetectorPort
from ....domain.models.coffee_cherry import DeteccionFruto, EstadoMadurez, BoundingBox

class MockFruitDetectorAdapter(FruitDetectorPort):
    def __init__(self, num_simulated_fruits: int = 15):
        self.num_simulated_fruits = num_simulated_fruits

    def detectar_frutos(self, imagen: Image.Image, umbral_confianza: float = 0.40) -> List[DeteccionFruto]:
        width, height = imagen.size
        detecciones = []
        np.random.seed(42)
        estados = [EstadoMadurez.VERDE, EstadoMadurez.PINTON, EstadoMadurez.MADURO, EstadoMadurez.SOBREMADURO]
        probabilidades = [0.25, 0.20, 0.45, 0.10]
        
        for i in range(self.num_simulated_fruits):
            conf = float(np.random.uniform(0.35, 0.98))
            if conf < umbral_confianza:
                continue
            idx = np.random.choice(len(estados), p=probabilidades)
            est = estados[idx]
            w_box, h_box = float(np.random.uniform(30, 60)), float(np.random.uniform(30, 60))
            x_min = float(np.random.uniform(10, max(11, width - w_box - 10)))
            y_min = float(np.random.uniform(10, max(11, height - h_box - 10)))
            
            bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_min + w_box, y_max=y_min + h_box)
            detecciones.append(DeteccionFruto(id_deteccion=str(uuid.uuid4())[:8], estado=est, confianza=conf, bbox=bbox))
            
        return detecciones

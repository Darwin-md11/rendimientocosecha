"""
Outbound Port: FruitDetectorPort
"""
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
from ..models.coffee_cherry import DeteccionFruto

class FruitDetectorPort(ABC):
    @abstractmethod
    def detectar_frutos(self, imagen: Image.Image, umbral_confianza: float = 0.40) -> List[DeteccionFruto]:
        pass

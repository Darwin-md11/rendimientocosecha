"""
Outbound Port: FruitDetectorPort

Contrato de todo detector (sintético, mock, YOLO, dos etapas...):
  - detectar_frutos: cajas en coordenadas de la imagen recibida.
  - info: procedencia del modelo (ModeloInfo). Obligatoria: la UI, la CLI y
    los reportes la usan para advertir cuando el modelo no está validado.
"""
from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from ..models.coffee_cherry import DeteccionFruto
from ..models.modelo_info import ModeloInfo


class FruitDetectorPort(ABC):
    @abstractmethod
    def detectar_frutos(self, imagen: Image.Image, umbral_confianza: float = 0.40) -> List[DeteccionFruto]:
        pass

    @abstractmethod
    def info(self) -> ModeloInfo:
        pass

    @property
    def es_simulado(self) -> bool:
        return self.info().es_simulado

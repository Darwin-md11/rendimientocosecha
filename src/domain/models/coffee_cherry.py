"""
Domain Model: EstadoMadurez, BoundingBox, DeteccionFruto
"""
from enum import Enum
from dataclasses import dataclass

class EstadoMadurez(str, Enum):
    VERDE = "VERDE"
    PINTON = "PINTON"
    MADURO = "MADURO"
    SOBREMADURO = "SOBREMADURO"

    @classmethod
    def from_string(cls, value: str) -> "EstadoMadurez":
        val_upper = value.strip().upper()
        for member in cls:
            if member.value == val_upper:
                return member
        raise ValueError(f"Estado de madurez invalido: {value}")

@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("Las coordenadas de la caja no pueden ser negativas.")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Las coordenadas maximas deben ser mayores que las minimas.")

@dataclass(frozen=True)
class DeteccionFruto:
    id_deteccion: str
    estado: EstadoMadurez
    confianza: float
    bbox: BoundingBox

    def __post_init__(self):
        if not (0.0 <= self.confianza <= 1.0):
            raise ValueError(f"La confianza debe estar entre 0.0 y 1.0, se recibio: {self.confianza}")

"""
Domain Value Object: ParametrosAgronomicos
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class ParametrosAgronomicos:
    peso_medio_fruto_g: float = 1.8
    factor_oclusion: float = 1.25
    plantas_muestra: int = 1
    plantas_totales_lote: int = 2500

    def __post_init__(self):
        if self.peso_medio_fruto_g <= 0:
            raise ValueError("El peso medio del fruto debe ser mayor a 0.")
        if self.factor_oclusion < 1.0:
            raise ValueError("El factor de oclusion debe ser >= 1.0.")
        if self.plantas_muestra <= 0:
            raise ValueError("Las plantas de muestra deben ser al menos 1.")
        if self.plantas_totales_lote < self.plantas_muestra:
            raise ValueError("Plantas totales debe ser mayor o igual a las de la muestra.")

    @property
    def peso_medio_fruto_kg(self) -> float:
        return self.peso_medio_fruto_g / 1000.0

"""
Domain Entity: ReporteRendimiento
"""
from dataclasses import dataclass
from typing import Dict, List
from .coffee_cherry import DeteccionFruto, EstadoMadurez
from .agronomic_params import ParametrosAgronomicos

@dataclass(frozen=True)
class ReporteRendimiento:
    total_frutos: int
    conteo_por_estado: Dict[EstadoMadurez, int]
    rendimiento_estimado_kg: float
    parametros: ParametrosAgronomicos
    detecciones: List[DeteccionFruto]

    @property
    def porcentaje_madurez(self) -> float:
        if self.total_frutos == 0:
            return 0.0
        maduros = self.conteo_por_estado.get(EstadoMadurez.MADURO, 0)
        return round((maduros / self.total_frutos) * 100, 2)

"""
Domain Entity: ReporteRendimiento
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from .agronomic_params import ParametrosAgronomicos
from .coffee_cherry import DeteccionFruto, EstadoMadurez


@dataclass(frozen=True)
class ReporteRendimiento:
    total_frutos: int
    conteo_por_estado: Dict[EstadoMadurez, int]
    rendimiento_estimado_kg: float
    parametros: ParametrosAgronomicos
    detecciones: List[DeteccionFruto]
    rendimiento_pergamino_kg: Optional[float] = None
    frutos_cosechables: Optional[int] = None  # cerezas en los estados "listos para cosechar"
    rendimiento_cosechable_kg: Optional[float] = None
    rendimiento_cosechable_pergamino_kg: Optional[float] = None

    @property
    def porcentaje_madurez(self) -> float:
        if self.total_frutos == 0:
            return 0.0
        maduros = self.conteo_por_estado.get(EstadoMadurez.MADURO, 0)
        return round((maduros / self.total_frutos) * 100, 2)

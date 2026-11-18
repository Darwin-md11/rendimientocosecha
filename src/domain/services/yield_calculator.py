"""
Domain Service: CalculadoraRendimientoService (RB-002)
"""
from typing import List, Optional

from ..models.agronomic_params import ParametrosAgronomicos
from ..models.coffee_cherry import DeteccionFruto, EstadoMadurez
from ..models.yield_report import ReporteRendimiento


class CalculadoraRendimientoService:
    @staticmethod
    def calcular(detecciones: List[DeteccionFruto], parametros: ParametrosAgronomicos,
                 factor_cereza_pergamino: Optional[float] = None) -> ReporteRendimiento:
        """kg de café cereza = frutos × factor_oclusión × peso_medio_kg × (plantas_lote / plantas_muestra);
        kg de pergamino = kg cereza × factor_cereza_pergamino (solo si se da el factor)."""
        total_frutos = len(detecciones)
        conteo = {estado: 0 for estado in EstadoMadurez}
        for det in detecciones:
            conteo[det.estado] = conteo.get(det.estado, 0) + 1

        if total_frutos == 0:
            rendimiento_kg = 0.0
        else:
            factor_escalado = parametros.plantas_totales_lote / parametros.plantas_muestra
            rendimiento_kg = (
                total_frutos *
                parametros.factor_oclusion *
                parametros.peso_medio_fruto_kg *
                factor_escalado
            )

        return ReporteRendimiento(
            total_frutos=total_frutos,
            conteo_por_estado=conteo,
            rendimiento_estimado_kg=round(rendimiento_kg, 3),
            parametros=parametros,
            detecciones=detecciones,
            rendimiento_pergamino_kg=None if factor_cereza_pergamino is None else round(rendimiento_kg * factor_cereza_pergamino, 3),
        )

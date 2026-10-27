"""
Domain Service: CalculadoraRendimientoService (RB-002)
"""
from typing import List
from ..models.coffee_cherry import DeteccionFruto, EstadoMadurez
from ..models.agronomic_params import ParametrosAgronomicos
from ..models.yield_report import ReporteRendimiento

class CalculadoraRendimientoService:
    @staticmethod
    def calcular(detecciones: List[DeteccionFruto], parametros: ParametrosAgronomicos) -> ReporteRendimiento:
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
            detecciones=detecciones
        )

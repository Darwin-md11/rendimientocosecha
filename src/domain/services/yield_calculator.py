"""
Domain Service: CalculadoraRendimientoService (RB-002)
"""
from typing import Iterable, List, Optional

from ..models.agronomic_params import ParametrosAgronomicos
from ..models.coffee_cherry import DeteccionFruto, EstadoMadurez
from ..models.yield_report import ReporteRendimiento


class CalculadoraRendimientoService:
    @staticmethod
    def calcular(detecciones: List[DeteccionFruto], parametros: ParametrosAgronomicos,
                 factor_cereza_pergamino: Optional[float] = None,
                 estados_cosechables: Optional[Iterable[EstadoMadurez]] = None) -> ReporteRendimiento:
        """kg de café cereza = frutos × factor_oclusión × peso_medio_kg × (plantas_lote / plantas_muestra);
        kg de pergamino = kg cereza × factor_cereza_pergamino (solo si se da el factor).
        Con `estados_cosechables`, lo mismo solo para las cerezas en esos estados ("listo para cosechar")."""
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

        def _kg(n: int) -> float:
            return n * parametros.factor_oclusion * parametros.peso_medio_fruto_kg * (parametros.plantas_totales_lote / parametros.plantas_muestra)

        def _perg(kg: float) -> Optional[float]:
            return None if factor_cereza_pergamino is None else round(kg * factor_cereza_pergamino, 3)

        cosechables = kg_cos = None
        if estados_cosechables is not None:
            estados = set(estados_cosechables)
            cosechables = sum(1 for d in detecciones if d.estado in estados)
            kg_cos = _kg(cosechables)

        return ReporteRendimiento(
            total_frutos=total_frutos,
            conteo_por_estado=conteo,
            rendimiento_estimado_kg=round(rendimiento_kg, 3),
            parametros=parametros,
            detecciones=detecciones,
            rendimiento_pergamino_kg=_perg(rendimiento_kg),
            frutos_cosechables=cosechables,
            rendimiento_cosechable_kg=None if kg_cos is None else round(kg_cos, 3),
            rendimiento_cosechable_pergamino_kg=None if kg_cos is None else _perg(kg_cos),
        )

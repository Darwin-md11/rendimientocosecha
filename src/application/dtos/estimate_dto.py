"""
Application DTOs
"""
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class SolicitudEstimacionDTO:
    imagen_bytes: bytes
    peso_medio_fruto_g: float = 1.8
    factor_oclusion: float = 1.25
    plantas_muestra: int = 1
    plantas_totales_lote: int = 2500
    umbral_confianza: float = 0.40

@dataclass
class RespuestaEstimacionDTO:
    total_frutos: int
    conteo_por_estado: Dict[str, int]
    rendimiento_estimado_kg: float
    porcentaje_madurez: float
    imagen_anotada_bytes: bytes
    detecciones_detalle: List[Dict[str, Any]]

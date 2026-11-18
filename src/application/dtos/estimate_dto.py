"""
Application DTOs
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SolicitudEstimacionDTO:
    imagen_bytes: bytes
    # None = usar el valor del catálogo agronómico (configs/agronomia.yaml), con su fuente
    peso_medio_fruto_g: Optional[float] = None
    factor_oclusion: Optional[float] = None
    plantas_muestra: Optional[int] = None
    plantas_totales_lote: Optional[int] = None
    umbral_confianza: float = 0.40

@dataclass
class RespuestaEstimacionDTO:
    total_frutos: int
    conteo_por_estado: Dict[str, int]
    rendimiento_estimado_kg: float
    porcentaje_madurez: float
    imagen_anotada_bytes: bytes
    detecciones_detalle: List[Dict[str, Any]]
    modelo: Dict[str, Any]  # ModeloInfo.como_dict(): qué modelo produjo este resultado
    advertencias: List[str]  # del modelo y del rendimiento; vacía solo si todo está validado y con fuente
    supuestos: List[Dict[str, Any]] = field(default_factory=list)  # factores usados, con valor, unidad y fuente
    unidad_rendimiento: str = "kg de café cereza"
    rendimiento_pergamino_kg: Optional[float] = None  # solo si el factor cereza→pergamino tiene valor y fuente
    # "Listo para cosechar": solo las cerezas en estados_cosechables (p. ej. MADURO y SOBREMADURO)
    estados_cosechables: List[str] = field(default_factory=list)
    fuente_estados_cosechables: Optional[str] = None
    frutos_cosechables: Optional[int] = None
    rendimiento_cosechable_kg: Optional[float] = None
    rendimiento_cosechable_pergamino_kg: Optional[float] = None
    version_parametros: Optional[str] = None

    @property
    def es_resultado_validado(self) -> bool:
        return not self.advertencias

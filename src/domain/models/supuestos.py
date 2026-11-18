"""
Domain Value Objects: supuestos agronómicos con su procedencia.

Cada factor que convierte conteo en rendimiento lleva valor, unidad y FUENTE. Un factor de literatura
sin fuente documental se marca "TODO(fuente)" y genera una advertencia en el reporte.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

TODO_FUENTE = "TODO(fuente)"
INGRESADO_POR_USUARIO = "ingresado por el usuario"
LITERATURA, CALIBRACION, DATO = "literatura", "calibracion", "dato"
TIPOS = (LITERATURA, CALIBRACION, DATO)
CLAVES_PARAMETROS = ("peso_medio_fruto_g", "factor_oclusion", "plantas_muestra", "plantas_totales_lote")


@dataclass(frozen=True)
class Supuesto:
    clave: str
    descripcion: str
    tipo: str
    valor: Optional[float]
    unidad: str
    fuente: Optional[str] = None
    incertidumbre: Optional[str] = None
    estado: Optional[str] = None  # p. ej. "referencia bibliográfica, pendiente de calibración local"

    def __post_init__(self):
        if self.tipo not in TIPOS:
            raise ValueError(f"{self.clave}: tipo debe ser uno de {TIPOS}.")

    @property
    def tiene_fuente(self) -> bool:
        """Literatura y calibración solo cuentan con fuente documental o de calibración; un dato, también si lo ingresó el usuario."""
        if not self.fuente:
            return False
        return self.tipo == DATO or self.fuente != INGRESADO_POR_USUARIO

    @property
    def fuente_mostrada(self) -> str:
        if self.tiene_fuente:
            return self.fuente
        if self.fuente == INGRESADO_POR_USUARIO:
            return f"{INGRESADO_POR_USUARIO} — {TODO_FUENTE}"
        return TODO_FUENTE

    def usado_con(self, valor: Optional[float]) -> "Supuesto":
        """El supuesto tal como se usó: si el usuario envió un valor, se registra como ingresado por él
        (para la literatura, solo si difiere del valor documentado)."""
        if valor is None:
            return self
        if self.tipo != DATO and self.valor is not None and float(valor) == float(self.valor):
            return self
        return replace(self, valor=float(valor), fuente=INGRESADO_POR_USUARIO, incertidumbre=None, estado=None)

    def como_dict(self) -> dict:
        return {
            "clave": self.clave,
            "descripcion": self.descripcion,
            "tipo": self.tipo,
            "valor": self.valor,
            "unidad": self.unidad,
            "fuente": self.fuente_mostrada,
            "tiene_fuente": self.tiene_fuente,
            "incertidumbre": self.incertidumbre,
            "estado": self.estado,
        }


@dataclass(frozen=True)
class CatalogoSupuestos:
    factores: Dict[str, Supuesto]
    salida: str = "kg de café cereza"
    muestreo_definido: bool = False
    supuesto_muestreo: str = ""
    validado_con_cosecha_real: bool = False
    origen: str = "valores por defecto del dominio (sin fuentes)"
    version_parametros: Optional[str] = None
    calibrado_localmente: bool = False
    simplificaciones: Tuple[str, ...] = ()
    estados_cosechables: Tuple[str, ...] = ()  # nombres de EstadoMadurez que cuentan como "listo para cosechar"
    fuente_estados_cosechables: Optional[str] = None

    def __post_init__(self):
        faltan = set(CLAVES_PARAMETROS) - set(self.factores)
        if faltan:
            raise ValueError(f"Faltan factores en el catálogo: {sorted(faltan)}")

    def valor(self, clave: str):
        return self.factores[clave].valor

    @staticmethod
    def desde_valores(valores: Dict[str, float], origen: str = "valores por defecto del dominio (sin fuentes)") -> "CatalogoSupuestos":
        """Catálogo mínimo con los valores dados y SIN fuentes (todas las advertencias activas)."""
        tipos = {
            "peso_medio_fruto_g": (LITERATURA, "Peso medio de una cereza de café", "g"),
            "factor_oclusion": (CALIBRACION, "Cerezas reales por cereza visible", "adimensional"),
            "plantas_muestra": (DATO, "Plantas que cubren las fotos analizadas", "plantas"),
            "plantas_totales_lote": (DATO, "Plantas del lote", "plantas"),
        }
        factores = {k: Supuesto(k, d, t, valores[k], u) for k, (t, d, u) in tipos.items()}
        return CatalogoSupuestos(factores, supuesto_muestreo="Unidad de muestreo sin definir.", origen=origen)

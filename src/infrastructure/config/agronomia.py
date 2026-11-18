"""Carga y valida configs/agronomia.yaml como CatalogoSupuestos (única fuente de los valores por defecto)."""

from pathlib import Path
from typing import Optional

import yaml

from ...domain.exceptions.domain_exceptions import ParametrosInvalidosException
from ...domain.models.coffee_cherry import EstadoMadurez
from ...domain.models.supuestos import CLAVES_PARAMETROS, CatalogoSupuestos, Supuesto

RUTA_AGRONOMIA = Path(__file__).resolve().parents[3] / "configs" / "agronomia.yaml"
_CAMPOS = ("descripcion", "tipo", "valor", "unidad")


def _estados(nombres, ruta: Path):
    validos = {e.value for e in EstadoMadurez}
    for n in nombres:
        if n not in validos:
            raise ParametrosInvalidosException(f"{ruta.name}: estado cosechable desconocido {n!r} (use {sorted(validos)})")
        yield n


def cargar_catalogo(ruta: Optional[Path] = None) -> CatalogoSupuestos:
    ruta = Path(ruta or RUTA_AGRONOMIA)
    try:
        cfg = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ParametrosInvalidosException(f"No se pudo leer {ruta}: {e}") from e
    factores = {}
    for clave, f in (cfg.get("factores") or {}).items():
        faltan = [c for c in _CAMPOS if c not in f]
        if faltan:
            raise ParametrosInvalidosException(f"{ruta.name}: al factor '{clave}' le faltan {faltan}")
        if f["valor"] is not None and float(f["valor"]) <= 0:
            raise ParametrosInvalidosException(f"{ruta.name}: '{clave}' debe ser > 0")
        if clave in CLAVES_PARAMETROS and f["valor"] is None:
            raise ParametrosInvalidosException(f"{ruta.name}: '{clave}' necesita un valor (la fórmula lo usa)")
        try:
            factores[clave] = Supuesto(
                clave,
                f["descripcion"],
                f["tipo"],
                None if f["valor"] is None else float(f["valor"]),
                f["unidad"],
                f.get("fuente"),
                f.get("incertidumbre"),
                f.get("estado"),
            )
        except ValueError as e:
            raise ParametrosInvalidosException(f"{ruta.name}: {e}") from e
    muestreo = cfg.get("muestreo") or {}
    try:
        return CatalogoSupuestos(
            factores,
            salida=cfg.get("salida", "kg de café cereza"),
            muestreo_definido=bool(muestreo.get("definido", False)),
            supuesto_muestreo=muestreo.get("supuesto", ""),
            validado_con_cosecha_real=bool(cfg.get("validado_con_cosecha_real", False)),
            origen=str(ruta.name),
            version_parametros=cfg.get("version_parametros"),
            calibrado_localmente=bool(cfg.get("calibrado_localmente", False)),
            simplificaciones=tuple(cfg.get("simplificaciones") or ()),
            estados_cosechables=tuple(_estados(cfg.get("estados_cosechables") or (), ruta)),
            fuente_estados_cosechables=cfg.get("fuente_estados_cosechables"),
        )
    except ValueError as e:
        raise ParametrosInvalidosException(f"{ruta.name}: {e}") from e

"""
Manifiestos de modelos: cada archivo de pesos (models/<nombre>.<ext>) tiene un
models/<nombre>.manifest.json versionado con su SHA-256 y su procedencia.
Sin dependencias pesadas: lo usan el detector sintético (joblib) y el YOLO (.pt).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from ...domain.exceptions.domain_exceptions import ModeloNoDisponibleException

COMANDO_DESCARGA = "python scripts/descargar_modelo.py"


def sha256_archivo(ruta) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def ruta_manifiesto(ruta_modelo) -> Path:
    ruta_modelo = Path(ruta_modelo)
    return ruta_modelo.with_name(ruta_modelo.stem + ".manifest.json")


def leer_manifiesto(ruta_modelo, campos_obligatorios: Iterable[str] = ("sha256",)) -> Dict[str, Any]:
    ruta = ruta_manifiesto(ruta_modelo)
    if not ruta.is_file():
        raise ModeloNoDisponibleException(
            f"Falta el manifiesto {ruta.name} junto al modelo {Path(ruta_modelo).name}. "
            "Sin manifiesto no se puede verificar el hash, y el modelo no se carga."
        )
    try:
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ModeloNoDisponibleException(f"Manifiesto ilegible {ruta}: {e}") from e
    faltan = set(campos_obligatorios) - manifiesto.keys()
    if faltan:
        raise ModeloNoDisponibleException(f"Al manifiesto {ruta.name} le faltan campos: {sorted(faltan)}")
    return manifiesto


def verificar_archivo(ruta_modelo, campos_obligatorios: Iterable[str] = ("sha256",)) -> tuple:
    """Comprueba existencia, manifiesto y SHA-256. Devuelve (manifiesto, sha256). No deserializa nada."""
    ruta = Path(ruta_modelo)
    if not ruta.is_file():
        raise ModeloNoDisponibleException(
            f"No existe el modelo {ruta}. Descárguelo (con verificación de hash) con: "
            f"{COMANDO_DESCARGA} --manifiesto models/{ruta_manifiesto(ruta).name}"
        )
    manifiesto = leer_manifiesto(ruta, campos_obligatorios)
    sha = sha256_archivo(ruta)
    if sha != manifiesto["sha256"]:
        raise ModeloNoDisponibleException(
            f"El SHA-256 de {ruta.name} no coincide con su manifiesto "
            f"(archivo {sha[:12]}…, manifiesto {str(manifiesto['sha256'])[:12]}…). "
            f"No se carga: puede estar corrupto o no ser el archivo publicado. Vuelva a descargarlo con: {COMANDO_DESCARGA}"
        )
    return manifiesto, sha

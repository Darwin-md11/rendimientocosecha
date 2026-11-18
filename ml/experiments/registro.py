"""
Registro reproducible de corridas (regla: todo número reportado es reproducible).
Cada corrida guarda: config, semilla, commit de git, versiones de librerías,
hardware y la versión (hash) del dataset preparado.
"""

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Optional

RAIZ = Path(__file__).resolve().parents[2]
PAQUETES = ("ultralytics", "torch", "torchvision", "numba", "numpy", "scikit-learn", "pillow", "opencv-python", "pyyaml")


def _version(paquete: str) -> Optional[str]:
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return None


def commit_git() -> dict:
    def _git(*args):
        try:
            return subprocess.run(["git", "-C", str(RAIZ), *args], capture_output=True, text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": _git("rev-parse", "HEAD"),
        "rama": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "cambios_sin_commit": bool(_git("status", "--porcelain")),
    }


def capturar_entorno(ruta_datos: Optional[Path] = None) -> dict:
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    version_dataset = None
    if ruta_datos is not None:
        man = Path(ruta_datos).parent / "manifiesto.json"
        if man.is_file():
            version_dataset = json.loads(man.read_text(encoding="utf-8"))["resumen"]["version_dataset_sha256"]
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "gpu": gpu,
        "paquetes": {p: _version(p) for p in PAQUETES},
        "git": commit_git(),
        "version_dataset_sha256": version_dataset,
        "colab": "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ,
    }


def fijar_semillas(semilla: int) -> None:
    random.seed(semilla)
    os.environ["PYTHONHASHSEED"] = str(semilla)
    try:
        import numpy as np

        np.random.seed(semilla)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(semilla)
        torch.cuda.manual_seed_all(semilla)
    except ImportError:
        pass


def id_corrida(nombre: str, semilla: int) -> str:
    return f"{nombre}_s{semilla}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"


def guardar_json(ruta: Path, datos) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return ruta

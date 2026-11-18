"""
Lectura común de un dataset YOLO preparado (data.yaml + images/ + labels/ + manifiesto.json).
La usan la evaluación de M2 y los clasificadores sobre recortes (M0, M1), para que todos
lean exactamente las mismas imágenes, cajas y sesiones.
"""
import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import yaml

from src.domain.models.coffee_cherry import EstadoMadurez
from src.infrastructure.adapters.detectors.yolo_detector_adapter import normalizar_nombre_clase

ESTADOS = list(EstadoMadurez)
NOMBRES = [e.value for e in ESTADOS]
EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imagenes_y_etiquetas(data_yaml: Path, split: str) -> Iterator[Tuple[Path, Path, Dict[int, int]]]:
    """(imagen, archivo_de_etiquetas, {clase_dataset: índice de EstadoMadurez}) para cada imagen de la partición."""
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    raiz = Path(data.get("path", Path(data_yaml).parent))
    carpeta = raiz / data[split]
    nombres = data["names"]
    nombres = [nombres[i] for i in sorted(nombres)] if isinstance(nombres, dict) else list(nombres)
    a_estado = {i: NOMBRES.index(normalizar_nombre_clase(n)) for i, n in enumerate(nombres) if normalizar_nombre_clase(n) in NOMBRES}
    partes = ["labels" if x == "images" else x for x in Path(data[split]).parts]  # convención YOLO: images/... -> labels/...
    carpeta_lbl = raiz.joinpath(*partes)
    for img in sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXT_IMAGEN):
        yield img, carpeta_lbl / (img.stem + ".txt"), a_estado


def cajas_gt(lbl: Path, ancho: int, alto: int, a_estado: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Cajas en píxeles xyxy y su índice de EstadoMadurez."""
    cajas, clases = [], []
    if lbl.is_file():
        for linea in lbl.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            c, cx, cy, w, h = linea.split()
            c = int(float(c))
            if c not in a_estado:
                continue
            cx, cy, w, h = float(cx) * ancho, float(cy) * alto, float(w) * ancho, float(h) * alto
            cajas.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            clases.append(a_estado[c])
    return np.asarray(cajas, float).reshape(-1, 4), np.asarray(clases, int)


def grupos_por_archivo(data_yaml: Path) -> Dict[str, str]:
    """{nombre de imagen: grupo (sesión o foto)} según el manifiesto de la preparación; vacío si no hay manifiesto."""
    man = Path(data_yaml).parent / "manifiesto.json"
    if not man.is_file():
        return {}
    return {a["archivo"]: a["grupo"] for a in json.loads(man.read_text(encoding="utf-8"))["archivos"]}


def version_dataset(data_yaml: Path) -> str:
    man = Path(data_yaml).parent / "manifiesto.json"
    if not man.is_file():
        return "sin_manifiesto"
    return json.loads(man.read_text(encoding="utf-8"))["resumen"]["version_dataset_sha256"]

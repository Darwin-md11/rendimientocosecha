"""
YOLO Fruit Detector Adapter (M2)

Detecta y clasifica cerezas en una pasada con un modelo YOLO (Ultralytics)
entrenado en ml/m2_yolo/. Decisiones (ver docs/decisiones/ADR-008):

- `import ultralytics` diferido: la app no exige PyTorch salvo que se use este detector.
- Pesos verificados con su manifiesto (SHA-256) antes de cargarlos.
- La imagen se entrega como PIL RGB *solo píxeles, sin EXIF*: Ultralytics trata un
  ndarray como BGR y aplica exif_transpose por su cuenta a las PIL; el caso de uso ya
  orientó la foto, así que una segunda rotación desalinearía las cajas.
- NMS agnóstico a la clase: una cereza no se cuenta dos veces con estados distintos.
- Confianzas comparadas con tolerancia (el modelo trabaja en float32).
- Cajas recortadas a la imagen; se descartan las vacías o no finitas.
- Al cargar se valida que las clases del modelo se puedan traducir a EstadoMadurez.
- Lock alrededor de predict: Gradio atiende peticiones concurrentes.
"""

import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ....domain.exceptions.domain_exceptions import ModeloNoDisponibleException
from ....domain.models.coffee_cherry import BoundingBox, DeteccionFruto, EstadoMadurez
from ....domain.models.modelo_info import FamiliaModelo, ModeloInfo
from ....domain.ports.fruit_detector_port import FruitDetectorPort
from ...ml.manifiesto import verificar_archivo

_TOL = 1e-6


def normalizar_nombre_clase(nombre: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in sin_tildes.upper() if ch.isalnum())


def mapa_clases(nombres) -> Dict[int, EstadoMadurez]:
    """{índice del modelo: EstadoMadurez} para las clases traducibles (las demás se ignoran)."""
    if isinstance(nombres, dict):
        items = nombres.items()
    else:
        items = enumerate(nombres)
    validos = {e.value: e for e in EstadoMadurez}
    return {int(i): validos[normalizar_nombre_clase(n)] for i, n in items if normalizar_nombre_clase(n) in validos}


def _np(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def info_desde_manifiesto(manifiesto: Dict[str, Any], sha256: Optional[str]) -> ModeloInfo:
    return ModeloInfo(
        nombre=manifiesto.get("nombre", "YOLO"),
        familia=FamiliaModelo.M2_YOLO,
        version=f"ultralytics {manifiesto.get('version_framework', '?')}, imgsz {manifiesto.get('imgsz', '?')}",
        datos_entrenamiento=manifiesto.get("datos_entrenamiento", "no registrado"),
        validado_con_datos_reales=bool(manifiesto.get("validado_con_datos_reales", False)),
        hash_pesos_sha256=sha256,
        fecha_entrenamiento=manifiesto.get("fecha_entrenamiento"),
        dominio_validacion=manifiesto.get("dominio_validacion"),
        validado_en_dominio_objetivo=bool(manifiesto.get("validado_en_dominio_objetivo", False)),
    )


class YoloFruitDetectorAdapter(FruitDetectorPort):
    def __init__(self, modelo: Any, info: ModeloInfo, imgsz: int = 640, iou: float = 0.6, max_det: int = 1000, device: Optional[str] = None):
        self.modelo = modelo
        self._info = info
        self.imgsz = imgsz
        self.iou = iou
        self.max_det = max_det
        self.device = device
        self._clases = mapa_clases(getattr(modelo, "names", {}))
        if not self._clases:
            raise ModeloNoDisponibleException(
                f"Ninguna clase del modelo YOLO ({getattr(modelo, 'names', None)}) corresponde a un estado de madurez "
                f"{[e.value for e in EstadoMadurez]}."
            )
        self._lock = threading.Lock()

    @classmethod
    def desde_archivo(cls, ruta_pesos, **kwargs) -> "YoloFruitDetectorAdapter":
        manifiesto, sha = verificar_archivo(ruta_pesos, ("sha256", "clases"))
        try:
            from ultralytics import YOLO  # import diferido: solo si se usa este detector
        except ImportError as e:
            raise ModeloNoDisponibleException(
                "El detector YOLO necesita la librería 'ultralytics' (y PyTorch). Instálela con: pip install -r requirements-ml.txt"
            ) from e
        modelo = YOLO(str(Path(ruta_pesos)))
        nombres_modelo = [modelo.names[i] for i in sorted(modelo.names)] if isinstance(modelo.names, dict) else list(modelo.names)
        if [normalizar_nombre_clase(n) for n in nombres_modelo] != [normalizar_nombre_clase(n) for n in manifiesto["clases"]]:
            raise ModeloNoDisponibleException(f"Las clases del modelo {nombres_modelo} no coinciden con su manifiesto {manifiesto['clases']}.")
        kwargs.setdefault("imgsz", int(manifiesto.get("imgsz", 640)))
        kwargs.setdefault("iou", float(manifiesto.get("iou_nms", 0.6)))
        return cls(modelo, info_desde_manifiesto(manifiesto, sha), **kwargs)

    def info(self) -> ModeloInfo:
        return self._info

    def detectar_frutos(self, imagen: Image.Image, umbral_confianza: float = 0.40) -> List[DeteccionFruto]:
        ancho, alto = imagen.size
        solo_pixeles = Image.fromarray(np.asarray(imagen.convert("RGB")))  # RGB, sin EXIF ni metadatos
        with self._lock:
            resultados = self.modelo.predict(
                source=solo_pixeles,
                conf=max(0.0, umbral_confianza - _TOL),
                iou=self.iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                agnostic_nms=True,
                device=self.device,
                verbose=False,
            )
        if not resultados:
            return []
        cajas = getattr(resultados[0], "boxes", None)
        if cajas is None or len(cajas) == 0:
            return []
        xyxy = _np(cajas.xyxy).astype(float).reshape(-1, 4)
        confs = _np(cajas.conf).astype(float).reshape(-1)
        clases = _np(cajas.cls).astype(int).reshape(-1)

        crudas = []
        for (x0, y0, x1, y1), conf, c in zip(xyxy, confs, clases):
            if c not in self._clases or not np.all(np.isfinite([x0, y0, x1, y1, conf])):
                continue
            if conf < umbral_confianza - _TOL:
                continue
            x0, x1 = float(np.clip(x0, 0, ancho)), float(np.clip(x1, 0, ancho))
            y0, y1 = float(np.clip(y0, 0, alto)), float(np.clip(y1, 0, alto))
            if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
                continue
            crudas.append((float(min(max(conf, 0.0), 1.0)), self._clases[c], (x0, y0, x1, y1)))
        crudas.sort(key=lambda t: (-t[0], t[2]))
        return [
            DeteccionFruto(id_deteccion=f"Y{i:03d}", estado=estado, confianza=conf, bbox=BoundingBox(*caja))
            for i, (conf, estado, caja) in enumerate(crudas, start=1)
        ]

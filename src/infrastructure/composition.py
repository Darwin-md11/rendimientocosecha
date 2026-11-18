"""
Composicion de dependencias: decide que adaptador de deteccion se inyecta.

  SIKAFE_DETECTOR=yolo      M2: YOLO entrenado con fotos reales (models/m2_yolo.pt,
                            o la ruta en SIKAFE_MODELO_YOLO). Requiere ultralytics.
  SIKAFE_DETECTOR=sintetico clasificador de pixeles entrenado SOLO con escenas
                            sinteticas: demo/regresion, "SINTÉTICO – NO VALIDADO" (alias "ml").
  SIKAFE_DETECTOR=mock      detector simulado (solo pruebas).

Por defecto: "yolo" si existe su manifiesto (models/m2_yolo.manifest.json, es decir,
el equipo ya publico un modelo real); si no, "sintetico" (ADR-003).

Regla: si el detector pedido no se puede construir, se lanza un error claro.
Nunca se reentrena al arrancar ni se cambia en silencio a otro detector.
"""
import os
from pathlib import Path
from typing import Optional

from ..domain.models.supuestos import CatalogoSupuestos
from ..domain.ports.fruit_detector_port import FruitDetectorPort
from .adapters.detectors.mock_detector_adapter import MockFruitDetectorAdapter

_MODELOS = Path(__file__).resolve().parents[2] / "models"
RUTA_MODELO_POR_DEFECTO = _MODELOS / "madurez_pixel.joblib"
RUTA_YOLO_POR_DEFECTO = _MODELOS / "m2_yolo.pt"
_ALIAS = {"ml": "sintetico"}
DETECTORES_DISPONIBLES = ("yolo", "sintetico", "mock")


def detector_por_defecto() -> str:
    ruta_yolo = Path(os.environ.get("SIKAFE_MODELO_YOLO", RUTA_YOLO_POR_DEFECTO))
    return "yolo" if ruta_yolo.with_name(ruta_yolo.stem + ".manifest.json").is_file() else "sintetico"


def construir_detector(tipo: Optional[str] = None, ruta_modelo: Optional[Path] = None) -> FruitDetectorPort:
    tipo = (tipo or os.environ.get("SIKAFE_DETECTOR") or detector_por_defecto()).strip().lower()
    tipo = _ALIAS.get(tipo, tipo)

    if tipo == "mock":
        return MockFruitDetectorAdapter(num_simulated_fruits=18)
    if tipo == "yolo":
        from .adapters.detectors.yolo_detector_adapter import YoloFruitDetectorAdapter

        ruta = Path(ruta_modelo or os.environ.get("SIKAFE_MODELO_YOLO", RUTA_YOLO_POR_DEFECTO))
        return YoloFruitDetectorAdapter.desde_archivo(ruta)
    if tipo != "sintetico":
        raise ValueError(f"Detector desconocido: {tipo!r} (use uno de {', '.join(DETECTORES_DISPONIBLES)}).")

    # Imports aqui para que el modo 'mock' no exija scikit-learn / scikit-image.
    from .adapters.detectors.ml_detector_adapter import MLFruitDetectorAdapter
    from .ml.modelo import ModeloMadurezPixel

    # Si falla, ModeloNoDisponibleException sube con el motivo y como resolverlo.
    modelo = ModeloMadurezPixel.cargar(Path(ruta_modelo) if ruta_modelo else RUTA_MODELO_POR_DEFECTO)
    return MLFruitDetectorAdapter(modelo)


def construir_catalogo(ruta: Optional[Path] = None) -> CatalogoSupuestos:
    """Factores agronómicos con su fuente (configs/agronomia.yaml): única fuente de los valores por defecto."""
    from .config.agronomia import cargar_catalogo

    return cargar_catalogo(ruta)

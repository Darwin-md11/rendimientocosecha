"""
Composicion de dependencias: decide que adaptador de deteccion se inyecta.

  SIKAFE_DETECTOR=ml    (por defecto) detector con machine learning
  SIKAFE_DETECTOR=mock  detector simulado (solo pruebas)
"""
import os
from pathlib import Path
from typing import Optional

from ..domain.exceptions.domain_exceptions import ModeloNoDisponibleException
from ..domain.ports.fruit_detector_port import FruitDetectorPort
from .adapters.detectors.mock_detector_adapter import MockFruitDetectorAdapter

RUTA_MODELO_POR_DEFECTO = Path(__file__).resolve().parents[2] / "models" / "madurez_pixel.joblib"


def construir_detector(
    tipo: Optional[str] = None,
    ruta_modelo: Optional[Path] = None,
    auto_entrenar: bool = True,
) -> FruitDetectorPort:
    tipo = (tipo or os.environ.get("SIKAFE_DETECTOR", "ml")).strip().lower()

    if tipo == "mock":
        return MockFruitDetectorAdapter(num_simulated_fruits=18)
    if tipo != "ml":
        raise ValueError(f"Detector desconocido: {tipo!r} (use 'ml' o 'mock').")

    # Imports aqui para que el modo 'mock' no exija scikit-learn / scikit-image.
    from .adapters.detectors.ml_detector_adapter import MLFruitDetectorAdapter
    from .ml.modelo import ModeloMadurezPixel

    ruta = Path(ruta_modelo) if ruta_modelo else RUTA_MODELO_POR_DEFECTO
    try:
        modelo = ModeloMadurezPixel.cargar(ruta)
    except ModeloNoDisponibleException as e:
        if not auto_entrenar:
            raise
        # Modelo ausente o incompatible con las librerias instaladas: se reentrena (~30 s).
        from .ml.training import entrenar

        print(f"[SIKAFE] {e}\n[SIKAFE] Entrenando un modelo nuevo con datos sinteticos...")
        modelo = entrenar()
        try:
            modelo.guardar(ruta)
        except OSError as err:
            print(f"[SIKAFE] No se pudo guardar el modelo en {ruta}: {err}")
    return MLFruitDetectorAdapter(modelo)

"""
Composicion de dependencias: decide que adaptador de deteccion se inyecta.

  SIKAFE_DETECTOR=sintetico (por defecto) clasificador de pixeles entrenado SOLO
                            con escenas sinteticas: demo/regresion, "SINTÉTICO – NO VALIDADO".
                            Se acepta "ml" como alias por compatibilidad.
  SIKAFE_DETECTOR=mock      detector simulado (solo pruebas).

Regla: si el detector pedido no se puede construir, se lanza un error claro.
Nunca se reentrena al arrancar ni se cambia en silencio a otro detector.
Cuando exista un modelo validado con datos reales, pasara a ser el valor por
defecto y el sintetico quedara solo para demo/regresion (ADR-003).
"""
import os
from pathlib import Path
from typing import Optional

from ..domain.ports.fruit_detector_port import FruitDetectorPort
from .adapters.detectors.mock_detector_adapter import MockFruitDetectorAdapter

RUTA_MODELO_POR_DEFECTO = Path(__file__).resolve().parents[2] / "models" / "madurez_pixel.joblib"
DETECTOR_POR_DEFECTO = "sintetico"
_ALIAS = {"ml": "sintetico"}
DETECTORES_DISPONIBLES = ("sintetico", "mock")


def construir_detector(tipo: Optional[str] = None, ruta_modelo: Optional[Path] = None) -> FruitDetectorPort:
    tipo = (tipo or os.environ.get("SIKAFE_DETECTOR", DETECTOR_POR_DEFECTO)).strip().lower()
    tipo = _ALIAS.get(tipo, tipo)

    if tipo == "mock":
        return MockFruitDetectorAdapter(num_simulated_fruits=18)
    if tipo != "sintetico":
        raise ValueError(f"Detector desconocido: {tipo!r} (use uno de {', '.join(DETECTORES_DISPONIBLES)}).")

    # Imports aqui para que el modo 'mock' no exija scikit-learn / scikit-image.
    from .adapters.detectors.ml_detector_adapter import MLFruitDetectorAdapter
    from .ml.modelo import ModeloMadurezPixel

    # Si falla, ModeloNoDisponibleException sube con el motivo y como resolverlo.
    modelo = ModeloMadurezPixel.cargar(Path(ruta_modelo) if ruta_modelo else RUTA_MODELO_POR_DEFECTO)
    return MLFruitDetectorAdapter(modelo)

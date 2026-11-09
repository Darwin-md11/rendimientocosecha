"""
ModeloMadurezPixel: envoltorio persistible del clasificador de pixeles.

Guarda el clasificador junto con metadatos (metricas, versiones) para poder
rechazar modelos incompatibles con la version actual de scikit-learn o de las
caracteristicas, en lugar de fallar de forma silenciosa o dar resultados raros.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import sklearn

from ...domain.exceptions.domain_exceptions import ModeloNoDisponibleException
from .features import CLASES, VERSION_CARACTERISTICAS, extraer_caracteristicas


@dataclass
class ModeloMadurezPixel:
    clasificador: Any
    metadatos: Dict[str, Any] = field(default_factory=dict)

    def predecir_proba(self, rgb: np.ndarray) -> np.ndarray:
        """Probabilidades por clase, forma (alto, ancho, len(CLASES))."""
        alto, ancho = rgb.shape[:2]
        probas = self.clasificador.predict_proba(extraer_caracteristicas(rgb))
        salida = np.zeros((alto * ancho, len(CLASES)), dtype=np.float32)
        salida[:, self.clasificador.classes_] = probas
        return salida.reshape(alto, ancho, len(CLASES))

    def guardar(self, ruta) -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clasificador": self.clasificador,
                "metadatos": self.metadatos,
                "version_sklearn": sklearn.__version__,
                "version_caracteristicas": VERSION_CARACTERISTICAS,
            },
            ruta,
            compress=3,
        )
        return ruta

    @classmethod
    def cargar(cls, ruta) -> "ModeloMadurezPixel":
        # joblib usa pickle: cargar solo modelos generados por este proyecto.
        ruta = Path(ruta)
        if not ruta.is_file():
            raise ModeloNoDisponibleException(f"No existe el modelo: {ruta}")
        try:
            paquete = joblib.load(ruta)
            if paquete["version_caracteristicas"] != VERSION_CARACTERISTICAS:
                raise ModeloNoDisponibleException(
                    "El modelo fue entrenado con otra version de caracteristicas."
                )
            if paquete["version_sklearn"] != sklearn.__version__:
                raise ModeloNoDisponibleException(
                    f"El modelo se guardo con scikit-learn {paquete['version_sklearn']} "
                    f"y se esta usando {sklearn.__version__}."
                )
            return cls(paquete["clasificador"], paquete.get("metadatos", {}))
        except ModeloNoDisponibleException:
            raise
        except Exception as e:
            raise ModeloNoDisponibleException(f"No se pudo leer el modelo {ruta}: {e}") from e

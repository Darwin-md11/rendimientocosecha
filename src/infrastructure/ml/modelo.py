"""
ModeloMadurezPixel: envoltorio persistible del clasificador de pixeles.

Cada archivo .joblib va acompañado de un manifiesto JSON (mismo nombre,
extensión .manifest.json) con su SHA-256, la versión de scikit-learn y la
procedencia. Al cargar se verifica el manifiesto ANTES de deserializar:
joblib usa pickle y ejecuta código al cargar, así que solo se aceptan archivos
cuyo hash coincide con el manifiesto versionado en git.

Si algo no coincide se lanza ModeloNoDisponibleException con un mensaje que
dice cómo arreglarlo. Nunca se reentrena ni se cambia de detector en silencio.
"""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import sklearn

from ...domain.exceptions.domain_exceptions import ModeloNoDisponibleException
from .features import CLASES, VERSION_CARACTERISTICAS, extraer_caracteristicas

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


def leer_manifiesto(ruta_modelo) -> Dict[str, Any]:
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
    faltan = {"sha256", "version_sklearn", "version_caracteristicas"} - manifiesto.keys()
    if faltan:
        raise ModeloNoDisponibleException(f"Al manifiesto {ruta.name} le faltan campos: {sorted(faltan)}")
    return manifiesto


@dataclass
class ModeloMadurezPixel:
    clasificador: Any
    metadatos: Dict[str, Any] = field(default_factory=dict)
    sha256: Optional[str] = None
    manifiesto: Dict[str, Any] = field(default_factory=dict)

    def predecir_proba(self, rgb: np.ndarray) -> np.ndarray:
        """Probabilidades por clase, forma (alto, ancho, len(CLASES))."""
        alto, ancho = rgb.shape[:2]
        probas = self.clasificador.predict_proba(extraer_caracteristicas(rgb))
        salida = np.zeros((alto * ancho, len(CLASES)), dtype=np.float32)
        salida[:, self.clasificador.classes_] = probas
        return salida.reshape(alto, ancho, len(CLASES))

    def guardar(self, ruta, url_descarga: Optional[str] = None) -> Path:
        """Guarda el .joblib y escribe su manifiesto (hash + versiones + procedencia)."""
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
        self.sha256 = sha256_archivo(ruta)
        reales = int(self.metadatos.get("imagenes_reales", 0))
        self.manifiesto = {
            "archivo": ruta.name,
            "sha256": self.sha256,
            "version_sklearn": sklearn.__version__,
            "version_caracteristicas": VERSION_CARACTERISTICAS,
            "familia": "SINTETICO" if reales == 0 else "M0",
            "datos_entrenamiento": (
                f"{self.metadatos.get('escenas_sinteticas', '?')} escenas sintéticas, {reales} imágenes reales"
            ),
            "validado_con_datos_reales": False,
            "semilla": self.metadatos.get("semilla"),
            "fecha_entrenamiento": datetime.now(timezone.utc).date().isoformat(),
            "url_descarga": url_descarga,
        }
        ruta_manifiesto(ruta).write_text(json.dumps(self.manifiesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return ruta

    @classmethod
    def cargar(cls, ruta) -> "ModeloMadurezPixel":
        ruta = Path(ruta)
        if not ruta.is_file():
            raise ModeloNoDisponibleException(
                f"No existe el modelo {ruta}. Descárguelo (con verificación de hash) con: {COMANDO_DESCARGA}"
            )
        manifiesto = leer_manifiesto(ruta)

        sha = sha256_archivo(ruta)
        if sha != manifiesto["sha256"]:
            raise ModeloNoDisponibleException(
                f"El SHA-256 de {ruta.name} no coincide con su manifiesto "
                f"(archivo {sha[:12]}…, manifiesto {str(manifiesto['sha256'])[:12]}…). "
                f"No se carga: puede estar corrupto o no ser el archivo publicado. Vuelva a descargarlo con: {COMANDO_DESCARGA}"
            )
        if manifiesto["version_caracteristicas"] != VERSION_CARACTERISTICAS:
            raise ModeloNoDisponibleException(
                f"El modelo usa la versión {manifiesto['version_caracteristicas']} de características "
                f"y el código la {VERSION_CARACTERISTICAS}."
            )
        if manifiesto["version_sklearn"] != sklearn.__version__:
            raise ModeloNoDisponibleException(
                f"El modelo se guardó con scikit-learn {manifiesto['version_sklearn']} y está instalada la "
                f"{sklearn.__version__}. Instale la versión exacta: pip install scikit-learn=={manifiesto['version_sklearn']} "
                "(o reentrene el modelo de demostración con: python -m ml.sintetico.training)."
            )

        try:
            paquete = joblib.load(ruta)  # seguro solo porque el hash ya coincidió con el manifiesto versionado
            clasificador = paquete["clasificador"]
            metadatos = paquete.get("metadatos", {})
        except Exception as e:
            raise ModeloNoDisponibleException(f"No se pudo leer el modelo {ruta}: {e}") from e
        return cls(clasificador, metadatos, sha256=sha, manifiesto=manifiesto)

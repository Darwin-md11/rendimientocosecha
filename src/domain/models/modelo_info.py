"""
Domain Value Object: ModeloInfo

Procedencia del modelo que produjo un resultado. Todo detector la expone
(FruitDetectorPort.info()) y viaja en la respuesta del caso de uso, para que
ningún resultado de un modelo simulado o no validado se presente como real.
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FamiliaModelo(str, Enum):
    SINTETICO = "SINTETICO"  # entrenado solo con escenas sintéticas (demo/regresión)
    MOCK = "MOCK"  # simulado: ignora la imagen
    M0_CLASICO = "M0"  # color/textura + SVM/RF sobre recortes
    M1_CNN = "M1"  # CNN con transfer learning sobre recortes
    M2_YOLO = "M2"  # detector YOLO


_FAMILIAS_SIMULADAS = (FamiliaModelo.SINTETICO, FamiliaModelo.MOCK)
_PATRON_SHA256 = re.compile(r"^[0-9a-f]{64}$")

ETIQUETA_SINTETICO = "SINTÉTICO – NO VALIDADO"
ETIQUETA_MOCK = "SIMULADO – NO VALIDADO"
ETIQUETA_NO_VALIDADO = "NO VALIDADO CON DATOS REALES"
ETIQUETA_OTRO_DOMINIO = "VALIDADO SOLO EN OTRO DOMINIO"


@dataclass(frozen=True)
class ModeloInfo:
    nombre: str
    familia: FamiliaModelo
    version: str
    datos_entrenamiento: str
    validado_con_datos_reales: bool
    hash_pesos_sha256: Optional[str] = None
    fecha_entrenamiento: Optional[str] = None
    dominio_validacion: Optional[str] = None  # p. ej. "Croppie test (Uganda), corrida X"
    validado_en_dominio_objetivo: bool = False  # validado con fotos de fincas de Norte de Santander

    def __post_init__(self):
        if not self.nombre.strip() or not self.version.strip() or not self.datos_entrenamiento.strip():
            raise ValueError("nombre, version y datos_entrenamiento son obligatorios.")
        if self.familia in _FAMILIAS_SIMULADAS and self.validado_con_datos_reales:
            raise ValueError(f"Un modelo de familia {self.familia.value} no puede declararse validado con datos reales.")
        if self.validado_en_dominio_objetivo and not self.validado_con_datos_reales:
            raise ValueError("validado_en_dominio_objetivo exige validado_con_datos_reales.")
        if self.hash_pesos_sha256 is not None and not _PATRON_SHA256.match(self.hash_pesos_sha256):
            raise ValueError("hash_pesos_sha256 debe ser un SHA-256 en hexadecimal (64 caracteres en minúscula).")

    @property
    def es_simulado(self) -> bool:
        return self.familia in _FAMILIAS_SIMULADAS

    @property
    def etiqueta(self) -> Optional[str]:
        """Etiqueta corta para banners e imágenes; None si el modelo está validado."""
        if self.familia is FamiliaModelo.SINTETICO:
            return ETIQUETA_SINTETICO
        if self.familia is FamiliaModelo.MOCK:
            return ETIQUETA_MOCK
        if not self.validado_con_datos_reales:
            return ETIQUETA_NO_VALIDADO
        if not self.validado_en_dominio_objetivo:
            return ETIQUETA_OTRO_DOMINIO
        return None

    @property
    def advertencias(self) -> list[str]:
        if self.familia is FamiliaModelo.SINTETICO:
            return [
                f"{ETIQUETA_SINTETICO}: el modelo '{self.nombre}' se entrenó solo con escenas sintéticas "
                "y no se ha validado con fotos reales. Los conteos y el rendimiento NO son válidos para decisiones de campo."
            ]
        if self.familia is FamiliaModelo.MOCK:
            return [f"{ETIQUETA_MOCK}: detector simulado que ignora la imagen (solo para pruebas)."]
        if not self.validado_con_datos_reales:
            return [f"{ETIQUETA_NO_VALIDADO}: el modelo '{self.nombre}' aún no se ha evaluado con datos reales de prueba."]
        if not self.validado_en_dominio_objetivo:
            return [
                f"{ETIQUETA_OTRO_DOMINIO}: '{self.nombre}' se evaluó con {self.dominio_validacion or 'datos de otro origen'}; "
                "aún no se ha evaluado con fotos de fincas de Norte de Santander. Los conteos son indicativos."
            ]
        return []

    def como_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "familia": self.familia.value,
            "version": self.version,
            "datos_entrenamiento": self.datos_entrenamiento,
            "validado_con_datos_reales": self.validado_con_datos_reales,
            "es_simulado": self.es_simulado,
            "hash_pesos_sha256": self.hash_pesos_sha256,
            "fecha_entrenamiento": self.fecha_entrenamiento,
            "dominio_validacion": self.dominio_validacion,
            "validado_en_dominio_objetivo": self.validado_en_dominio_objetivo,
        }

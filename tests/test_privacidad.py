"""Ley 1581: la imagen anotada que devuelve la app no conserva metadatos EXIF (cámara, GPS) de la foto original."""
import io

from PIL import Image

from src.application.dtos.estimate_dto import SolicitudEstimacionDTO
from src.application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from src.infrastructure.adapters.detectors.mock_detector_adapter import MockFruitDetectorAdapter


def test_imagen_anotada_sin_exif_ni_gps_y_con_rotacion_aplicada():
    exif = Image.Exif()
    exif[0x010F] = "CamaraDePrueba"  # fabricante
    exif[0x0112] = 6  # orientación: girar 90°
    exif.get_ifd(0x8825)[2] = (7.0, 22.0, 0.0)  # latitud GPS
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (30, 120, 40)).save(buf, "JPEG", exif=exif)
    entrada = Image.open(io.BytesIO(buf.getvalue())).getexif()
    assert entrada.get_ifd(0x8825)  # la foto de entrada sí trae GPS

    r = AnalizarImagenCosechaUseCase(MockFruitDetectorAdapter(5)).ejecutar(SolicitudEstimacionDTO(imagen_bytes=buf.getvalue()))
    salida = Image.open(io.BytesIO(r.imagen_anotada_bytes))
    assert dict(salida.getexif()) == {}
    assert salida.width == 200  # se aplicó la orientación EXIF antes de descartar los metadatos

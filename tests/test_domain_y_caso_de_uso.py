import io

import numpy as np
import pytest
from PIL import Image

from src.application.dtos.estimate_dto import SolicitudEstimacionDTO
from src.application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from src.domain.exceptions.domain_exceptions import (
    ImagenInvalidaException,
    ParametrosInvalidosException,
    UmbralInvalidoException,
)
from src.domain.models.agronomic_params import ParametrosAgronomicos
from src.domain.models.coffee_cherry import BoundingBox, DeteccionFruto, EstadoMadurez
from src.domain.services.yield_calculator import CalculadoraRendimientoService
from src.infrastructure.adapters.detectors.mock_detector_adapter import MockFruitDetectorAdapter


def _bytes(modo="RGB", formato="JPEG", tam=(200, 150)):
    buf = io.BytesIO()
    Image.new(modo, tam, (30, 120, 30, 255) if modo == "RGBA" else (30, 120, 30)).save(buf, format=formato)
    return buf.getvalue()


def _det(estado, i=0):
    return DeteccionFruto(f"d{i}", estado, 0.9, BoundingBox(1, 1, 10, 10))


def test_calculo_de_rendimiento():
    params = ParametrosAgronomicos(peso_medio_fruto_g=2.0, factor_oclusion=1.5, plantas_muestra=2, plantas_totales_lote=100)
    dets = [_det(EstadoMadurez.MADURO, i) for i in range(4)]
    reporte = CalculadoraRendimientoService.calcular(dets, params)
    # 4 frutos * 1.5 * 0.002 kg * (100 / 2) = 0.6 kg
    assert reporte.rendimiento_estimado_kg == pytest.approx(0.6)
    assert reporte.porcentaje_madurez == 100.0


def test_sin_detecciones_rinde_cero():
    reporte = CalculadoraRendimientoService.calcular([], ParametrosAgronomicos())
    assert reporte.total_frutos == 0 and reporte.rendimiento_estimado_kg == 0.0 and reporte.porcentaje_madurez == 0.0


@pytest.mark.parametrize("kwargs", [
    dict(peso_medio_fruto_g=0), dict(factor_oclusion=0.5),
    dict(plantas_muestra=0), dict(plantas_muestra=10, plantas_totales_lote=5),
])
def test_parametros_agronomicos_invalidos(kwargs):
    with pytest.raises(ValueError):
        ParametrosAgronomicos(**kwargs)


def test_caso_de_uso_con_jpeg_y_png_rgba():
    uc = AnalizarImagenCosechaUseCase(MockFruitDetectorAdapter(10))
    for datos in (_bytes(), _bytes("RGBA", "PNG")):
        r = uc.ejecutar(SolicitudEstimacionDTO(imagen_bytes=datos))
        assert r.total_frutos == len(r.detecciones_detalle)
        Image.open(io.BytesIO(r.imagen_anotada_bytes)).verify()


def test_errores_de_validacion():
    uc = AnalizarImagenCosechaUseCase(MockFruitDetectorAdapter(5))
    with pytest.raises(UmbralInvalidoException):
        uc.ejecutar(SolicitudEstimacionDTO(imagen_bytes=_bytes(), umbral_confianza=1.5))
    with pytest.raises(ImagenInvalidaException):
        uc.ejecutar(SolicitudEstimacionDTO(imagen_bytes=b""))
    with pytest.raises(ImagenInvalidaException):
        uc.ejecutar(SolicitudEstimacionDTO(imagen_bytes=b"esto no es una imagen"))
    # Antes llegaba como ValueError generico, ahora es una excepcion de dominio.
    with pytest.raises(ParametrosInvalidosException):
        uc.ejecutar(SolicitudEstimacionDTO(imagen_bytes=_bytes(), peso_medio_fruto_g=-1))


def test_mock_no_modifica_el_estado_global_de_numpy():
    np.random.seed(123)
    esperado = np.random.random()
    np.random.seed(123)
    MockFruitDetectorAdapter(10).detectar_frutos(Image.new("RGB", (200, 150)))
    assert np.random.random() == esperado


def test_mock_cajas_dentro_de_la_imagen_incluso_si_es_pequena():
    for tam in [(40, 40), (64, 48), (640, 480)]:
        for d in MockFruitDetectorAdapter(30).detectar_frutos(Image.new("RGB", tam), 0.0):
            assert 0 <= d.bbox.x_min < d.bbox.x_max <= tam[0]
            assert 0 <= d.bbox.y_min < d.bbox.y_max <= tam[1]

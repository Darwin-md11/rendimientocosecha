import numpy as np
import pytest
from PIL import Image

from src.domain.exceptions.domain_exceptions import ModeloNoDisponibleException
from src.infrastructure.adapters.detectors.ml_detector_adapter import MLFruitDetectorAdapter
from src.infrastructure.composition import RUTA_MODELO_POR_DEFECTO, construir_detector
from src.infrastructure.ml.modelo import ModeloMadurezPixel
from src.infrastructure.ml.synthetic_data import generar_escena
from src.infrastructure.ml.training import evaluar


@pytest.fixture(scope="module")
def modelo():
    return ModeloMadurezPixel.cargar(RUTA_MODELO_POR_DEFECTO)


def test_el_detector_depende_de_la_imagen(modelo):
    det = MLFruitDetectorAdapter(modelo)
    vacia = Image.new("RGB", (320, 240), (25, 90, 30))  # solo "follaje" uniforme
    assert det.detectar_frutos(vacia) == []

    escena = generar_escena(np.random.default_rng(3))
    con_frutos = det.detectar_frutos(Image.fromarray(escena.rgb))
    assert len(con_frutos) > 0


def test_resultados_deterministas_y_cajas_validas(modelo):
    det = MLFruitDetectorAdapter(modelo)
    escena = generar_escena(np.random.default_rng(11))
    img = Image.fromarray(escena.rgb)
    a, b = det.detectar_frutos(img), det.detectar_frutos(img)
    assert [(d.estado, d.bbox) for d in a] == [(d.estado, d.bbox) for d in b]
    for d in a:
        assert 0 <= d.bbox.x_min < d.bbox.x_max <= img.width
        assert 0 <= d.bbox.y_min < d.bbox.y_max <= img.height
        assert 0.0 <= d.confianza <= 1.0


def test_imagen_grande_devuelve_cajas_en_coordenadas_originales(modelo):
    det = MLFruitDetectorAdapter(modelo)
    escena = generar_escena(np.random.default_rng(5))
    grande = Image.fromarray(escena.rgb).resize((1280, 960), Image.BICUBIC)
    dets = det.detectar_frutos(grande)
    assert dets, "deberia detectar frutos tambien en la version ampliada"
    assert all(d.bbox.x_max <= 1280 and d.bbox.y_max <= 960 for d in dets)
    assert max(d.bbox.x_max for d in dets) > 320  # coordenadas reescaladas a la imagen original


def test_umbral_alto_filtra_detecciones(modelo):
    det = MLFruitDetectorAdapter(modelo)
    img = Image.fromarray(generar_escena(np.random.default_rng(8)).rgb)
    assert len(det.detectar_frutos(img, 0.95)) <= len(det.detectar_frutos(img, 0.2))


def test_calidad_minima_en_escenas_sinteticas_nuevas(modelo):
    m = evaluar(modelo, n_escenas=15, semilla=987_654)
    assert m["f1_deteccion"] > 0.6
    assert m["exactitud_estado_en_detectados"] > 0.85


def test_modelo_inexistente_o_corrupto(tmp_path):
    with pytest.raises(ModeloNoDisponibleException):
        ModeloMadurezPixel.cargar(tmp_path / "no_existe.joblib")
    malo = tmp_path / "malo.joblib"
    malo.write_bytes(b"no es un modelo")
    with pytest.raises(ModeloNoDisponibleException):
        ModeloMadurezPixel.cargar(malo)
    with pytest.raises(ModeloNoDisponibleException):
        construir_detector("ml", ruta_modelo=malo, auto_entrenar=False)


def test_detector_desconocido():
    with pytest.raises(ValueError):
        construir_detector("otro")

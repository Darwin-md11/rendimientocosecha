"""
Fase 0: procedencia del modelo, advertencias visibles y contrato de detectores.

Criterio de aceptación: ningún resultado de un detector no validado puede
presentarse como "✅" ni sin su advertencia (UI, CLI, imagen anotada).
"""
import io
import os
from pathlib import Path

import pytest
from contrato_detector import verificar_contrato
from PIL import Image

from src.application.dtos.estimate_dto import SolicitudEstimacionDTO
from src.application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from src.domain.models.coffee_cherry import BoundingBox, DeteccionFruto, EstadoMadurez
from src.domain.models.modelo_info import FamiliaModelo, ModeloInfo
from src.domain.ports.fruit_detector_port import FruitDetectorPort
from src.infrastructure.adapters.detectors.mock_detector_adapter import MockFruitDetectorAdapter
from src.infrastructure.adapters.ui import presentacion
from src.infrastructure.adapters.ui.cli_app import IMAGEN_DEMO, run_cli
from src.infrastructure.composition import construir_detector

RAIZ = Path(__file__).resolve().parents[1]


class DetectorValidadoFalso(FruitDetectorPort):
    """Simula un modelo real ya validado (p. ej. YOLO evaluado en test)."""

    def detectar_frutos(self, imagen, umbral_confianza=0.40):
        return [DeteccionFruto("V001", EstadoMadurez.MADURO, 0.9, BoundingBox(1, 1, 10, 10))]

    def info(self):
        return ModeloInfo("YOLO de prueba", FamiliaModelo.M2_YOLO, "t", "dataset ficticio", True, "a" * 64)


def _jpeg(tam=(200, 150)):
    b = io.BytesIO()
    Image.new("RGB", tam, (30, 120, 30)).save(b, format="JPEG")
    return b.getvalue()


# ---------------------------------------------------------------- ModeloInfo
def test_modelo_sintetico_o_mock_no_puede_declararse_validado():
    for familia in (FamiliaModelo.SINTETICO, FamiliaModelo.MOCK):
        with pytest.raises(ValueError):
            ModeloInfo("x", familia, "1", "y", validado_con_datos_reales=True)


def test_hash_debe_ser_sha256():
    with pytest.raises(ValueError):
        ModeloInfo("x", FamiliaModelo.M2_YOLO, "1", "y", False, hash_pesos_sha256="abc")


@pytest.mark.parametrize(
    "familia, validado, etiqueta",
    [
        (FamiliaModelo.SINTETICO, False, "SINTÉTICO – NO VALIDADO"),
        (FamiliaModelo.MOCK, False, "SIMULADO – NO VALIDADO"),
        (FamiliaModelo.M2_YOLO, False, "NO VALIDADO CON DATOS REALES"),
        (FamiliaModelo.M2_YOLO, True, None),
    ],
)
def test_etiqueta_y_advertencias_segun_procedencia(familia, validado, etiqueta):
    info = ModeloInfo("m", familia, "1", "d", validado)
    assert info.etiqueta == etiqueta
    assert bool(info.advertencias) == (etiqueta is not None)


# ------------------------------------------------------------ contrato
def test_contrato_detector_mock():
    verificar_contrato(MockFruitDetectorAdapter(20), Image.new("RGB", (320, 240)))


def test_contrato_detector_sintetico():
    det = construir_detector("sintetico")
    assert det.es_simulado and det.info().familia is FamiliaModelo.SINTETICO
    assert det.info().hash_pesos_sha256  # trazable al archivo exacto de pesos
    verificar_contrato(det, Image.open(IMAGEN_DEMO).convert("RGB"))


def test_alias_ml_sigue_funcionando():
    assert construir_detector("ml").info().familia is FamiliaModelo.SINTETICO


# ------------------------------------------------------------ caso de uso
def test_respuesta_incluye_procedencia_y_advertencia():
    r = AnalizarImagenCosechaUseCase(MockFruitDetectorAdapter(5)).ejecutar(SolicitudEstimacionDTO(imagen_bytes=_jpeg()))
    assert r.modelo["familia"] == "MOCK" and r.modelo["es_simulado"] is True
    assert r.advertencias and not r.es_resultado_validado


def test_modelo_validado_no_genera_advertencias():
    r = AnalizarImagenCosechaUseCase(DetectorValidadoFalso()).ejecutar(SolicitudEstimacionDTO(imagen_bytes=_jpeg()))
    assert r.advertencias == [] and r.es_resultado_validado


def test_imagen_anotada_lleva_franja_de_aviso_solo_si_no_esta_validado():
    tam = (200, 150)
    no_val = AnalizarImagenCosechaUseCase(MockFruitDetectorAdapter(5)).ejecutar(SolicitudEstimacionDTO(imagen_bytes=_jpeg(tam)))
    val = AnalizarImagenCosechaUseCase(DetectorValidadoFalso()).ejecutar(SolicitudEstimacionDTO(imagen_bytes=_jpeg(tam)))
    alto_no_val = Image.open(io.BytesIO(no_val.imagen_anotada_bytes)).size[1]
    assert alto_no_val > tam[1]  # franja añadida arriba
    assert Image.open(io.BytesIO(val.imagen_anotada_bytes)).size == tam


# ------------------------------------------------------------ presentación (criterio de aceptación)
def test_mensaje_de_estado_nunca_es_ok_si_no_esta_validado():
    r = AnalizarImagenCosechaUseCase(MockFruitDetectorAdapter(5)).ejecutar(SolicitudEstimacionDTO(imagen_bytes=_jpeg()))
    msg = presentacion.mensaje_estado(r)
    assert "✅" not in msg and "NO VALIDADO" in msg and r.advertencias[0] in msg

    r_ok = AnalizarImagenCosechaUseCase(DetectorValidadoFalso()).ejecutar(SolicitudEstimacionDTO(imagen_bytes=_jpeg()))
    assert presentacion.mensaje_estado(r_ok).startswith("✅")


def test_banner_permanente_muestra_procedencia_y_aviso():
    banner = presentacion.banner_markdown(construir_detector("sintetico").info())
    assert "SINTÉTICO – NO VALIDADO" in banner and "Validado con datos reales | NO" in banner
    assert "⚠️" not in presentacion.banner_markdown(DetectorValidadoFalso().info())


@pytest.mark.parametrize("detector", ["sintetico", "mock"])
def test_cli_muestra_advertencia_y_no_dice_ok(detector, capsys):
    assert run_cli(["--detector", detector]) == 0
    salida = capsys.readouterr().out
    assert "NO VALIDADO" in salida and "Procedencia del modelo" in salida
    assert "✅" not in salida


def test_ui_y_cli_no_tienen_mensajes_de_exito_propios():
    """El '✅' solo puede salir de presentacion.mensaje_estado, que revisa las advertencias."""
    for nombre in ("gradio_app.py", "cli_app.py"):
        texto = (RAIZ / "src/infrastructure/adapters/ui" / nombre).read_text(encoding="utf-8")
        assert "✅" not in texto, nombre


def test_gradio_muestra_advertencia(monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setitem(os.environ, "SIKAFE_DETECTOR", "mock")
    from src.infrastructure.adapters.ui import gradio_app

    salida = gradio_app.procesar_estimacion(Image.new("RGB", (200, 150)), 1.8, 1.25, 1, 2500, 0.4)
    assert "✅" not in salida[1] and "NO VALIDADO" in salida[1]

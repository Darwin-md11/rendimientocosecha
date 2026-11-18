"""
Adaptador YOLO probado con un modelo falso que imita la interfaz de Ultralytics
(results[0].boxes.xyxy/conf/cls, model.names). Así se prueba sin PyTorch; con el
modelo real hay que repetir la suite de contrato en Colab (ver notebook).
"""

import json
import sys

import numpy as np
import pytest
from contrato_detector import verificar_contrato
from PIL import Image

from src.domain.exceptions.domain_exceptions import ModeloNoDisponibleException
from src.domain.models.coffee_cherry import EstadoMadurez
from src.domain.models.modelo_info import FamiliaModelo, ModeloInfo
from src.infrastructure import composition
from src.infrastructure.adapters.detectors import yolo_detector_adapter as Y
from src.infrastructure.ml.manifiesto import sha256_archivo

INFO = ModeloInfo("YOLO falso", FamiliaModelo.M2_YOLO, "t", "fixture", False)


class _Cajas:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = np.asarray(xyxy, np.float32).reshape(-1, 4), np.asarray(conf, np.float32), np.asarray(cls, np.float32)

    def __len__(self):
        return len(self.conf)


class _Resultado:
    def __init__(self, cajas):
        self.boxes = cajas


class ModeloFalso:
    """Devuelve cajas en coordenadas relativas escaladas al tamaño de la imagen; filtra conf > umbral como Ultralytics."""

    def __init__(self, names=None, cajas_rel=None, confs=None, clases=None):
        self.names = names if names is not None else {0: "VERDE", 1: "PINTON", 2: "MADURO", 3: "SOBREMADURO"}
        self.cajas_rel = (
            cajas_rel if cajas_rel is not None else [[0.1, 0.1, 0.2, 0.2], [0.4, 0.4, 0.5, 0.5], [0.6, 0.1, 0.7, 0.25], [0.8, 0.8, 0.95, 0.95]]
        )
        self.confs = confs if confs is not None else [0.9, 0.7, 0.5, 0.3]
        self.clases = clases if clases is not None else [0, 1, 2, 3]
        self.llamadas = []

    def predict(self, source, **kw):
        self.llamadas.append((source, kw))
        w, h = source.size
        escala = np.array([w, h, w, h])
        sel = [i for i, c in enumerate(self.confs) if np.float32(c) > kw["conf"]]
        return [_Resultado(_Cajas([np.array(self.cajas_rel[i]) * escala for i in sel], [self.confs[i] for i in sel], [self.clases[i] for i in sel]))]


def test_contrato_con_modelo_falso():
    verificar_contrato(Y.YoloFruitDetectorAdapter(ModeloFalso(), INFO), Image.new("RGB", (640, 480)))


def test_entrega_pil_rgb_sin_exif_y_nms_agnostico():
    modelo = ModeloFalso()
    det = Y.YoloFruitDetectorAdapter(modelo, INFO, imgsz=960, iou=0.5)
    img = Image.new("L", (320, 240))
    exif = Image.Exif()
    exif[0x0112] = 6  # orientación: rotar 90°
    img.info["exif"] = exif.tobytes()
    det.detectar_frutos(img, 0.4)
    source, kw = modelo.llamadas[-1]
    assert isinstance(source, Image.Image) and source.mode == "RGB" and "exif" not in source.info
    assert len(source.getexif()) == 0
    assert kw["agnostic_nms"] is True and kw["imgsz"] == 960 and kw["iou"] == 0.5 and kw["verbose"] is False


def test_umbral_con_tolerancia_float32():
    det = Y.YoloFruitDetectorAdapter(ModeloFalso(confs=[0.7, 0.69], cajas_rel=[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]], clases=[2, 2]), INFO)
    dets = det.detectar_frutos(Image.new("RGB", (100, 100)), 0.7)
    assert len(dets) == 1 and dets[0].estado is EstadoMadurez.MADURO  # float32(0.7) < 0.7 pero se conserva


def test_cajas_fuera_vacias_o_no_finitas():
    modelo = ModeloFalso(
        cajas_rel=[[-0.2, -0.1, 0.3, 0.3], [0.5, 0.5, 0.5, 0.6], [np.nan, 0.1, 0.2, 0.2], [0.9, 0.9, 1.3, 1.2]],
        confs=[0.9, 0.9, 0.9, 0.9],
        clases=[0, 1, 2, 3],
    )
    dets = Y.YoloFruitDetectorAdapter(modelo, INFO).detectar_frutos(Image.new("RGB", (100, 100)), 0.1)
    assert len(dets) == 2  # la vacía y la NaN se descartan
    for d in dets:
        assert 0 <= d.bbox.x_min < d.bbox.x_max <= 100 and 0 <= d.bbox.y_min < d.bbox.y_max <= 100


def test_traduce_nombres_de_clase_y_falla_si_ninguno_sirve():
    assert Y.mapa_clases(["Verde", "pintón", "Maduro", "sobre-maduro", "hoja"]) == {
        0: EstadoMadurez.VERDE,
        1: EstadoMadurez.PINTON,
        2: EstadoMadurez.MADURO,
        3: EstadoMadurez.SOBREMADURO,
    }
    with pytest.raises(ModeloNoDisponibleException, match="Ninguna clase"):
        Y.YoloFruitDetectorAdapter(ModeloFalso(names={0: "person", 1: "car"}), INFO)


def test_clases_no_traducibles_se_ignoran():
    modelo = ModeloFalso(names={0: "VERDE", 1: "hoja"}, clases=[0, 1, 0, 1])
    dets = Y.YoloFruitDetectorAdapter(modelo, INFO).detectar_frutos(Image.new("RGB", (100, 100)), 0.1)
    assert {d.estado for d in dets} == {EstadoMadurez.VERDE} and len(dets) == 2


# ------------------------------------------------------------------ carga desde archivo
def _publicar(tmp_path, **extra):
    pesos = tmp_path / "m2_yolo.pt"
    pesos.write_bytes(b"pesos falsos")
    man = {
        "archivo": pesos.name,
        "sha256": sha256_archivo(pesos),
        "clases": ["VERDE", "PINTON", "MADURO", "SOBREMADURO"],
        "nombre": "YOLO11n Croppie",
        "version_framework": "8.x",
        "imgsz": 640,
        "datos_entrenamiento": "Croppie train",
        "validado_con_datos_reales": True,
        "dominio_validacion": "Croppie test (Uganda)",
        **extra,
    }
    (tmp_path / "m2_yolo.manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return pesos


def test_hash_incorrecto_no_carga_ni_importa_ultralytics(tmp_path, monkeypatch):
    pesos = _publicar(tmp_path)
    pesos.write_bytes(b"otro archivo")
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # si se intentara importar, fallaría distinto
    with pytest.raises(ModeloNoDisponibleException, match="SHA-256"):
        Y.YoloFruitDetectorAdapter.desde_archivo(pesos)


def test_sin_ultralytics_mensaje_claro(tmp_path, monkeypatch):
    pesos = _publicar(tmp_path)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ModeloNoDisponibleException, match="requirements-ml.txt"):
        Y.YoloFruitDetectorAdapter.desde_archivo(pesos)


def test_carga_con_ultralytics_simulado_y_procedencia(tmp_path, monkeypatch):
    pesos = _publicar(tmp_path)

    class _Mod:
        @staticmethod
        def YOLO(ruta):
            assert ruta == str(pesos)
            return ModeloFalso()

    monkeypatch.setitem(sys.modules, "ultralytics", _Mod)
    det = Y.YoloFruitDetectorAdapter.desde_archivo(pesos)
    info = det.info()
    assert info.familia is FamiliaModelo.M2_YOLO and info.validado_con_datos_reales and not info.validado_en_dominio_objetivo
    assert "OTRO DOMINIO" in info.etiqueta and info.hash_pesos_sha256 == sha256_archivo(pesos)


def test_clases_del_modelo_distintas_al_manifiesto(tmp_path, monkeypatch):
    pesos = _publicar(tmp_path, clases=["VERDE", "MADURO"])

    class _Mod:
        YOLO = staticmethod(lambda ruta: ModeloFalso())

    monkeypatch.setitem(sys.modules, "ultralytics", _Mod)
    with pytest.raises(ModeloNoDisponibleException, match="no coinciden"):
        Y.YoloFruitDetectorAdapter.desde_archivo(pesos)


def test_composicion_elige_yolo_solo_si_hay_modelo_publicado(tmp_path, monkeypatch):
    monkeypatch.delenv("SIKAFE_DETECTOR", raising=False)
    monkeypatch.setenv("SIKAFE_MODELO_YOLO", str(tmp_path / "m2_yolo.pt"))
    assert composition.detector_por_defecto() == "sintetico"
    _publicar(tmp_path)
    assert composition.detector_por_defecto() == "yolo"
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ModeloNoDisponibleException):  # pedido yolo y no carga: error, nunca el sintético
        composition.construir_detector()

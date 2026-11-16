"""
Evaluación de M2 de punta a punta con un Ultralytics simulado: dataset de juguete
preparado con ml.datos.preparar_yolo, corrida con manifiesto, métricas y regla del test único.
"""

import json
import sys

import pytest
import yaml
from test_preparar_yolo import croppie_juguete  # noqa: F401  (fixture)
from test_yolo_adapter import ModeloFalso

from ml.datos import preparar_yolo
from ml.experiments.registro import capturar_entorno
from ml.m2_yolo import entrenar, evaluar
from src.infrastructure.adapters.detectors.yolo_detector_adapter import YoloFruitDetectorAdapter

# Cajas del dataset de juguete en xyxy relativo y su clase SIKAFE (VERDE, MADURO, PINTON, SOBREMADURO)
CAJAS_JUGUETE = [[0.15, 0.15, 0.25, 0.25], [0.45, 0.45, 0.55, 0.55], [0.65, 0.25, 0.75, 0.35], [0.25, 0.65, 0.35, 0.75]]
CLASES_JUGUETE = [0, 2, 1, 3]


@pytest.fixture
def corrida(croppie_juguete, tmp_path, monkeypatch):  # noqa: F811
    datos = tmp_path / "yolo"
    preparar_yolo.preparar(croppie_juguete, datos, "croppie", semilla=0)
    corrida = tmp_path / "corridas" / "m2_prueba"
    corrida.mkdir(parents=True)
    cfg = {
        "nombre": "m2_prueba",
        "modelo_base": "yolo11n.pt",
        "imgsz": 640,
        "semilla": 0,
        "iou_nms": 0.6,
        "conf_reporte": 0.25,
        "datos": str(datos / "data.yaml"),
    }
    (corrida / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (corrida / "m2_yolo.pt").write_bytes(b"pesos de prueba")
    entrenar.escribir_manifiesto_app(
        corrida, cfg, datos / "data.yaml", ["VERDE", "PINTON", "MADURO", "SOBREMADURO"], capturar_entorno(datos / "data.yaml")
    )

    class _Ultralytics:  # el detector "perfecto" predice exactamente las cajas anotadas
        YOLO = staticmethod(lambda ruta: ModeloFalso(cajas_rel=CAJAS_JUGUETE, confs=[0.9, 0.8, 0.7, 0.6], clases=CLASES_JUGUETE))

    monkeypatch.setitem(sys.modules, "ultralytics", _Ultralytics)
    return corrida


def test_manifiesto_de_la_corrida(corrida):
    man = json.loads((corrida / "m2_yolo.manifest.json").read_text(encoding="utf-8"))
    assert man["familia"] == "M2" and man["validado_con_datos_reales"] is False and man["clases"][1] == "PINTON"
    assert "croppie" in man["datos_entrenamiento"] and "CC BY-SA" in man["datos_entrenamiento"]


def test_detector_perfecto_da_metricas_perfectas(corrida):
    detector = YoloFruitDetectorAdapter.desde_archivo(corrida / "m2_yolo.pt")
    met, filas, _ = evaluar.evaluar(detector, corrida.parent.parent / "yolo" / "data.yaml", "val")
    assert met["map50"] == pytest.approx(1.0) and met["map50_95"] > 0.9
    assert met["deteccion"]["con_clase"]["f1"] == pytest.approx(1.0)
    assert met["conteo"]["total"]["mae"] == 0 and all(f["conteo_anotado"] == 4 for f in filas)
    assert met["matriz_confusion"]["valores"][1][1] == met["imagenes"]  # PINTON bien clasificado en todas


def test_test_solo_una_vez_y_manifiesto_validado(corrida):
    with pytest.raises(SystemExit, match="--final"):
        evaluar.main(["--corrida", str(corrida), "--split", "test"])
    assert evaluar.main(["--corrida", str(corrida), "--split", "test", "--final"]) == 0
    man = json.loads((corrida / "m2_yolo.manifest.json").read_text(encoding="utf-8"))
    assert man["validado_con_datos_reales"] is True and man["validado_en_dominio_objetivo"] is False
    assert "croppie partición test" in man["dominio_validacion"] and man["metricas_test"]["map50"] == pytest.approx(1.0)
    for archivo in (
        "metricas_test.json",
        "resumen_test.md",
        "conteos_test.csv",
        "predicciones_test.jsonl",
        "matriz_confusion_test.png",
        "dispersion_conteo_test.png",
    ):
        assert (corrida / archivo).is_file(), archivo
    with pytest.raises(SystemExit, match="ya se usó"):
        evaluar.main(["--corrida", str(corrida), "--split", "test", "--final"])
    # El manifiesto se cambió después de publicar: el hash de los pesos no cambia, así que la app sigue cargándolo.
    info = YoloFruitDetectorAdapter.desde_archivo(corrida / "m2_yolo.pt").info()
    assert info.etiqueta == "VALIDADO SOLO EN OTRO DOMINIO"

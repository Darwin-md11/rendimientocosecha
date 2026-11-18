"""Generador del informe e intervalos de confianza de M2 (bootstrap)."""

import csv
import json
import sys

import numpy as np
import pytest
import yaml
from test_m0_clasico import datos_color  # noqa: F401  (fixture)
from test_preparar_yolo import croppie_juguete  # noqa: F401  (fixture)
from test_yolo_adapter import ModeloFalso

from ml.datos import preparar_yolo
from ml.experiments.registro import capturar_entorno
from ml.informe import bootstrap_m2 as B
from ml.informe import generar as G
from ml.m0_clasico import entrenar as M0
from ml.m2_yolo import entrenar as M2E
from ml.m2_yolo import evaluar as M2V
from src.evaluation import metricas as M

# Detector imperfecto sobre el dataset de juguete: acierta VERDE y MADURO, confunde PINTÓN con MADURO, pierde SOBREMADURO
# y añade un falso positivo (cajas en coordenadas relativas, como CAJAS_JUGUETE de test_evaluar_m2)
CAJAS = [[0.15, 0.15, 0.25, 0.25], [0.45, 0.45, 0.55, 0.55], [0.65, 0.25, 0.75, 0.35], [0.85, 0.05, 0.95, 0.12]]
CONFS = [0.9, 0.8, 0.6, 0.3]
CLASES = [0, 2, 2, 1]


def test_ap_ponderada_igual_a_la_comun_y_a_repetir():
    rng = np.random.default_rng(0)
    conf = rng.random(200)
    tp = rng.random(200) < 0.6
    orden = np.argsort(-conf, kind="stable")
    assert B.ap_ponderada(tp[orden], np.ones(200, int), 150) == pytest.approx(M.average_precision(conf, tp, 150))
    w = rng.integers(0, 4, 200)
    repetido = M.average_precision(np.repeat(conf, w), np.repeat(tp, w), 150 * 2)
    assert B.ap_ponderada(tp[orden], w[orden], 150 * 2) == pytest.approx(repetido)


@pytest.fixture
def corrida_m2(croppie_juguete, tmp_path, monkeypatch):  # noqa: F811
    datos = tmp_path / "yolo"
    preparar_yolo.preparar(croppie_juguete, datos, "croppie", semilla=0)
    corrida = tmp_path / "corridas" / "m2_prueba"
    corrida.mkdir(parents=True)
    cfg = {
        "nombre": "m2_prueba",
        "modelo_base": "yolo11n.pt",
        "imgsz": 640,
        "epochs": 2,
        "patience": 1,
        "batch": 2,
        "semilla": 0,
        "iou_nms": 0.6,
        "conf_reporte": 0.25,
        "datos": str(datos / "data.yaml"),
    }
    (corrida / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (corrida / "m2_yolo.pt").write_bytes(b"pesos")
    entorno = capturar_entorno(datos / "data.yaml")
    (corrida / "entorno.json").write_text(json.dumps(entorno), encoding="utf-8")
    M2E.escribir_manifiesto_app(corrida, cfg, datos / "data.yaml", ["VERDE", "PINTON", "MADURO", "SOBREMADURO"], entorno)
    (corrida / "train").mkdir()
    with open(corrida / "train" / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "time", "metrics/mAP50(B)", "metrics/mAP50-95(B)"])
        w.writerows([[1, 10.0, 0.5, 0.3], [2, 20.0, 0.6, 0.4]])

    class _U:
        YOLO = staticmethod(lambda ruta: ModeloFalso(cajas_rel=CAJAS, confs=CONFS, clases=CLASES))

    monkeypatch.setitem(sys.modules, "ultralytics", _U)
    for split, final in (("val", []), ("test", ["--final"])):
        assert M2V.main(["--corrida", str(corrida), "--split", split, *final]) == 0
    return corrida, datos / "data.yaml"


def test_bootstrap_sin_remuestreo_reproduce_las_metricas_registradas(corrida_m2):
    corrida, data_yaml = corrida_m2
    reg = json.loads((corrida / "metricas_test.json").read_text(encoding="utf-8"))
    pc = B.precalcular(B.leer_predicciones(corrida / "predicciones_test.jsonl"), data_yaml, "test")
    p = B.metricas_ponderadas(pc, np.ones(len(pc.imagenes), int))
    assert 0 < reg["map50"] < 1  # no trivial
    assert p["map50"] == pytest.approx(reg["map50"], abs=1e-6)
    assert p["f1_con_clase"] == pytest.approx(reg["deteccion"]["con_clase"]["f1"])
    assert p["f1_localizacion"] == pytest.approx(reg["deteccion"]["localizacion"]["f1"])
    assert p["exactitud_cajas_emparejadas"] == pytest.approx(reg["clasificacion_en_cajas_emparejadas"]["exactitud"])
    assert p["conteo_mae"] == pytest.approx(reg["conteo"]["total"]["mae"])
    ic = B.intervalos(pc, n=200, nivel="sesion")
    for k in ("map50", "f1_con_clase", "conteo_mae"):
        assert ic[k]["li"] <= ic[k]["punto"] + 1e-9 and ic[k]["punto"] <= ic[k]["ls"] + 1e-9
    assert ic["_meta"]["nivel"] == "sesion" and ic["_meta"]["sesiones"] >= 1


def test_generar_informe_completo(corrida_m2, datos_color, tmp_path):  # noqa: F811
    corrida_m2_dir, data_yaml = corrida_m2
    base = corrida_m2_dir.parent
    ids = {"m2": corrida_m2_dir.name}
    for modelo in ("rf", "svm"):
        c = M0.entrenar(datos_color / "yolo" / "data.yaml", base, modelo, max_train=2000)
        M0.evaluar_final(c)
        ids[f"m0_{modelo}"] = c.name
    for clave, f1 in (("m1", 0.8), ("m1_sin_preentrenar", 0.7)):  # M1 simulado: sin PyTorch aquí
        c = base / f"m1_{clave}"
        c.mkdir()
        (c / "config.yaml").write_text(
            yaml.safe_dump({"arquitectura": "mobilenet_v2", "preentrenado": clave == "m1", "lado": 96, "epocas_cabeza": 2, "epocas": 10, "lr": 3e-4}),
            encoding="utf-8",
        )
        (c / "tiempos.json").write_text(json.dumps({"entrenamiento_s": 600, "gpu": True}), encoding="utf-8")
        (c / "historial.csv").write_text("etapa,epoca,f1_macro_val\ncabeza,1,0.5\ncompleta,1,0.7\n", encoding="utf-8")
        for split in ("val", "test"):
            met = {
                "exactitud": 0.9,
                "f1_macro": f1,
                "por_clase": {str(i): {"f1": f1} for i in range(4)},
                "ms_por_recorte_prediccion": 0.2,
                "dispositivo_prediccion": "cuda",
            }
            (c / f"metricas_{split}.json").write_text(json.dumps(met), encoding="utf-8")
        ids[clave] = c.name
    cfg = yaml.safe_load((G.RAIZ / "configs" / "informe.yaml").read_text(encoding="utf-8"))
    cfg["corridas"] = ids
    ruta_cfg = tmp_path / "informe.yaml"
    ruta_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "cuda_m0_val_20000.md").write_text("# CUDA en M0\n\n| a | b |\n", encoding="utf-8")

    salida = tmp_path / "informe"
    md = G.generar(base, data_yaml, salida, bench, ruta_cfg, n_boot=100).read_text(encoding="utf-8")
    for esperado in (
        "## 8.2 Datos",
        "## 8.3 Modelos",
        "## 8.4 Evaluación",
        "Tabla 8.4.2",
        "IC 95 % (sesiones)",
        "Tabla 8.4.3",
        "Transfer learning mejora",
        "#### CUDA en M0",
        "## Trazabilidad",
        "M0 — color+textura + SVM",
    ):
        assert esperado in md, esperado
    assert md.count("| M1 — MobileNetV2") >= 2
    for fig in ("8_2_proporcion_clases.png", "8_3_curva_m1.png", "8_3_curva_m2.png", "8_4_f1_por_clase.png", "8_4_matriz_confusion_m0_rf.png"):
        assert (salida / "figuras" / fig).is_file(), fig
    tablas = json.loads((salida / "tablas.json").read_text(encoding="utf-8"))
    assert "m2_intervalos" in tablas and set(tablas["m2_intervalos"]) == {"imagen", "sesion"}


def test_generar_falla_si_falta_una_corrida(tmp_path):
    with pytest.raises(SystemExit, match="No se encuentran"):
        G.generar(tmp_path, tmp_path / "data.yaml", tmp_path / "out", n_boot=0)

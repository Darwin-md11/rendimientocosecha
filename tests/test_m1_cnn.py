"""
M1 (CNN con transfer learning sobre recortes). Sin PyTorch se prueba todo lo que no lo necesita:
la caché de recortes (mismas cerezas y clases que M0), los pesos por clase y la importación diferida.
El entrenamiento real se prueba solo si PyTorch está instalado (en Colab: celda --rapido del notebook).
"""

import importlib
import json
import sys

import numpy as np
import pytest
import yaml
from test_m0_clasico import datos_color  # noqa: F401  (fixture)

from ml.m0_clasico import entrenar as M0
from ml.m1_cnn import recortes as RC


def test_el_modulo_de_entrenamiento_no_exige_pytorch_al_importarse(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torchvision", None)
    sys.modules.pop("ml.m1_cnn.entrenar", None)
    modulo = importlib.import_module("ml.m1_cnn.entrenar")
    assert modulo.RAPIDO["epocas"] == 1
    sys.modules.pop("ml.m1_cnn.entrenar", None)


def test_pesos_por_clase_a_mano():
    y = np.array([0] * 6 + [1] * 3 + [2] * 3)  # 12 recortes; la clase 3 no aparece
    w = RC.pesos_por_clase(y, 4)
    assert w == pytest.approx([12 / (4 * 6), 12 / (4 * 3), 12 / (4 * 3), 0.0])


def test_cache_de_recortes_mismas_cerezas_que_m0(datos_color, tmp_path):  # noqa: F811
    datos = datos_color / "yolo" / "data.yaml"
    c = RC.construir_cache(datos, "train", tmp_path, lado=32)
    X, y, g = RC.cargar(c)
    assert c["cache"] is False and X.shape == (c["n"], 32, 32, 3) and X.dtype == np.uint8
    m0 = M0.extraer_split(datos, "train")
    assert np.array_equal(y, m0["y"]) and np.array_equal(g, m0["grupos"])  # mismas cerezas, en el mismo orden
    assert X.max() > 0 and not list(tmp_path.glob("*.parcial*"))
    assert RC.construir_cache(datos, "train", tmp_path, lado=32)["cache"] is True
    # El color del recorte corresponde a su clase (el rojo domina en MADURO)
    maduro = X[y == 2].reshape(-1, 3).mean(0)
    assert maduro[0] > maduro[1] and maduro[0] > maduro[2]


def test_limitar_indices():
    from ml.m1_cnn.entrenar import _limitar

    assert len(_limitar(100, 10, 0)) == 10 and np.array_equal(_limitar(100, 10, 0), _limitar(100, 10, 0))
    assert len(_limitar(5, 10, 0)) == 5 and len(_limitar(5, None, 0)) == 5


def test_entrenamiento_rapido_con_pytorch(datos_color, tmp_path):  # noqa: F811
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ml.m1_cnn import entrenar as M1

    cfg = yaml.safe_load(open("ml/experiments/configs/m1_mobilenetv2_96.yaml", encoding="utf-8"))
    cfg.update({"preentrenado": False, "lado": 32, "batch": 16, "workers": 0, "amp": False})
    ruta_cfg = tmp_path / "cfg.yaml"
    ruta_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    corrida = M1.entrenar(ruta_cfg, datos_color / "yolo" / "data.yaml", tmp_path / "corridas", rapido=True)
    met = json.loads((corrida / "metricas_val.json").read_text(encoding="utf-8"))
    assert met["recortes"] > 0 and (corrida / "historial.csv").is_file() and (corrida / "modelo_m1.manifest.json").is_file()
    assert "partición `test`" in M1.evaluar_final(corrida)
    with pytest.raises(SystemExit, match="ya se usó"):
        M1.evaluar_final(corrida)

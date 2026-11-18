"""Métricas puras contra ejemplos calculados a mano."""
import math

import numpy as np
import pytest

from src.evaluation import metricas as M


def test_iou_a_mano():
    # Cajas 10x10 desplazadas 5 en x: intersección 50, unión 150 -> 1/3
    assert M.iou_matriz([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)
    assert M.iou_matriz([[0, 0, 10, 10]], [[20, 20, 30, 30]])[0, 0] == 0
    assert M.iou_matriz(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)


def test_emparejamiento_greedy_por_confianza():
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 10], [1, 0, 11, 10]]  # dos predicciones sobre la misma cereza
    pares = M.emparejar(pred, [0.6, 0.9], gt, 0.5)
    assert pares == [(1, 0)]  # gana la de mayor confianza; la otra es falso positivo


def test_average_precision_a_mano():
    # 1 gt; predicciones [FP 0.9, TP 0.8]: precisión 0,5 en todo el recall -> AP 0,5
    assert M.average_precision([0.9, 0.8], [False, True], 1) == pytest.approx(0.5)
    # 2 gt; [TP .9, FP .8, TP .7]: recall .5,.5,1; precisión 1,.5,.667; envolvente 1 hasta r=.5 y 2/3 después.
    # 101 puntos: 51 (r=0..0.50) con 1 y 50 con 2/3 -> (51 + 50*2/3)/101
    esperado = (51 + 50 * 2 / 3) / 101
    assert M.average_precision([0.9, 0.8, 0.7], [True, False, True], 2) == pytest.approx(esperado)
    assert M.average_precision([], [], 3) == 0.0
    assert math.isnan(M.average_precision([0.5], [False], 0))


def _im(pred, conf, cls_p, gt, cls_g):
    return M.ImagenEval.crear(pred, conf, cls_p, gt, cls_g)


def test_deteccion_y_conteo_de_un_caso_pequeno():
    # Imagen 1: 2 gt (VERDE=0, MADURO=2). Pred: una acierta VERDE, otra cae en MADURO pero dice PINTON (1), y un FP.
    im1 = _im([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]], [0.9, 0.8, 0.7], [0, 1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]], [0, 2])
    # Imagen 2: 1 gt (SOBREMADURO=3) no detectado; una predicción con confianza bajo el umbral.
    im2 = _im([[0, 0, 5, 5]], [0.1], [3], [[40, 40, 50, 50]], [3])
    r = M.metricas_deteccion([im1, im2], umbral_iou=0.5, umbral_conf=0.25)
    # Localización (sin clase): TP=2, FP=1, FN=1
    assert (r["localizacion"]["tp"], r["localizacion"]["fp"], r["localizacion"]["fn"]) == (2, 1, 1)
    assert r["localizacion"]["precision"] == pytest.approx(2 / 3) and r["localizacion"]["recall"] == pytest.approx(2 / 3)
    # Con clase: solo el VERDE es TP -> TP=1, FP=2, FN=2
    assert (r["con_clase"]["tp"], r["con_clase"]["fp"], r["con_clase"]["fn"]) == (1, 2, 2)
    assert r["por_clase"][2]["fn"] == 1 and r["por_clase"][1]["fp"] == 1

    m = M.matriz_confusion_deteccion([im1, im2], n_clases=4)
    assert m[0, 0] == 1  # VERDE bien
    assert m[2, 1] == 1  # MADURO predicho como PINTON
    assert m[3, 4] == 1  # SOBREMADURO no detectado (fondo)
    assert m[4, 2] == 1  # FP predicho MADURO sobre fondo
    assert m.sum() == 4


def test_map_detector_perfecto_y_umbrales_coco():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    im = _im(gt, [0.9, 0.8], [0, 1], gt, [0, 1])
    r = M.mean_average_precision([im], n_clases=4, umbrales_iou=M.UMBRALES_COCO)
    assert r["map"] == pytest.approx(1.0)
    assert r["ap_por_clase"][2] == [None] * 10  # clase sin gt: no cuenta en la media


def test_map_baja_con_iou_exigente():
    # Predicción con IoU 0,68 respecto al gt: vale a 0,50..0,65 (4 de 10 umbrales), no a 0,70+
    im = _im([[0, 0, 10, 10]], [0.9], [0], [[0, 0, 10, 14.7]], [0])
    iou = M.iou_matriz([[0, 0, 10, 10]], [[0, 0, 10, 14.7]])[0, 0]
    assert 0.65 < iou < 0.70
    r = M.mean_average_precision([im], n_clases=1, umbrales_iou=M.UMBRALES_COCO)
    assert r["map_por_umbral"][0] == pytest.approx(1.0) and r["map"] == pytest.approx(0.4)


def test_clasificacion_a_mano():
    r = M.metricas_clasificacion([0, 0, 1, 1, 2, 3], [0, 1, 1, 1, 2, 2], n_clases=4)
    assert r["exactitud"] == pytest.approx(4 / 6)
    assert r["por_clase"][1]["precision"] == pytest.approx(2 / 3) and r["por_clase"][1]["recall"] == 1.0
    assert r["por_clase"][3]["recall"] == 0.0
    assert r["matriz_confusion"][3][2] == 1


def test_conteo_a_mano():
    manual, auto = [10, 20, 30], [12, 18, 33]
    r = M.metricas_conteo(manual, auto)
    # errores 2, -2, 3 -> MAE 7/3, RMSE sqrt(17/3), sesgo (2-2+3)/3 = 1; SS_res 17, SS_tot 200 -> R² 0,915
    assert r["mae"] == pytest.approx(7 / 3) and r["rmse"] == pytest.approx(math.sqrt(17 / 3))
    assert r["sesgo_medio"] == pytest.approx(1.0) and r["r2"] == pytest.approx(1 - 17 / 200)
    assert 0 < r["r2_pearson"] <= 1
    # Un conteo que siempre duplica correlaciona perfecto (r² Pearson = 1) pero su R² es malo: por eso se reportan ambos.
    r2 = M.metricas_conteo([10, 20, 30], [20, 40, 60])
    assert r2["r2_pearson"] == pytest.approx(1.0) and r2["r2"] < 0
    with pytest.raises(ValueError):
        M.metricas_conteo([1, 2], [1])


def test_clasificacion_desde_matriz_coincide_con_etiquetas():
    a = M.metricas_clasificacion([0, 0, 1, 1, 2, 3], [0, 1, 1, 1, 2, 2], n_clases=4)
    b = M.metricas_clasificacion_desde_matriz(a["matriz_confusion"])
    assert a == b
    # Desde la matriz de detección: solo las cajas emparejadas (sin "fondo")
    det = np.array([[5, 1, 0, 0, 2], [0, 3, 0, 0, 1], [0, 1, 4, 0, 0], [0, 0, 1, 2, 3], [1, 0, 0, 0, 0]])
    r = M.metricas_clasificacion_desde_matriz(det[:4, :4])
    assert r["exactitud"] == pytest.approx(14 / 17) and r["por_clase"][3]["recall"] == pytest.approx(2 / 3)

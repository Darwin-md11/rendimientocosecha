"""
CUDA en M0 y M2. Sin GPU no se puede ejecutar el kernel; lo que sí se prueba aquí:
  - que la función que se compila para la GPU (descriptor_secuencial) da lo mismo que la referencia numpy,
  - que el reparto "un hilo por recorte" (simulado en CPU) escribe cada resultado en su fila,
  - que la verificación de equivalencia detecta diferencias,
  - que los scripts de comparación funcionan y reportan la ausencia de GPU sin fallar.
La ejecución real en GPU se valida en Colab con ml.benchmark.cuda_m0 (sale con error si no hay equivalencia).
"""

import json
import sys

import numpy as np
import pytest
from test_m0_clasico import datos_color  # noqa: F401  (fixture)
from test_yolo_adapter import ModeloFalso

from ml.benchmark import cuda_m0, yolo_cpu_gpu
from ml.m0_clasico import caracteristicas as C
from ml.m0_clasico import descriptor as D
from ml.m0_clasico import descriptor_cuda as DC


def _recortes_variados(n_ruido=6):
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:48, 0:48]
    disco = np.zeros((48, 48, 3), np.uint8)
    m = (yy - 24) ** 2 + (xx - 24) ** 2 < 300
    disco[m], disco[~m] = (190, 30, 30), (40, 90, 40)
    casos = [
        np.full((48, 48, 3), 128, np.uint8),  # uniforme: empates en LBP, saturación 0
        np.zeros((48, 48, 3), np.uint8),
        np.full((48, 48, 3), 255, np.uint8),
        np.stack([xx * 5, yy * 5, (xx + yy) * 2], -1).clip(0, 255).astype(np.uint8),  # gradientes
        disco,
    ]
    casos += [rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for _ in range(n_ruido)]
    return np.stack(casos)


def test_funcion_del_kernel_igual_a_la_referencia():
    imgs = _recortes_variados()
    ref = D.descriptor_lote_numpy(imgs)
    seq = np.stack([D.descriptor_secuencial_cpu(i) for i in imgs])
    hist = np.r_[0:96, 108:118]
    assert np.array_equal(ref[:, hist], seq[:, hist])  # histogramas idénticos (aritmética entera o mismo redondeo)
    assert np.allclose(ref, seq, rtol=0, atol=1e-9)  # resto: solo difiere el orden de las sumas
    assert ref.shape == (len(imgs), D.DIMENSION) == (len(imgs), 442)


def test_propiedades_del_descriptor():
    imgs = _recortes_variados()
    ref = D.descriptor_lote_numpy(imgs)
    for a, b in [(0, 16), (16, 32), (32, 48), (48, 64), (108, 118)]:
        assert np.allclose(ref[:, a:b].sum(1), 1.0)  # histogramas normalizados
    assert ref[:, 118:].min() >= 0 and ref[:, 118:].max() <= 1.0  # HOG L2-Hys
    assert np.all(ref[0, 118:] == 0)  # imagen uniforme: sin gradiente
    assert ref[1, D.O_LBP + 8] == pytest.approx(1.0)  # todo negro: los 8 vecinos >= centro -> código 8


def test_malla_de_hilos_simulada_y_verificacion():
    imgs = _recortes_variados()
    ref = D.descriptor_lote_numpy(imgs)
    sim = DC.simular_malla_en_cpu(imgs)
    eq = cuda_m0.verificar_equivalencia(ref, sim)
    assert eq["equivalente"] and eq["fraccion_recortes_con_algun_cambio_de_bin"] == 0
    malo = sim.copy()
    malo[3, 200] += 1e-3  # un error en HOG
    assert not cuda_m0.verificar_equivalencia(ref, malo)["equivalente"]
    malo = sim.copy()
    malo[:, 5] += 1 / 2304  # todos los recortes con un píxel en otro bin
    assert not cuda_m0.verificar_equivalencia(ref, malo)["equivalente"]


@pytest.mark.skipif(DC.cuda_disponible(), reason="hay GPU: este caso prueba el mensaje sin GPU")
def test_sin_gpu_mensaje_claro():
    with pytest.raises(DC.CudaNoDisponible):
        C.descriptor_lote(_recortes_variados()[:2], dispositivo="gpu")


def test_benchmark_m0_solo_cpu(datos_color, tmp_path):  # noqa: F811
    imgs, t = cuda_m0.recolectar_recortes(datos_color / "yolo" / "data.yaml", "train", n=50)
    assert imgs.shape == (50, 48, 48, 3) and t > 0
    res = cuda_m0.comparar(imgs, n_por_recorte=20, repeticiones=1, lote=16)
    assert res["cpu_lote"]["ms_por_recorte"] > 0
    if not DC.cuda_disponible():
        assert res["gpu"]["disponible"] is False
        assert "GPU no disponible" in cuda_m0.tabla_md(res, {"plataforma": "prueba"})
    assert (
        cuda_m0.main(
            ["--datos", str(datos_color / "yolo" / "data.yaml"), "--split", "train", "--n", "40", "--repeticiones", "1", "--salida", str(tmp_path)]
        )
        == 0
    )
    guardado = json.loads((tmp_path / "cuda_m0_train_40.json").read_text(encoding="utf-8"))
    assert guardado["n_recortes"] == 40 and (tmp_path / "cuda_m0_train_40.md").is_file()


def test_benchmark_yolo_pasa_el_dispositivo_al_modelo(tmp_path, monkeypatch):
    from test_yolo_adapter import _publicar

    pesos = _publicar(tmp_path)
    creados = []

    class _Ultralytics:
        @staticmethod
        def YOLO(ruta):
            creados.append(ModeloFalso())
            return creados[-1]

    monkeypatch.setitem(sys.modules, "ultralytics", _Ultralytics)
    from PIL import Image

    imgs = [Image.new("RGB", (64, 48)) for _ in range(8)]
    monkeypatch.setattr(yolo_cpu_gpu, "gpu_disponible", lambda: False)
    res = yolo_cpu_gpu.comparar(pesos, imgs, calentamiento=2)
    assert res["gpu"] is None and creados[-1].llamadas[-1][1]["device"] == "cpu"
    assert "GPU no disponible" in yolo_cpu_gpu.tabla_md(res, {"plataforma": "prueba"})

    monkeypatch.setattr(yolo_cpu_gpu, "gpu_disponible", lambda: True)
    res = yolo_cpu_gpu.comparar(pesos, imgs, calentamiento=2)
    assert creados[-1].llamadas[-1][1]["device"] == 0 and res["consistencia"]["pct_imagenes_mismo_conteo"] == 100.0
    assert "Aceleración" in yolo_cpu_gpu.tabla_md(res, {"plataforma": "prueba", "gpu": "GPU de prueba"})

"""M0 (color + textura + RF/SVM sobre recortes) con un dataset de juguete donde el color define la clase."""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ml.datos import preparar_yolo
from ml.m0_clasico import caracteristicas as C
from ml.m0_clasico import entrenar as E

# id Croppie -> color RGB típico (dark_brown, green, yellow, red); low_visibility no se usa
COLORES = {0: (70, 40, 25), 1: (60, 160, 60), 4: (225, 200, 60), 3: (190, 30, 30)}
POSICIONES = [(0.2, 0.3), (0.45, 0.6), (0.7, 0.3), (0.85, 0.75)]


def _tile(ruta_img, ruta_lbl, rng, ancho=96, alto=72):
    fondo = rng.integers(20, 90, (alto, ancho, 3), dtype=np.uint8)  # follaje oscuro con ruido
    img = Image.fromarray(fondo)
    d = ImageDraw.Draw(img)
    filas = []
    clases = list(COLORES)
    rng.shuffle(clases)
    for (cx, cy), c in zip(POSICIONES, clases):
        r = 7
        color = tuple(int(np.clip(v + rng.integers(-15, 16), 0, 255)) for v in COLORES[c])
        d.ellipse([cx * ancho - r, cy * alto - r, cx * ancho + r, cy * alto + r], fill=color)
        filas.append(f"{c} {cx} {cy} {2 * r / ancho:.4f} {2 * r / alto:.4f}")
    ruta_img.parent.mkdir(parents=True, exist_ok=True)
    ruta_lbl.parent.mkdir(parents=True, exist_ok=True)
    img.save(ruta_img, quality=95)
    ruta_lbl.write_text("\n".join(filas) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def datos_color(tmp_path_factory):
    base = tmp_path_factory.mktemp("m0")
    rng = np.random.default_rng(0)
    raw = base / "raw"
    for f in range(40):
        for t in range(3):
            nombre = f"IMG_{1000 + f}_{t}_0"
            _tile(raw / "train" / "images" / f"{nombre}.jpg", raw / "train" / "labels" / f"{nombre}.txt", rng)
    preparar_yolo.preparar(raw, base / "yolo", "croppie", semilla=0)
    return base


def test_descriptor_dimension_determinismo_y_color():
    rojo = np.zeros((20, 20, 3), np.uint8)
    rojo[..., 0] = 200
    verde = np.zeros((20, 20, 3), np.uint8)
    verde[..., 1] = 200
    a, b = C.caracteristicas_recorte(rojo), C.caracteristicas_recorte(rojo)
    assert a.shape == (C.DIMENSION,) == (442,) and a.dtype == np.float32 and np.array_equal(a, b)
    assert not np.allclose(a[:48], C.caracteristicas_recorte(verde)[:48])  # el histograma HSV distingue el color
    assert C.extraer_lote([]).shape == (0, C.DIMENSION)


def test_recorte_con_margen_y_limites():
    img = np.zeros((50, 60, 3), np.uint8)
    assert C.recortar(img, (10, 10, 20, 20), margen=0.1).shape == (12, 12, 3)
    assert C.recortar(img, (-5, -5, 70, 70)).shape == (50, 60, 3)


def test_submuestra_estratificada():
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    idx = E.submuestrear(y, 100, semilla=0)
    conteo = np.bincount(y[idx])
    assert 95 <= len(idx) <= 102 and conteo[0] == 90 and conteo[1] == 9 and conteo[2] == 1
    assert np.array_equal(idx, E.submuestrear(y, 100, semilla=0))


@pytest.mark.parametrize("modelo", ["rf", "svm"])
def test_entrenar_evaluar_y_test_unico(datos_color, modelo):
    salida = datos_color / f"corridas_{modelo}"
    corrida = E.entrenar(datos_color / "yolo" / "data.yaml", salida, modelo, semilla=0, max_train=5000, folds=3)
    met = json.loads((corrida / "metricas_val.json").read_text(encoding="utf-8"))
    assert met["exactitud"] > 0.9 and met["f1_macro"] > 0.9, met  # el color separa las clases
    cv = json.loads((corrida / "cv.json").read_text(encoding="utf-8"))
    assert cv["criterio"].startswith("f1_macro") and cv["mejor"] in E.GRILLAS[modelo]
    man = json.loads((corrida / "modelo_m0.manifest.json").read_text(encoding="utf-8"))
    assert man["familia"] == "M0" and man["validado_con_datos_reales"] is False
    for f in ("resumen_val.md", "matriz_confusion_val.png", "tiempos.json", "config.yaml", "entorno.json"):
        assert (corrida / f).is_file(), f
    assert list((salida / "cache_m0").glob("*_train.npz"))

    resumen = E.evaluar_final(corrida)
    assert "partición `test`" in resumen and (corrida / "metricas_test.json").is_file()
    with pytest.raises(SystemExit, match="ya se usó"):
        E.evaluar_final(corrida)


def test_cache_reutilizada_y_modelo_alterado_no_se_carga(datos_color):
    salida = datos_color / "corridas_rf"
    E.caracteristicas_con_cache(datos_color / "yolo" / "data.yaml", "train", salida / "cache_m0")
    d = E.caracteristicas_con_cache(datos_color / "yolo" / "data.yaml", "train", salida / "cache_m0")
    assert d["cache"] is True and d["X"].shape[1] == C.DIMENSION
    corrida = next(salida.glob("m0_rf_*"), None) or E.entrenar(datos_color / "yolo" / "data.yaml", salida, "rf", max_train=5000)
    (corrida / "evaluacion_final.json").unlink(missing_ok=True)
    with open(corrida / "modelo_m0.joblib", "ab") as fh:
        fh.write(b"\0")
    from src.domain.exceptions.domain_exceptions import ModeloNoDisponibleException

    with pytest.raises(ModeloNoDisponibleException, match="SHA-256"):
        E.evaluar_final(corrida)

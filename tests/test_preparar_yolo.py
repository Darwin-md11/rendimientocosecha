"""Preparación de datos YOLO: mapeo de clases, enmascarado, partición por grupo y prueba de fuga."""
import json

import numpy as np
import pytest
import yaml
from PIL import Image

from ml.datos import preparar_yolo as py

N_FOTOS = 40  # fotos originales del dataset de juguete
CROPPIE_IDS = {"dark_brown_cherry": 0, "green_cherry": 1, "low_visibility_unsure": 2, "red_cherry": 3, "yellow_cherry": 4}


def _img(ruta, semilla, tam=(64, 48)):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(semilla).integers(0, 255, (tam[1], tam[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(ruta, quality=95)


def _lbl(ruta, filas):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(" ".join(str(v) for v in f) for f in filas) + "\n")


@pytest.fixture
def croppie_juguete(tmp_path):
    """Como Croppie: train/ y val/ con images/ y labels/; cada foto original -> 4 recortes + la foto reducida."""
    origen = tmp_path / "raw"
    semilla = 0
    for f in range(N_FOTOS):
        split = "train" if f < 30 else "val"
        for sufijo in ["", "_0_0", "_0_1", "_1_0", "_1_1"]:
            nombre = f"IMG_{1000 + f}{sufijo}"
            semilla += 1
            _img(origen / split / "images" / f"{nombre}.jpg", semilla)
            _lbl(
                origen / split / "labels" / f"{nombre}.txt",
                [
                    (CROPPIE_IDS["green_cherry"], 0.2, 0.2, 0.1, 0.1),
                    (CROPPIE_IDS["red_cherry"], 0.5, 0.5, 0.1, 0.1),
                    (CROPPIE_IDS["yellow_cherry"], 0.7, 0.3, 0.1, 0.1),
                    (CROPPIE_IDS["dark_brown_cherry"], 0.3, 0.7, 0.1, 0.1),
                    (CROPPIE_IDS["low_visibility_unsure"], 0.8, 0.8, 0.2, 0.2),
                ],
            )
    return origen


def test_clave_de_grupo_con_los_candidatos():
    fc, idx = py.CANDIDATOS_GRUPO["fila_columna"], py.CANDIDATOS_GRUPO["indice"]
    assert py.clave_grupo("IMG_1234_2_3.jpg", fc) == "IMG_1234"
    assert py.clave_grupo("IMG_1234.jpg", fc) == "IMG_1234"  # la foto reducida cae en su mismo grupo
    assert py.clave_grupo("foto-tile_2_3.rf.0a1b2c.jpg", fc) == "foto"
    assert py.clave_grupo("foto_crop7.jpg", idx) == "foto"


def test_elige_patron_plausible_y_rechaza_el_que_une_todo():
    nombres = [f"IMG_{1000 + f}{s}.jpg" for f in range(40) for s in ["", "_0_0", "_0_1"]]
    nombre, _, _ = py.elegir_patron(nombres)
    assert nombre == "fila_columna"
    # "indice" sobre fotos sueltas IMG_1000..IMG_1039 las uniría todas en "IMG": se rechaza
    with pytest.raises(py.ErrorPreparacion, match="Ningún patrón"):
        py.elegir_patron([f"IMG_{1000 + f}.jpg" for f in range(200)])


def test_preparacion_completa(croppie_juguete, tmp_path):
    destino = tmp_path / "out"
    r = py.preparar(croppie_juguete, destino, "croppie", semilla=0)

    # Partición por grupo, sin fuga y ~70/15/15
    man = json.loads((destino / "manifiesto.json").read_text())
    split_por_grupo = {}
    for a in man["archivos"]:
        assert split_por_grupo.setdefault(a["grupo"], a["split"]) == a["split"], "un grupo en dos particiones"
    assert r["grupos"] == N_FOTOS and r["prueba_fuga"] == "OK"
    st = r["estadisticas"]["por_split"]
    total = sum(d["imagenes"] for d in st.values())
    assert total == N_FOTOS * 5
    assert abs(st["train"]["imagenes"] / total - 0.70) < 0.08 and st["test"]["imagenes"] > 0

    # Mapeo: 4 clases SIKAFE; low_visibility excluida y contada
    assert r["mapeo"]["yellow_cherry"] == "PINTON" and r["mapeo"]["low_visibility_unsure"] is None
    assert st["train"]["cajas_por_clase"]["VERDE"] == st["train"]["imagenes"]
    assert st["train"]["excluidas_por_clase"]["low_visibility_unsure"] == st["train"]["imagenes"]

    # Etiquetas reescritas con índices 0..3 y sin la clase excluida
    una = next((destino / "labels" / "train").glob("*.txt"))
    clases = sorted(int(linea.split()[0]) for linea in una.read_text().splitlines())
    assert clases == [0, 1, 2, 3]

    # La región excluida queda enmascarada en gris (no se aprende como fondo)
    img = np.asarray(Image.open(destino / "images" / "train" / una.with_suffix(".jpg").name))
    h, w = img.shape[:2]
    centro = img[int(0.8 * h) - 2 : int(0.8 * h) + 2, int(0.8 * w) - 2 : int(0.8 * w) + 2].astype(int)
    assert np.abs(centro - 114).max() < 12

    # data.yaml listo para Ultralytics, y artefactos de EDA
    data = yaml.safe_load((destino / "data.yaml").read_text())
    assert data["names"] == {0: "VERDE", 1: "PINTON", 2: "MADURO", 3: "SOBREMADURO"} and data["test"] == "images/test"
    assert (destino / "eda.md").read_text().startswith("# Análisis exploratorio") and (destino / "eda_clases.png").is_file()


def test_misma_semilla_mismo_dataset_otra_semilla_otro(croppie_juguete, tmp_path):
    v = [py.preparar(croppie_juguete, tmp_path / f"o{i}", "croppie", semilla=s)["version_dataset_sha256"] for i, s in enumerate([0, 0, 1])]
    assert v[0] == v[1] != v[2]


def test_imagen_duplicada_en_dos_grupos_se_detecta_como_fuga(croppie_juguete, tmp_path):
    # La misma foto con otro nombre (otro grupo): si cae en otra partición, es fuga.
    for i in range(10):
        dup = croppie_juguete / "val" / "images" / f"COPIA_{i}.jpg"
        dup.write_bytes((croppie_juguete / "train" / "images" / "IMG_1000.jpg").read_bytes())
    with pytest.raises(py.ErrorPreparacion, match="FUGA"):
        py.preparar(croppie_juguete, tmp_path / "out", "croppie", semilla=0)


def test_split_original_y_errores_de_formato(croppie_juguete, tmp_path):
    r = py.preparar(croppie_juguete, tmp_path / "out", "croppie", usar_split_original=True)
    assert "original" in r["estrategia_particion"] and r["estadisticas"]["por_split"]["test"]["imagenes"] == 0

    malo = next((croppie_juguete / "train" / "labels").glob("*.txt"))
    malo.write_text("1 0.5 0.5\n")
    with pytest.raises(py.ErrorPreparacion, match="5 valores"):
        py.preparar(croppie_juguete, tmp_path / "out2", "croppie")


def test_dataset_no_configurado():
    with pytest.raises(py.ErrorPreparacion, match="no está"):
        py.cargar_mapeo("inexistente")


def test_inspeccionar(croppie_juguete):
    info = py.inspeccionar(croppie_juguete)
    assert info["imagenes"] == N_FOTOS * 5 and info["patron_elegido"] == "fila_columna"
    assert info["grupos_en_varios_splits_originales"] == 0 and info["cajas_por_clase_origen"]["2"] == N_FOTOS * 5

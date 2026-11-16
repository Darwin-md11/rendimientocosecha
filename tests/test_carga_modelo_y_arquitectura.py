"""
Fase 0: falla rápida y segura al cargar modelos, y separación app (src/) / laboratorio (ml/).
"""
import ast
import json
import shutil
from pathlib import Path

import pytest

from src.domain.exceptions.domain_exceptions import ModeloNoDisponibleException
from src.infrastructure import composition
from src.infrastructure.composition import RUTA_MODELO_POR_DEFECTO, construir_detector
from src.infrastructure.ml import modelo as modulo_modelo
from src.infrastructure.ml.modelo import ModeloMadurezPixel, ruta_manifiesto, sha256_archivo

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def copia_modelo(tmp_path):
    destino = tmp_path / "madurez_pixel.joblib"
    shutil.copy(RUTA_MODELO_POR_DEFECTO, destino)
    shutil.copy(ruta_manifiesto(RUTA_MODELO_POR_DEFECTO), ruta_manifiesto(destino))
    return destino


@pytest.fixture
def prohibir_joblib_load(monkeypatch):
    """Si el manifiesto no cuadra, el archivo NO debe deserializarse (pickle ejecuta código)."""

    def _no(*a, **k):
        raise AssertionError("joblib.load no debía llamarse")

    monkeypatch.setattr(modulo_modelo.joblib, "load", _no)


def _editar_manifiesto(ruta, **cambios):
    m = ruta_manifiesto(ruta)
    datos = json.loads(m.read_text(encoding="utf-8"))
    datos.update(cambios)
    m.write_text(json.dumps(datos), encoding="utf-8")


def test_manifiesto_versionado_coincide_con_el_modelo():
    manifiesto = json.loads(ruta_manifiesto(RUTA_MODELO_POR_DEFECTO).read_text(encoding="utf-8"))
    assert manifiesto["sha256"] == sha256_archivo(RUTA_MODELO_POR_DEFECTO)
    assert manifiesto["validado_con_datos_reales"] is False and manifiesto["familia"] == "SINTETICO"


def test_carga_correcta_expone_hash(copia_modelo):
    m = ModeloMadurezPixel.cargar(copia_modelo)
    assert m.sha256 == sha256_archivo(copia_modelo)


def test_hash_distinto_no_carga(copia_modelo, prohibir_joblib_load):
    with open(copia_modelo, "ab") as f:
        f.write(b"\0")  # archivo alterado
    with pytest.raises(ModeloNoDisponibleException, match="SHA-256"):
        ModeloMadurezPixel.cargar(copia_modelo)


def test_sin_manifiesto_no_carga(copia_modelo, prohibir_joblib_load):
    ruta_manifiesto(copia_modelo).unlink()
    with pytest.raises(ModeloNoDisponibleException, match="manifiesto"):
        ModeloMadurezPixel.cargar(copia_modelo)


def test_version_de_sklearn_distinta_falla_con_instruccion(copia_modelo, prohibir_joblib_load):
    _editar_manifiesto(copia_modelo, version_sklearn="0.0.1")
    with pytest.raises(ModeloNoDisponibleException, match=r"pip install scikit-learn==0\.0\.1"):
        ModeloMadurezPixel.cargar(copia_modelo)


def test_modelo_ausente_indica_como_descargarlo(tmp_path):
    with pytest.raises(ModeloNoDisponibleException, match="descargar_modelo"):
        ModeloMadurezPixel.cargar(tmp_path / "no_existe.joblib")


def test_composicion_no_reentrena_ni_cambia_de_detector(copia_modelo, monkeypatch):
    """Antes: si el modelo fallaba se reentrenaba en silencio. Ahora: error explícito."""
    ruta_manifiesto(copia_modelo).unlink()
    monkeypatch.setenv("SIKAFE_DETECTOR", "sintetico")
    with pytest.raises(ModeloNoDisponibleException):
        construir_detector(ruta_modelo=copia_modelo)
    assert "auto_entrenar" not in composition.construir_detector.__code__.co_varnames


# ------------------------------------------------------------ arquitectura
def _modulos_importados(archivo: Path, solo_nivel_modulo: bool = False):
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    for nodo in arbol.body if solo_nivel_modulo else ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            yield from (a.name for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
            yield nodo.module


def test_la_app_no_depende_del_laboratorio():
    """src/ (app liviana) nunca importa de ml/ (entrenamiento, datos, experimentos)."""
    infractores = [
        f"{archivo.relative_to(RAIZ)} -> {mod}"
        for archivo in (RAIZ / "src").rglob("*.py")
        for mod in _modulos_importados(archivo)
        if mod == "ml" or mod.startswith("ml.")
    ]
    assert infractores == []


def test_la_app_no_importa_librerias_de_entrenamiento_pesadas():
    pesadas = ("torch", "ultralytics", "tensorflow")
    infractores = [
        f"{archivo.relative_to(RAIZ)} -> {mod}"
        for archivo in (RAIZ / "src").rglob("*.py")
        for mod in _modulos_importados(archivo, solo_nivel_modulo=True)
        if mod.split(".")[0] in pesadas
    ]
    assert infractores == [], "use import diferido dentro del adaptador que lo necesite"


def _es_texto_sin_encoding(archivo: Path):
    for n in ast.walk(ast.parse(archivo.read_text(encoding="utf-8"))):
        if not isinstance(n, ast.Call):
            continue
        kws = {k.arg for k in n.keywords}
        f = n.func
        nombre = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if "encoding" in kws:
            continue
        if nombre == "read_text" and not n.args:
            yield n.lineno
        elif nombre == "write_text" and len(n.args) < 2:
            yield n.lineno
        elif nombre == "open" and isinstance(f, ast.Name):
            modo = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else "r"
            if "b" not in str(modo):
                yield n.lineno


def test_toda_la_lectura_y_escritura_de_texto_declara_utf8():
    """En Windows la codificación por defecto no es UTF-8 (cp1252): sin encoding explícito, las tildes se corrompen."""
    infractores = [
        f"{archivo.relative_to(RAIZ)}:{linea}"
        for carpeta in ("src", "ml", "tests", "scripts")
        for archivo in (RAIZ / carpeta).rglob("*.py")
        for linea in _es_texto_sin_encoding(archivo)
    ]
    assert infractores == [], 'agregue encoding="utf-8"'

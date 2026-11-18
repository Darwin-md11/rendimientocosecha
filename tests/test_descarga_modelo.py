"""El script de descarga verifica el SHA-256 y nunca deja un archivo sin verificar."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("descargar_modelo", Path(__file__).resolve().parents[1] / "scripts" / "descargar_modelo.py")
descargar_modelo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(descargar_modelo)


def _preparar(tmp_path, contenido=b"pesos de prueba", sha=None):
    origen = tmp_path / "publicado.bin"
    origen.write_bytes(contenido)
    destino_dir = tmp_path / "models"
    destino_dir.mkdir()
    manifiesto = destino_dir / "m.manifest.json"
    manifiesto.write_text(
        json.dumps(
            {
                "archivo": "m.joblib",
                "sha256": sha or hashlib.sha256(contenido).hexdigest(),
                "url_descarga": origen.as_uri(),  # file:// evita depender de la red
            }
        ),
        encoding="utf-8",
    )
    return manifiesto, destino_dir / "m.joblib"


def test_descarga_y_verifica(tmp_path):
    manifiesto, destino = _preparar(tmp_path)
    assert descargar_modelo.descargar(manifiesto) == destino
    assert destino.read_bytes() == b"pesos de prueba"
    descargar_modelo.descargar(manifiesto)  # segunda vez: ya existe y coincide


def test_hash_incorrecto_descarta_el_archivo(tmp_path):
    manifiesto, destino = _preparar(tmp_path, sha="0" * 64)
    with pytest.raises(SystemExit, match="SHA-256 incorrecto"):
        descargar_modelo.descargar(manifiesto)
    assert not destino.exists() and not list(destino.parent.glob("*.parcial"))


def test_url_inexistente_falla_con_mensaje(tmp_path):
    manifiesto, destino = _preparar(tmp_path)
    with pytest.raises(SystemExit, match="Release"):
        descargar_modelo.descargar(manifiesto, url=(tmp_path / "no_hay.bin").as_uri())
    assert not destino.exists()

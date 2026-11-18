"""Publicación de M2: solo la corrida oficial, con test evaluado y hash correcto; la descarga baja todos los modelos."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

_RAIZ = Path(__file__).resolve().parents[1]


def _cargar(nombre):
    spec = importlib.util.spec_from_file_location(nombre, _RAIZ / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _cargar("publicar_modelo_m2")
D = _cargar("descargar_modelo")


def _corrida(tmp_path, corrida_id="m2_oficial", validado=True, contenido=b"pesos yolo"):
    c = tmp_path / "corrida"
    c.mkdir()
    (c / "m2_yolo.pt").write_bytes(contenido)
    man = {
        "archivo": "m2_yolo.pt",
        "sha256": hashlib.sha256(contenido).hexdigest(),
        "clases": ["VERDE"],
        "corrida_id": corrida_id,
        "validado_con_datos_reales": validado,
        "url_descarga": None,
    }
    (c / "m2_yolo.manifest.json").write_text(json.dumps(man), encoding="utf-8")
    cfg = tmp_path / "informe.yaml"
    cfg.write_text(yaml.safe_dump({"corridas": {"m2": "m2_oficial"}}), encoding="utf-8")
    return c, cfg


def test_prepara_manifiesto_con_url_de_release(tmp_path):
    c, cfg = _corrida(tmp_path)
    salida = P.preparar(c, tmp_path / "models", "equipo/repo", "modelo-m2-v1", cfg)
    man = json.loads(salida.read_text(encoding="utf-8"))
    assert man["url_descarga"] == "https://github.com/equipo/repo/releases/download/modelo-m2-v1/m2_yolo.pt"
    assert man["release_tag"] == "modelo-m2-v1" and man["licencia"].startswith("TODO(fuente)")
    assert "advertencias_publicacion" not in man
    assert (tmp_path / "models" / "m2_yolo.pt").read_bytes() == b"pesos yolo"


@pytest.mark.parametrize("kwargs, motivo", [({"corrida_id": "otra"}, "no es la oficial"), ({"validado": False}, "evaluación final")])
def test_rechaza_corrida_no_oficial_o_sin_test(tmp_path, kwargs, motivo):
    c, cfg = _corrida(tmp_path, **kwargs)
    with pytest.raises(SystemExit, match=motivo):
        P.preparar(c, tmp_path / "models", config=cfg)
    man = json.loads(P.preparar(c, tmp_path / "models", config=cfg, forzar=True).read_text(encoding="utf-8"))
    assert motivo in " ".join(man["advertencias_publicacion"])  # forzar queda anotado


def test_hash_incorrecto_no_se_puede_forzar(tmp_path):
    c, cfg = _corrida(tmp_path)
    (c / "m2_yolo.pt").write_bytes(b"otro archivo")
    with pytest.raises(SystemExit, match="no coincide"):
        P.preparar(c, tmp_path / "models", config=cfg, forzar=True)
    assert not (tmp_path / "models").exists()


def test_config_real_apunta_a_la_corrida_m2_oficial():
    assert P.corrida_oficial(_RAIZ / "configs" / "informe.yaml") == "m2_yolo11n_640_s0_20261116-1754"


def test_descarga_sin_argumentos_baja_todos_los_modelos(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for nombre in ("a", "b"):
        origen = tmp_path / f"{nombre}.bin"
        origen.write_bytes(nombre.encode())
        (models / f"{nombre}.manifest.json").write_text(
            json.dumps({"archivo": f"{nombre}.pt", "sha256": hashlib.sha256(nombre.encode()).hexdigest(), "url_descarga": origen.as_uri()}),
            encoding="utf-8",
        )
    assert D.main(["--carpeta", str(models)]) == 0
    assert (models / "a.pt").read_bytes() == b"a" and (models / "b.pt").read_bytes() == b"b"

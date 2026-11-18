"""Los notebooks de Colab/Kaggle: JSON válido, código que compila y sin comentarios en líneas de magia (%cd los toma como ruta)."""
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "ml" / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_valido(ruta):
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    for i, celda in enumerate(nb["cells"]):
        if celda["cell_type"] != "code":
            continue
        lineas = "".join(celda["source"]).splitlines()
        for linea in lineas:
            assert not re.match(r"^\s*%\w+.*#", linea), f"celda {i}: comentario en una línea de magia: {linea!r}"
        python = "\n".join(re.sub(r"^(\s*)[!%].*$", r"\1pass", linea) for linea in lineas)
        compile(python, f"{ruta.name}:celda{i}", "exec")

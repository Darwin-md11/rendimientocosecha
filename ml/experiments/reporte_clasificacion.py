"""
Reporte común de clasificación de madurez sobre recortes (M0, M1): mismas métricas, misma tabla y
misma figura, para que la comparación del informe sea directa.
"""

from pathlib import Path
from typing import List

import numpy as np

from ml.datos.lectura_yolo import NOMBRES
from ml.experiments.registro import guardar_json
from src.evaluation import metricas as M


def metricas(y_real, y_pred, segundos_prediccion: float, dispositivo: str) -> dict:
    met = M.metricas_clasificacion(y_real, y_pred, len(NOMBRES))
    met["clases"] = NOMBRES
    met["recortes"] = int(len(y_real))
    met["ms_por_recorte_prediccion"] = 1000 * segundos_prediccion / max(1, len(y_real))
    met["dispositivo_prediccion"] = dispositivo
    return met


def resumen_md(titulo: str, contexto: List[str], met: dict, generador: str) -> str:
    filas = "\n".join(
        f"| {n} | {met['por_clase'][i]['precision']:.3f} | {met['por_clase'][i]['recall']:.3f} | {met['por_clase'][i]['f1']:.3f} | {met['por_clase'][i]['soporte']} |"
        for i, n in enumerate(NOMBRES)
    )
    lineas = "\n".join(f"- {c}" for c in contexto)
    return f"""# {titulo}

{lineas}

| Métrica | Valor |
|---|---|
| Exactitud | {met["exactitud"]:.3f} |
| F1 macro | {met["f1_macro"]:.3f} |
| Predicción (ms por recorte, {met["dispositivo_prediccion"].upper()}) | {met["ms_por_recorte_prediccion"]:.3f} |

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
{filas}

Generado por `{generador}` (no editar a mano).
"""


def grafico_confusion(ruta_png: Path, met: dict, titulo: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("AVISO: matplotlib no está instalado; no se genera la matriz de confusión.")
        return
    m = np.asarray(met["matriz_confusion"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(NOMBRES)), NOMBRES, rotation=45, ha="right")
    ax.set_yticks(range(len(NOMBRES)), NOMBRES)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


def guardar(corrida: Path, split: str, met: dict, titulo: str, contexto: List[str], generador: str, etiqueta_modelo: str) -> str:
    guardar_json(corrida / f"metricas_{split}.json", met)
    resumen = resumen_md(titulo, contexto, met, generador)
    (corrida / f"resumen_{split}.md").write_text(resumen, encoding="utf-8")
    grafico_confusion(corrida / f"matriz_confusion_{split}.png", met, f"{etiqueta_modelo} – matriz de confusión ({split})")
    return resumen

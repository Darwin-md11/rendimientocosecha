"""
Evalúa M2 con las métricas comunes de src/evaluation (las mismas para M0/M1/M2) y a
través del MISMO adaptador que usa la app (YoloFruitDetectorAdapter): lo que se
reporta es exactamente lo que el usuario obtiene.

Uso:
    python -m ml.m2_yolo.evaluar --corrida <carpeta_corrida> --split val            # libre
    python -m ml.m2_yolo.evaluar --corrida <carpeta_corrida> --split test --final   # UNA sola vez

Regla del protocolo: el test se toca una sola vez al final (--final queda registrado;
repetirlo exige --forzar y también queda registrado). Con --final el manifiesto del
modelo pasa a validado_con_datos_reales=true y anota el dominio de validación.

Salidas en la corrida: metricas_<split>.json, resumen_<split>.md, conteos_<split>.csv,
predicciones_<split>.jsonl, matriz_confusion_<split>.png, dispersion_conteo_<split>.png
"""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml
from PIL import Image, ImageOps

from ml.experiments.registro import capturar_entorno, guardar_json
from src.domain.models.coffee_cherry import EstadoMadurez
from src.evaluation import metricas as M
from src.infrastructure.adapters.detectors.yolo_detector_adapter import YoloFruitDetectorAdapter, normalizar_nombre_clase
from src.infrastructure.ml.manifiesto import ruta_manifiesto

ESTADOS = list(EstadoMadurez)
NOMBRES = [e.value for e in ESTADOS]
CONF_MAP = 0.001  # umbral bajo para la curva precisión-recall (mAP)


def _imagenes_y_etiquetas(data_yaml: Path, split: str):
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    raiz = Path(data.get("path", Path(data_yaml).parent))
    carpeta = raiz / data[split]
    nombres = data["names"]
    nombres = [nombres[i] for i in sorted(nombres)] if isinstance(nombres, dict) else list(nombres)
    a_estado = {i: NOMBRES.index(normalizar_nombre_clase(n)) for i, n in enumerate(nombres) if normalizar_nombre_clase(n) in NOMBRES}
    partes = list(Path(data[split]).parts)
    partes = ["labels" if x == "images" else x for x in partes]  # convención YOLO: images/... -> labels/...
    carpeta_lbl = raiz.joinpath(*partes)
    imagenes = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    for img in imagenes:
        yield img, carpeta_lbl / (img.stem + ".txt"), a_estado


def _gt(lbl: Path, ancho: int, alto: int, a_estado: dict):
    cajas, clases = [], []
    if lbl.is_file():
        for linea in lbl.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            c, cx, cy, w, h = linea.split()
            c = int(float(c))
            if c not in a_estado:
                continue
            cx, cy, w, h = float(cx) * ancho, float(cy) * alto, float(w) * ancho, float(h) * alto
            cajas.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            clases.append(a_estado[c])
    return np.asarray(cajas, float).reshape(-1, 4), np.asarray(clases, int)


def evaluar(detector, data_yaml: Path, split: str, conf_reporte: float = 0.25, n_calentamiento: int = 3):
    imagenes: List[M.ImagenEval] = []
    filas, predicciones, tiempos = [], [], []
    for k, (ruta, lbl, a_estado) in enumerate(_imagenes_y_etiquetas(data_yaml, split)):
        with Image.open(ruta) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")  # igual que el caso de uso de la app
        t0 = time.perf_counter()
        dets = detector.detectar_frutos(img, CONF_MAP)
        dt = time.perf_counter() - t0
        if k >= n_calentamiento:
            tiempos.append(dt)
        gt_cajas, gt_clases = _gt(lbl, img.width, img.height, a_estado)
        pc = np.asarray([[d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max] for d in dets], float).reshape(-1, 4)
        pf = np.asarray([d.confianza for d in dets], float)
        pk = np.asarray([ESTADOS.index(d.estado) for d in dets], int)
        imagenes.append(M.ImagenEval.crear(pc, pf, pk, gt_cajas, gt_clases))
        n_auto = int((pf >= conf_reporte - 1e-9).sum())
        filas.append({"imagen": ruta.name, "conteo_anotado": len(gt_clases), "conteo_automatico": n_auto,
                      **{f"auto_{n}": int(((pf >= conf_reporte - 1e-9) & (pk == i)).sum()) for i, n in enumerate(NOMBRES)},
                      **{f"anotado_{n}": int((gt_clases == i).sum()) for i, n in enumerate(NOMBRES)}})
        predicciones.append({"imagen": ruta.name, "cajas": pc.round(2).tolist(), "conf": pf.round(4).tolist(), "clase": pk.tolist()})
    if not imagenes:
        raise SystemExit(f"No hay imágenes en la partición {split}")

    det = M.metricas_deteccion(imagenes, 0.5, conf_reporte, len(NOMBRES))
    map50 = M.mean_average_precision(imagenes, len(NOMBRES), (0.5,))
    map_coco = M.mean_average_precision(imagenes, len(NOMBRES), M.UMBRALES_COCO)
    conteo = M.metricas_conteo([f["conteo_anotado"] for f in filas], [f["conteo_automatico"] for f in filas])
    conteo_por_clase = {n: M.metricas_conteo([f[f"anotado_{n}"] for f in filas], [f[f"auto_{n}"] for f in filas]) for n in NOMBRES}
    matriz = M.matriz_confusion_deteccion(imagenes, len(NOMBRES), 0.5, conf_reporte)
    metricas = {
        "split": split, "imagenes": len(imagenes), "cerezas_anotadas": int(sum(f["conteo_anotado"] for f in filas)),
        "clases": NOMBRES, "conf_reporte": conf_reporte, "implementacion": "src/evaluation/metricas.py (COCO 101 puntos)",
        "map50": map50["map"], "map50_95": map_coco["map"],
        "ap50_por_clase": {NOMBRES[c]: v[0] for c, v in map50["ap_por_clase"].items()},
        "deteccion": det, "matriz_confusion": {"filas_real_columnas_pred": NOMBRES + ["fondo"], "valores": matriz.tolist()},
        "conteo": {"nota": "conteo_anotado = cajas etiquetadas en el dataset (proxy del conteo manual en campo)", "total": conteo, "por_clase": conteo_por_clase},
        "tiempo_inferencia_s": {"mediana": statistics.median(tiempos) if tiempos else None, "n": len(tiempos), "calentamiento": n_calentamiento},
    }
    return metricas, filas, predicciones


def _graficos(corrida: Path, split: str, metricas: dict, filas: list) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = np.asarray(metricas["matriz_confusion"]["valores"])
    etiquetas = metricas["matriz_confusion"]["filas_real_columnas_pred"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=45, ha="right")
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black", fontsize=8)
    ax.set_title(f"Matriz de confusión ({split}, IoU 0,5, conf {metricas['conf_reporte']})")
    fig.tight_layout()
    fig.savefig(corrida / f"matriz_confusion_{split}.png", dpi=150)
    plt.close(fig)

    x = [f["conteo_anotado"] for f in filas]
    y = [f["conteo_automatico"] for f in filas]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=10, alpha=0.6)
    tope = max(max(x, default=1), max(y, default=1))
    ax.plot([0, tope], [0, tope], "k--", lw=1, label="y = x")
    c = metricas["conteo"]["total"]
    r2 = "n/d" if c["r2"] is None else f"{c['r2']:.3f}"
    ax.set_title(f"Conteo por imagen ({split}): MAE {c['mae']:.2f}, R² {r2}")
    ax.set_xlabel("Conteo anotado")
    ax.set_ylabel("Conteo automático")
    ax.legend()
    fig.tight_layout()
    fig.savefig(corrida / f"dispersion_conteo_{split}.png", dpi=150)
    plt.close(fig)


def _resumen_md(corrida: Path, split: str, met: dict) -> str:
    d = met["deteccion"]
    c = met["conteo"]["total"]

    def f(v):
        return "n/d" if v is None else f"{v:.3f}"

    filas_clase = "\n".join(
        f"| {n} | {f(met['ap50_por_clase'][n])} | {f(d['por_clase'][i]['precision'])} | {f(d['por_clase'][i]['recall'])} | {f(d['por_clase'][i]['f1'])} |"
        for i, n in enumerate(met["clases"])
    )
    return f"""# Evaluación M2 — corrida `{corrida.name}` — partición `{split}`

Imágenes: {met['imagenes']} · cerezas anotadas: {met['cerezas_anotadas']} · métricas: {met['implementacion']}

| Métrica | Valor |
|---|---|
| mAP@0,5 | {f(met['map50'])} |
| mAP@0,5:0,95 | {f(met['map50_95'])} |
| Precisión / Recall / F1 (con clase, conf {met['conf_reporte']}) | {f(d['con_clase']['precision'])} / {f(d['con_clase']['recall'])} / {f(d['con_clase']['f1'])} |
| Precisión / Recall / F1 (solo localización) | {f(d['localizacion']['precision'])} / {f(d['localizacion']['recall'])} / {f(d['localizacion']['f1'])} |
| Conteo por imagen: MAE / RMSE / sesgo | {f(c['mae'])} / {f(c['rmse'])} / {f(c['sesgo_medio'])} |
| Conteo por imagen: R² / r² Pearson | {f(c['r2'])} / {f(c['r2_pearson'])} |
| Tiempo de inferencia (mediana, s/imagen) | {f(met['tiempo_inferencia_s']['mediana'])} |

| Clase | AP@0,5 | Precisión | Recall | F1 |
|---|---|---|---|---|
{filas_clase}

Nota: el "conteo anotado" son las cajas etiquetadas del dataset, no un conteo manual en campo.
Generado por `python -m ml.m2_yolo.evaluar` (no editar a mano).
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrida", type=Path, required=True)
    p.add_argument("--datos", type=Path, help="data.yaml (por defecto, el de la config de la corrida)")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--final", action="store_true", help="evaluación única sobre test")
    p.add_argument("--forzar", action="store_true")
    p.add_argument("--conf", type=float, help="umbral para P/R/F1 y conteo (por defecto el de la config)")
    a = p.parse_args(argv)

    cfg = yaml.safe_load((a.corrida / "config.yaml").read_text(encoding="utf-8"))
    datos = a.datos or Path(cfg["datos"])
    registro_final = a.corrida / "evaluacion_final.json"
    if a.split == "test":
        if not a.final:
            raise SystemExit("El test se evalúa una sola vez al final: use --split test --final.")
        if registro_final.is_file() and not a.forzar:
            raise SystemExit(f"Ya existe {registro_final}: el test ya se usó. Repetir exige --forzar (queda registrado).")

    pesos = a.corrida / "m2_yolo.pt"
    detector = YoloFruitDetectorAdapter.desde_archivo(pesos)
    conf = a.conf if a.conf is not None else cfg.get("conf_reporte", 0.25)
    metricas, filas, preds = evaluar(detector, datos, a.split, conf)
    metricas["entorno"] = capturar_entorno(datos)

    guardar_json(a.corrida / f"metricas_{a.split}.json", metricas)
    with open(a.corrida / f"conteos_{a.split}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    with open(a.corrida / f"predicciones_{a.split}.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in preds)
    resumen = _resumen_md(a.corrida, a.split, metricas)
    (a.corrida / f"resumen_{a.split}.md").write_text(resumen, encoding="utf-8")
    _graficos(a.corrida, a.split, metricas, filas)

    if a.final:
        previo = json.loads(registro_final.read_text(encoding="utf-8")) if registro_final.is_file() else {"ejecuciones": []}
        previo["ejecuciones"].append({"fecha": metricas["entorno"]["fecha_utc"], "forzado": bool(a.forzar), "map50": metricas["map50"]})
        guardar_json(registro_final, previo)
        man_ruta = ruta_manifiesto(pesos)
        man = json.loads(man_ruta.read_text(encoding="utf-8"))
        datos_res = json.loads((Path(datos).parent / "manifiesto.json").read_text(encoding="utf-8"))["resumen"] if (Path(datos).parent / "manifiesto.json").is_file() else {}
        man.update({
            "validado_con_datos_reales": True,
            "dominio_validacion": f"{datos_res.get('dataset', 'dataset')} partición test ({metricas['imagenes']} imágenes), corrida {a.corrida.name}",
            "validado_en_dominio_objetivo": False,
            "metricas_test": {"map50": metricas["map50"], "map50_95": metricas["map50_95"], "f1": metricas["deteccion"]["con_clase"]["f1"],
                              "mae_conteo": metricas["conteo"]["total"]["mae"], "r2_conteo": metricas["conteo"]["total"]["r2"]},
        })
        guardar_json(man_ruta, man)
    print(resumen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Genera las tablas y figuras del capítulo 8 (8.2 datos, 8.3 modelos, 8.4 evaluación) a partir de las carpetas de las
corridas oficiales (configs/informe.yaml). No entrena ni evalúa: solo lee lo que dejaron los scripts (y, para los
intervalos de confianza de M2, recalcula métricas desde las predicciones de test guardadas).

Uso (Colab, con el dataset preparado en /content/sikafe_yolo):
    python -m ml.informe.generar --corridas /content/drive/MyDrive/sikafe/corridas \
        --datos /content/sikafe_yolo/data.yaml --benchmarks /content/drive/MyDrive/sikafe/benchmarks \
        --salida /content/drive/MyDrive/sikafe/informe

Salida: <salida>/informe_resultados.md, <salida>/figuras/*.png, <salida>/tablas.json
"""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import yaml

from ml.datos.lectura_yolo import NOMBRES
from ml.informe import bootstrap_m2 as B

RAIZ = Path(__file__).resolve().parents[2]
CLASIFICADORES = ("m0_rf", "m0_svm", "m1_sin_preentrenar", "m1")


def _json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else None


def _yaml(ruta: Path):
    return yaml.safe_load(ruta.read_text(encoding="utf-8")) if ruta.is_file() else None


def _f(v, d=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}".replace(".", ",")


_NUMERO_CON_PUNTO = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?=e[-+]\d|(?![\w.]))")


def coma_decimal(texto: str) -> str:
    """1.5 -> 1,5 y 1.39e-12 -> 1,39e-12; no toca versiones (0.64.0), plataformas (glibc2.39) ni identificadores."""
    return _NUMERO_CON_PUNTO.sub(r"\1,\2", texto)


def _num(v) -> str:
    return "—" if v is None else f"{v:g}".replace(".", ",")


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("AVISO: matplotlib no está instalado; no se generan figuras.")
        return None


# ====================================================================== 8.2 Datos
def seccion_datos(datos_yaml: Path, figuras: Path) -> str:
    est = _json(Path(datos_yaml).parent / "estadisticas.json")
    if not est:
        return "## 8.2 Datos\n\nNo se encontró `estadisticas.json` junto a data.yaml (ejecute ml.datos.preparar_yolo).\n"
    st = est["estadisticas"]["por_split"]
    total_img = sum(d["imagenes"] for d in st.values())
    filas = []
    for s, d in st.items():
        cajas = d["cajas"]
        pct = " | ".join(f"{d['cajas_por_clase'][c]} ({_f(100 * d['cajas_por_clase'][c] / max(cajas, 1), 1)} %)" for c in NOMBRES)
        filas.append(
            f"| {s} | {d['imagenes']} ({_f(100 * d['imagenes'] / total_img, 1)} %) | {est['grupos_por_split'].get(s, '—')} | {cajas} | {pct} |"
        )
    ag = est.get("agrupamiento", {})
    plt = _plt()
    if plt:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        izq = np.zeros(len(st))
        for c in NOMBRES:
            v = np.array([100 * d["cajas_por_clase"][c] / max(d["cajas"], 1) for d in st.values()])
            ax.barh(list(st), v, left=izq, label=c)
            izq += v
        ax.set_xlabel("% de cerezas anotadas")
        ax.set_title("Proporción de estados de madurez por partición")
        ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.45))
        fig.tight_layout()
        fig.savefig(figuras / "8_2_proporcion_clases.png", dpi=150)
        plt.close(fig)
    eda_png = Path(datos_yaml).parent / "eda_clases.png"
    if eda_png.is_file():
        shutil.copy2(eda_png, figuras / "8_2_cerezas_por_clase.png")
    excl = {}
    for d in st.values():
        for k, v in d["excluidas_por_clase"].items():
            excl[k] = excl.get(k, 0) + v
    return f"""## 8.2 Datos

- Dataset: {est["dataset"]} · {est.get("licencia")} · {est.get("atribucion")}
- Versión del dataset preparado (SHA-256 del manifiesto): `{est["version_dataset_sha256"]}`
- Partición: {est["estrategia_particion"]}
- Agrupamiento: nivel **{ag.get("nivel", "—")}** ({ag.get("fotos", "—")} fotos originales{"; umbral " + str(ag["sesion_minutos"]) + " min" if ag.get("sesion_minutos") else ""}) · prueba de fuga: **{est["prueba_fuga"]}**
- Mapeo de clases: {est["mapeo"]} · cajas excluidas y enmascaradas: {excl}

Tabla 8.2.1 — Particiones y cerezas anotadas por estado de madurez

| Partición | Imágenes | Sesiones | Cerezas | {" | ".join(NOMBRES)} |
|---|---|---|---|{"---|" * len(NOMBRES)}
{chr(10).join(filas)}

Figuras: `figuras/8_2_cerezas_por_clase.png`, `figuras/8_2_proporcion_clases.png`.
"""


# ====================================================================== 8.3 Modelos
def _tiempo_m2(corrida: Path):
    rc = corrida / "train" / "results.csv"
    if not rc.is_file():
        return None, []
    with open(rc, encoding="utf-8") as fh:
        filas = [{k.strip(): v for k, v in f.items()} for f in csv.DictReader(fh)]
    return (float(filas[-1]["time"]) if filas and "time" in filas[-1] else None), filas


def seccion_modelos(cfg: dict, base: Path, figuras: Path) -> str:
    C = cfg["corridas"]
    filas = []
    for k in ("m0_rf", "m0_svm"):
        c = base / C[k]
        conf, t = _yaml(c / "config.yaml") or {}, _json(c / "tiempos.json") or {}
        filas.append(
            f"| {cfg['etiquetas'][k]} | descriptor v{conf.get('descriptor_version')} ({conf.get('dimension')} dim.) + "
            f"{conf.get('modelo', '').upper()} `{conf.get('params')}` | CV por sesión ({conf.get('folds')} pliegues), "
            f"train submuestreado a {conf.get('n_train')} | {_f((t.get('validacion_cruzada_s', 0) + t.get('entrenamiento_final_s', 0)) / 60, 1)} min (CPU) |"
        )
    plt = _plt()
    for k in ("m1_sin_preentrenar", "m1"):
        c = base / C[k]
        conf, t = _yaml(c / "config.yaml") or {}, _json(c / "tiempos.json") or {}
        filas.append(
            f"| {cfg['etiquetas'][k]} | {conf.get('arquitectura')} {'preentrenada (ImageNet)' if conf.get('preentrenado', True) else 'sin preentrenar'}, "
            f"{conf.get('lado')} px | {conf.get('epocas_cabeza')} épocas cabeza + {conf.get('epocas')} ajuste; AdamW, lr {_num(conf.get('lr'))}; "
            f"pesos por clase{'; AMP' if conf.get('amp') else ''} | {_f(t.get('entrenamiento_s', 0) / 60, 1)} min ({'GPU' if t.get('gpu') else 'CPU'}) |"
        )
    if plt:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for k in ("m1_sin_preentrenar", "m1"):
            h = base / C[k] / "historial.csv"
            if h.is_file():
                with open(h, encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                ax.plot(range(1, len(rows) + 1), [float(r["f1_macro_val"]) for r in rows], marker="o", label=cfg["etiquetas"][k])
        ax.set_xlabel("Época (cabeza + ajuste completo)")
        ax.set_ylabel("F1 macro en val")
        ax.set_title("M1: curva de entrenamiento")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figuras / "8_3_curva_m1.png", dpi=150)
        plt.close(fig)
    c = base / C["m2"]
    conf, ent = _yaml(c / "config.yaml") or {}, _json(c / "entorno.json") or {}
    t2, res = _tiempo_m2(c)
    filas.append(
        f"| {cfg['etiquetas']['m2'].split(' (')[0]} | {conf.get('modelo_base')} (preentrenado COCO), {conf.get('imgsz')} px | "
        f"{conf.get('epochs')} épocas, paciencia {conf.get('patience')}, batch {conf.get('batch')}; NMS agnóstico (IoU {_num(conf.get('iou_nms'))}) | "
        f"{_f(t2 / 3600, 2) + ' h' if t2 else '—'} ({ent.get('gpu') or 'GPU'}) |"
    )
    if plt and res:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ep = [int(float(r["epoch"])) for r in res]
        for col, et in (("metrics/mAP50(B)", "mAP@0,5 (val, Ultralytics)"), ("metrics/mAP50-95(B)", "mAP@0,5:0,95 (val, Ultralytics)")):
            if col in res[0]:
                ax.plot(ep, [float(r[col]) for r in res], label=et)
        ax.set_xlabel("Época")
        ax.set_title("M2: curva de entrenamiento (YOLO11n)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figuras / "8_3_curva_m2.png", dpi=150)
        plt.close(fig)
    return f"""## 8.3 Modelos

Tabla 8.3.1 — Configuración y entrenamiento de los modelos (corridas oficiales)

| Modelo | Método | Entrenamiento | Tiempo |
|---|---|---|---|
{chr(10).join(filas)}

Figuras: `figuras/8_3_curva_m1.png`, `figuras/8_3_curva_m2.png`.
"""


# ====================================================================== 8.4 Evaluación
def _met_clasif(c: Path, split: str):
    m = _json(c / f"metricas_{split}.json")
    if not m:
        return None
    return {
        "exactitud": m["exactitud"],
        "f1_macro": m["f1_macro"],
        "f1": [m["por_clase"][str(i)]["f1"] for i in range(len(NOMBRES))],
        "ms": m.get("ms_por_recorte_prediccion"),
        "disp": m.get("dispositivo_prediccion", "cpu"),
    }


def seccion_evaluacion(cfg: dict, base: Path, datos_yaml: Path, benchmarks: Path, figuras: Path, n_boot: int, tablas: dict) -> str:
    C, E = cfg["corridas"], cfg["etiquetas"]
    filas, comp, f1_test = [], [], {}
    for k in CLASIFICADORES:
        te, va = _met_clasif(base / C[k], "test"), _met_clasif(base / C[k], "val")
        if te:
            f1_test[E[k]] = te["f1"]
            filas.append(
                f"| {E[k]} | {_f(te['exactitud'])} | **{_f(te['f1_macro'])}** | {' | '.join(_f(v) for v in te['f1'])} | {_f(te['ms'])} ({te['disp'].upper()}) |"
            )
            comp.append(f"| {E[k]} | {_f(va['f1_macro']) if va else '—'} | {_f(te['f1_macro'])} |")
    m2t, m2v = _json(base / C["m2"] / "metricas_test.json"), _json(base / C["m2"] / "metricas_val.json")
    if m2t:
        ce = m2t["clasificacion_en_cajas_emparejadas"]
        f1_m2 = [ce["por_clase"][str(i)]["f1"] for i in range(len(NOMBRES))]
        f1_test[E["m2"]] = f1_m2
        filas.append(f"| {E['m2']}* | {_f(ce['exactitud'])} | **{_f(ce['f1_macro'])}** | {' | '.join(_f(v) for v in f1_m2)} | — |")
        if m2v:
            comp.append(f"| {E['m2']}* | {_f(m2v['clasificacion_en_cajas_emparejadas']['f1_macro'])} | {_f(ce['f1_macro'])} |")
    tablas["clasificacion_test_f1_por_clase"] = f1_test
    plt = _plt()
    if plt and f1_test:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ancho = 0.8 / len(f1_test)
        for i, (et, v) in enumerate(f1_test.items()):
            ax.bar(np.arange(len(NOMBRES)) + (i - (len(f1_test) - 1) / 2) * ancho, v, ancho, label=et)
        ax.set_xticks(range(len(NOMBRES)), NOMBRES)
        ax.set_ylabel("F1 (test)")
        ax.set_ylim(0, 1)
        ax.set_title("F1 por estado de madurez en test")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figuras / "8_4_f1_por_clase.png", dpi=150)
        plt.close(fig)
    for k in list(CLASIFICADORES) + ["m2"]:
        png = base / C[k] / "matriz_confusion_test.png"
        if png.is_file():
            shutil.copy2(png, figuras / f"8_4_matriz_confusion_{k}.png")
    disp = base / C["m2"] / "dispersion_conteo_test.png"
    if disp.is_file():
        shutil.copy2(disp, figuras / "8_4_dispersion_conteo_m2.png")

    # ---- M2 detección y conteo con IC
    texto_ic, ic = "", None
    pred = base / C["m2"] / "predicciones_test.jsonl"
    if m2t and pred.is_file() and n_boot > 0:
        pc = B.precalcular(B.leer_predicciones(pred), datos_yaml, "test", m2t.get("conf_reporte", 0.25))
        ic = B.resumen_intervalos(pc, n_boot)
        tablas["m2_intervalos"] = ic
        chequeo = abs(ic["imagen"]["map50"]["punto"] - m2t["map50"])
        if chequeo > 0.01:
            print(
                f"AVISO: la mAP recalculada desde las predicciones ({ic['imagen']['map50']['punto']:.3f}) difiere de la registrada ({m2t['map50']:.3f})"
            )

        def fila(nombre, clave, valor):
            a, b = ic["imagen"][clave], ic["sesion"][clave]
            return f"| {nombre} | {_f(valor)} | [{_f(a['li'])}; {_f(a['ls'])}] | [{_f(b['li'])}; {_f(b['ls'])}] |"

        d = m2t["deteccion"]
        texto_ic = "\n".join(
            [
                fila("mAP@0,5", "map50", m2t["map50"]),
                *(fila(f"AP@0,5 {n}", f"ap50_{n}", m2t["ap50_por_clase"][n]) for n in NOMBRES),
                fila("F1 con clase (conf. 0,25)", "f1_con_clase", d["con_clase"]["f1"]),
                fila("F1 solo localización", "f1_localizacion", d["localizacion"]["f1"]),
                fila(
                    "Exactitud de madurez en cerezas localizadas",
                    "exactitud_cajas_emparejadas",
                    m2t["clasificacion_en_cajas_emparejadas"]["exactitud"],
                ),
                fila(
                    "F1 macro de madurez en cerezas localizadas", "f1_macro_cajas_emparejadas", m2t["clasificacion_en_cajas_emparejadas"]["f1_macro"]
                ),
                fila("Conteo: MAE (cerezas/imagen)", "conteo_mae", m2t["conteo"]["total"]["mae"]),
                fila("Conteo: R² frente al conteo anotado", "conteo_r2", m2t["conteo"]["total"]["r2"]),
            ]
        )
    elif m2t:
        texto_ic = "_Sin intervalos: faltan las predicciones de test o el dataset preparado._"

    tabla_m2 = ""
    if m2t:
        dd, ct = m2t["deteccion"], m2t["conteo"]["total"]
        meta = ic["sesion"]["_meta"] if ic else {}
        tabla_m2 = f"""Tabla 8.4.2 — M2: detección y conteo en test (IC 95 %, bootstrap percentil con {meta.get("remuestreos", "—")} remuestreos;
por imagen y por sesión de captura, {meta.get("sesiones", "—")} sesiones)

| Métrica | Valor | IC 95 % (imágenes) | IC 95 % (sesiones) |
|---|---|---|---|
{texto_ic}

Además: mAP@0,5:0,95 = {_f(m2t["map50_95"])}; precisión/recall con clase {_f(dd["con_clase"]["precision"])}/{_f(dd["con_clase"]["recall"])};
sesgo del conteo {_f(ct["sesgo_medio"], 2)} cerezas/imagen; RMSE {_f(ct["rmse"], 2)}; r² de Pearson {_f(ct["r2_pearson"])};
tiempo de inferencia (mediana) {_f(m2t["tiempo_inferencia_s"]["mediana"] * 1000, 1)} ms/imagen.
El intervalo por sesiones es el más honesto (la sesión es la unidad de la partición), pero con tan pocas sesiones el bootstrap
percentil es poco fiable: léase como orden de magnitud de la incertidumbre.
"""

    # ---- hipótesis
    mejor_m0 = max((_met_clasif(base / C[k], "test") or {"f1_macro": float("nan")})["f1_macro"] for k in ("m0_rf", "m0_svm"))
    m1 = _met_clasif(base / C["m1"], "test")
    m1s = _met_clasif(base / C["m1_sin_preentrenar"], "test")
    hip = []
    if m2t and m1:
        hip.append(
            f"| El aprendizaje profundo supera a la línea base clásica en F1 macro | M1 {_f(m1['f1_macro'])}, M2 {_f(m2t['clasificacion_en_cajas_emparejadas']['f1_macro'])}* vs mejor M0 {_f(mejor_m0)} | {'Sí' if min(m1['f1_macro'], m2t['clasificacion_en_cajas_emparejadas']['f1_macro']) > mejor_m0 else 'No'} (1 semilla) |"
        )
        ic_map = f" (IC sesiones [{_f(ic['sesion']['map50']['li'])}; {_f(ic['sesion']['map50']['ls'])}])" if ic else ""
        hip.append(f"| mAP@0,5 ≥ 0,80 | {_f(m2t['map50'])}{ic_map} | {'Sí' if m2t['map50'] >= 0.80 else 'No'} |")
        r2 = m2t["conteo"]["total"]["r2"]
        ic_r2 = f" (IC sesiones [{_f(ic['sesion']['conteo_r2']['li'])}; {_f(ic['sesion']['conteo_r2']['ls'])}])" if ic else ""
        hip.append(
            f"| R² del conteo ≥ 0,80 | {_f(r2)}{ic_r2} frente al conteo **anotado** | {'Sí' if r2 is not None and r2 >= 0.80 else 'No'} frente a anotaciones; sin conteo manual en campo |"
        )
    if m1 and m1s:
        hip.append(
            f"| Transfer learning mejora la CNN (experimento d) | F1 macro {_f(m1['f1_macro'])} vs {_f(m1s['f1_macro'])} sin preentrenar | {'Sí' if m1['f1_macro'] > m1s['f1_macro'] else 'No'} (1 semilla) |"
        )

    bench = []
    for nombre in cfg.get("benchmarks", []):
        ruta = Path(benchmarks) / nombre if benchmarks else None
        if ruta and ruta.is_file():
            bench.append(coma_decimal(ruta.read_text(encoding="utf-8")).replace("# ", "#### ", 1))
    return f"""## 8.4 Evaluación

Partición de prueba evaluada **una sola vez** con los modelos congelados. Métricas comunes (`src/evaluation/metricas.py`).

Tabla 8.4.1 — Clasificación de madurez en test (mismas cerezas anotadas para M0 y M1)

| Modelo | Exactitud | F1 macro | {" | ".join("F1 " + n for n in NOMBRES)} | ms/recorte |
|---|---|---|{"---|" * len(NOMBRES)}---|
{chr(10).join(filas)}

\\* M2 detecta y clasifica: su exactitud/F1 se mide solo en las cerezas que localiza (IoU ≥ 0,5, conf. ≥ 0,25), con sesgo de selección.
Sin intervalos para M0/M1 (no se guardaron sus predicciones por recorte). Una sola semilla por modelo.

Tabla 8.4.1b — F1 macro en validación (usada para elegir hiperparámetros) y en test

| Modelo | Val | Test |
|---|---|---|
{chr(10).join(comp)}

{tabla_m2}
Tabla 8.4.3 — Contraste con la hipótesis y las metas

| Afirmación | Resultado (test) | ¿Se cumple? |
|---|---|---|
{chr(10).join(hip)}

Figuras: `figuras/8_4_f1_por_clase.png`, `figuras/8_4_matriz_confusion_*.png`, `figuras/8_4_dispersion_conteo_m2.png`.

### Aceleración con CUDA

{chr(10).join(bench) if bench else "_No se encontraron los reportes de benchmark._"}
"""


def _hardware_entrenamiento(clave: str, corrida: Path, ent: dict) -> str:
    """Dónde entrenó el modelo (no solo qué GPU había en el entorno)."""
    gpu = ent.get("gpu")
    if clave.startswith("m0"):
        disp = ((_yaml(corrida / "config.yaml") or {}).get("dispositivo_descriptor") or "cpu").upper()
        base = "CPU (scikit-learn)" + ("" if disp == "CPU" else f"; descriptor en {disp}")
        return base + (f"; {gpu} disponible, no usada para entrenar" if gpu and disp == "CPU" else "")
    if clave.startswith("m1"):
        usa_gpu = (_json(corrida / "tiempos.json") or {}).get("gpu")
        return f"GPU {gpu}" if usa_gpu and gpu else ("GPU" if usa_gpu else "CPU")
    return f"GPU {gpu}" if gpu else "CPU"


def _trazabilidad(cfg: dict, base: Path) -> str:
    filas = []
    for k, id_ in cfg["corridas"].items():
        ent = _json(base / id_ / "entorno.json") or {}
        git = ent.get("git") or {}
        filas.append(
            f"| {cfg['etiquetas'][k]} | `{id_}` | `{(git.get('commit') or '—')[:10]}` | `{(ent.get('version_dataset_sha256') or '—')[:12]}` | "
            f"{_hardware_entrenamiento(k, base / id_, ent)} |"
        )
    cab = "| Modelo | Corrida | Commit | Dataset | Entrenamiento en |\n|---|---|---|---|---|\n"
    return "## Trazabilidad\n\n" + cab + "\n".join(filas) + "\n"


def generar(
    corridas: Path, datos_yaml: Path, salida: Path, benchmarks: Path = None, config: Path = RAIZ / "configs" / "informe.yaml", n_boot: int = 2000
) -> Path:
    cfg = _yaml(Path(config))
    faltan = [id_ for id_ in cfg["corridas"].values() if not (Path(corridas) / id_).is_dir()]
    if faltan:
        raise SystemExit(f"No se encuentran estas corridas en {corridas}: {faltan}")
    figuras = Path(salida) / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    tablas = {}
    partes = [
        "# Resultados para el capítulo 8 (generado)\n\nGenerado por `python -m ml.informe.generar` desde las corridas oficiales de `configs/informe.yaml`. No editar a mano.\n",
        seccion_datos(datos_yaml, figuras),
        seccion_modelos(cfg, Path(corridas), figuras),
        seccion_evaluacion(cfg, Path(corridas), datos_yaml, benchmarks, figuras, n_boot, tablas),
        _trazabilidad(cfg, Path(corridas)),
    ]
    ruta = Path(salida) / "informe_resultados.md"
    ruta.write_text("\n".join(partes), encoding="utf-8")
    (Path(salida) / "tablas.json").write_text(json.dumps(tablas, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return ruta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corridas", type=Path, required=True)
    p.add_argument("--datos", type=Path, required=True, help="data.yaml del dataset preparado (para las etiquetas de test)")
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--benchmarks", type=Path)
    p.add_argument("--config", type=Path, default=RAIZ / "configs" / "informe.yaml")
    p.add_argument("--bootstrap", type=int, default=2000, help="remuestreos para los IC de M2 (0 = sin IC)")
    a = p.parse_args(argv)
    ruta = generar(a.corridas, a.datos, a.salida, a.benchmarks, a.config, a.bootstrap)
    print(ruta.read_text(encoding="utf-8"))
    print(f"\nFiguras en {a.salida / 'figuras'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

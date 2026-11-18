"""
Intervalos de confianza (bootstrap percentil 95 %) para las métricas de M2 en test, a partir de las predicciones
guardadas por ml/m2_yolo/evaluar.py (no se vuelve a ejecutar el modelo ni se toma ninguna decisión con el test).

Dos niveles de remuestreo:
  - por imagen: supone imágenes independientes (optimista: los recortes de una misma sesión se parecen);
  - por sesión de captura: remuestrea las sesiones completas (la unidad de la partición). Con pocas sesiones en test
    (8 en este proyecto) el intervalo es ancho y el bootstrap percentil es poco fiable: se reporta como cota de la
    incertidumbre real, no como intervalo exacto.

Eficiencia: el emparejamiento de cajas se calcula UNA vez por imagen; cada remuestreo solo pondera esas contribuciones
(la AP ponderada con sumas acumuladas equivale a repetir las imágenes; lo comprueba la prueba).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ml.datos.lectura_yolo import NOMBRES, cajas_gt, grupos_por_archivo, imagenes_y_etiquetas
from src.evaluation import metricas as M

N_CLASES = len(NOMBRES)
_PUNTOS = np.linspace(0, 1, 101)


@dataclass
class Precalculo:
    imagenes: List[str]
    grupos: np.ndarray
    # AP@0,5 por clase: entradas ordenadas por confianza descendente
    ap_tp: List[np.ndarray]
    ap_img: List[np.ndarray]
    n_gt: np.ndarray  # (n_imagenes, n_clases)
    # Por imagen, a la confianza de reporte
    tp_clase: np.ndarray
    fp_clase: np.ndarray
    fn_clase: np.ndarray
    tp_loc: np.ndarray
    fp_loc: np.ndarray
    fn_loc: np.ndarray
    conf_emparejadas: np.ndarray  # (n_imagenes, n_clases, n_clases): real x predicho, cajas emparejadas sin clase
    anotado: np.ndarray
    automatico: np.ndarray


def leer_predicciones(ruta_jsonl: Path) -> Dict[str, dict]:
    with open(ruta_jsonl, encoding="utf-8") as fh:
        return {d["imagen"]: d for d in (json.loads(linea) for linea in fh if linea.strip())}


def precalcular(predicciones: Dict[str, dict], data_yaml: Path, split: str = "test", conf_reporte: float = 0.25) -> Precalculo:
    grupos_de = grupos_por_archivo(data_yaml)
    nombres, grupos = [], []
    ap_conf = [[] for _ in range(N_CLASES)]
    ap_tp = [[] for _ in range(N_CLASES)]
    ap_img = [[] for _ in range(N_CLASES)]
    filas = []
    from PIL import Image

    for k, (ruta, lbl, a_estado) in enumerate(imagenes_y_etiquetas(data_yaml, split)):
        if ruta.name not in predicciones:
            raise ValueError(f"No hay predicciones para {ruta.name}: ¿son del mismo dataset y partición?")
        with Image.open(ruta) as im:
            ancho, alto = im.size
        gt, gc = cajas_gt(lbl, ancho, alto, a_estado)
        p = predicciones[ruta.name]
        pc = np.asarray(p["cajas"], float).reshape(-1, 4)
        pf = np.asarray(p["conf"], float)
        pk = np.asarray(p["clase"], int)
        n_gt = np.bincount(gc, minlength=N_CLASES)
        for c in range(N_CLASES):
            m = pk == c
            emp = {i for i, _ in M.emparejar(pc[m], pf[m], gt[gc == c], 0.5)}
            ap_conf[c].append(pf[m])
            ap_tp[c].append(np.array([i in emp for i in range(int(m.sum()))], bool))
            ap_img[c].append(np.full(int(m.sum()), k))
        keep = pf >= conf_reporte - 1e-9
        kc, kf, kk = pc[keep], pf[keep], pk[keep]
        pares_loc = M.emparejar(kc, kf, gt, 0.5)
        pares_cls = M.emparejar(kc, kf, gt, 0.5, kk, gc)
        conf = np.zeros((N_CLASES, N_CLASES), int)
        for i, j in pares_loc:
            conf[gc[j], kk[i]] += 1
        filas.append(
            (
                len(pares_cls),
                len(kk) - len(pares_cls),
                len(gc) - len(pares_cls),
                len(pares_loc),
                len(kk) - len(pares_loc),
                len(gc) - len(pares_loc),
                conf,
                len(gc),
                len(kk),
                n_gt,
            )
        )
        nombres.append(ruta.name)
        grupos.append(grupos_de.get(ruta.name, ruta.stem))
    ap_tp_s, ap_img_s = [], []
    for c in range(N_CLASES):
        conf_c = np.concatenate(ap_conf[c]) if ap_conf[c] else np.zeros(0)
        orden = np.argsort(-conf_c, kind="stable")
        ap_tp_s.append(np.concatenate(ap_tp[c])[orden] if ap_tp[c] else np.zeros(0, bool))
        ap_img_s.append(np.concatenate(ap_img[c])[orden] if ap_img[c] else np.zeros(0, int))
    col = list(zip(*filas))
    return Precalculo(
        imagenes=nombres,
        grupos=np.asarray(grupos),
        ap_tp=ap_tp_s,
        ap_img=ap_img_s,
        n_gt=np.stack(col[9]),
        tp_clase=np.asarray(col[0]),
        fp_clase=np.asarray(col[1]),
        fn_clase=np.asarray(col[2]),
        tp_loc=np.asarray(col[3]),
        fp_loc=np.asarray(col[4]),
        fn_loc=np.asarray(col[5]),
        conf_emparejadas=np.stack(col[6]),
        anotado=np.asarray(col[7]),
        automatico=np.asarray(col[8]),
    )


def ap_ponderada(tp_ordenado: np.ndarray, peso_entrada: np.ndarray, n_gt: float) -> float:
    """AP (101 puntos) de entradas ya ordenadas por confianza, cada una con un peso entero (veces que se repite su imagen)."""
    if n_gt <= 0:
        return float("nan")
    m = peso_entrada > 0
    if not m.any():
        return 0.0
    w, tp = peso_entrada[m].astype(float), tp_ordenado[m]
    tp_acum = np.cumsum(w * tp)
    fp_acum = np.cumsum(w * (~tp))
    recall = tp_acum / n_gt
    precision = tp_acum / (tp_acum + fp_acum)
    envolvente = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, _PUNTOS, side="left")
    return float(np.mean([envolvente[i] if i < len(envolvente) else 0.0 for i in idx]))


def metricas_ponderadas(pc: Precalculo, w: np.ndarray) -> Dict[str, float]:
    """Métricas de M2 cuando la imagen i aparece w[i] veces."""
    n_gt = (pc.n_gt * w[:, None]).sum(0)
    aps = [ap_ponderada(pc.ap_tp[c], w[pc.ap_img[c]], n_gt[c]) for c in range(N_CLASES)]
    aps_validas = [a for a in aps if not np.isnan(a)]

    def f1(tp, fp, fn):
        tp, fp, fn = (float((x * w).sum()) for x in (tp, fp, fn))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    conf = (pc.conf_emparejadas * w[:, None, None]).sum(0)
    clas = M.metricas_clasificacion_desde_matriz(conf)
    conteo = M.metricas_conteo(np.repeat(pc.anotado, w), np.repeat(pc.automatico, w))
    salida = {
        "map50": float(np.mean(aps_validas)) if aps_validas else 0.0,
        "f1_con_clase": f1(pc.tp_clase, pc.fp_clase, pc.fn_clase),
        "f1_localizacion": f1(pc.tp_loc, pc.fp_loc, pc.fn_loc),
        "exactitud_cajas_emparejadas": clas["exactitud"],
        "f1_macro_cajas_emparejadas": clas["f1_macro"],
        "conteo_mae": conteo["mae"],
        "conteo_r2": conteo["r2"] if conteo["r2"] is not None else float("nan"),
    }
    for c, n in enumerate(NOMBRES):
        salida[f"ap50_{n}"] = aps[c]
    return salida


def intervalos(pc: Precalculo, n: int = 2000, semilla: int = 0, nivel: str = "imagen", alfa: float = 0.05) -> Dict[str, dict]:
    """{métrica: {punto, li, ls}} con bootstrap percentil; nivel = 'imagen' o 'sesion'."""
    rng = np.random.default_rng(semilla)
    n_img = len(pc.imagenes)
    punto = metricas_ponderadas(pc, np.ones(n_img, int))
    if nivel == "imagen":

        def muestra():
            return np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
    elif nivel == "sesion":
        sesiones, de_imagen = np.unique(pc.grupos, return_inverse=True)

        def muestra():
            cuenta = np.bincount(rng.integers(0, len(sesiones), len(sesiones)), minlength=len(sesiones))
            return cuenta[de_imagen]
    else:
        raise ValueError(nivel)
    valores = {k: [] for k in punto}
    for _ in range(n):
        for k, v in metricas_ponderadas(pc, muestra()).items():
            valores[k].append(v)
    res = {}
    for k, v in valores.items():
        arr = np.asarray(v, float)
        arr = arr[~np.isnan(arr)]
        li, ls = np.quantile(arr, [alfa / 2, 1 - alfa / 2]) if len(arr) else (np.nan, np.nan)
        res[k] = {"punto": punto[k], "li": float(li), "ls": float(ls)}
    res["_meta"] = {
        "nivel": nivel,
        "remuestreos": n,
        "semilla": semilla,
        "imagenes": n_img,
        "sesiones": int(len(np.unique(pc.grupos))),
        "metodo": "bootstrap percentil 95 %",
    }
    return res


def resumen_intervalos(pc: Precalculo, n: int = 2000, semilla: int = 0) -> Dict[str, Optional[dict]]:
    return {"imagen": intervalos(pc, n, semilla, "imagen"), "sesion": intervalos(pc, n, semilla, "sesion")}

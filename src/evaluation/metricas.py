"""
Métricas de evaluación de SIKAFE (puras, solo numpy). Una sola implementación
común para M0, M1 y M2, para que las comparaciones sean justas (ADR-007).

Convenciones
- Cajas en formato xyxy (píxeles o normalizadas, pero las mismas en pred y gt).
- Emparejamiento tipo COCO: las predicciones se recorren de mayor a menor
  confianza y cada una toma el gt libre con mayor IoU (>= umbral).
- AP: interpolación de 101 puntos de recall (como COCO). mAP = media de AP
  sobre las clases con al menos un gt.
- Matriz de confusión de detección: emparejamiento SIN considerar la clase;
  fila = clase real, columna = clase predicha; el último índice es "fondo"
  (fila fondo = falsos positivos, columna fondo = cerezas no detectadas).
- Conteo: R² es el coeficiente de determinación del conteo automático como
  predicción del manual (1 - SS_res/SS_tot, puede ser negativo); también se
  da r² de Pearson (correlación al cuadrado), que mide asociación lineal.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

UMBRALES_COCO = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.50:0.05:0.95


@dataclass
class ImagenEval:
    """Predicciones y verdad de terreno de UNA imagen."""

    pred_cajas: np.ndarray  # (N, 4) xyxy
    pred_conf: np.ndarray  # (N,)
    pred_clase: np.ndarray  # (N,) int
    gt_cajas: np.ndarray  # (M, 4)
    gt_clase: np.ndarray  # (M,) int

    @staticmethod
    def crear(pred_cajas, pred_conf, pred_clase, gt_cajas, gt_clase) -> "ImagenEval":
        return ImagenEval(
            np.asarray(pred_cajas, dtype=float).reshape(-1, 4),
            np.asarray(pred_conf, dtype=float).reshape(-1),
            np.asarray(pred_clase, dtype=int).reshape(-1),
            np.asarray(gt_cajas, dtype=float).reshape(-1, 4),
            np.asarray(gt_clase, dtype=int).reshape(-1),
        )


# ------------------------------------------------------------------ geometría
def iou_matriz(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def emparejar(pred_cajas, pred_conf, gt_cajas, umbral_iou: float, pred_clase=None, gt_clase=None) -> List[tuple]:
    """Pares (i_pred, j_gt). Con clases, solo se emparejan predicciones y gt de la misma clase."""
    iou = iou_matriz(pred_cajas, gt_cajas)
    if pred_clase is not None and gt_clase is not None and iou.size:
        iou = np.where(np.asarray(pred_clase)[:, None] == np.asarray(gt_clase)[None, :], iou, 0.0)
    libres = np.ones(iou.shape[1], dtype=bool)
    pares = []
    for i in np.argsort(-np.asarray(pred_conf, dtype=float), kind="stable"):
        if not libres.any():
            break
        candidatos = np.where(libres, iou[i], -1.0)
        j = int(np.argmax(candidatos))
        if candidatos[j] >= umbral_iou:
            pares.append((int(i), j))
            libres[j] = False
    return pares


# ------------------------------------------------------------------ detección
def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def metricas_deteccion(imagenes: Sequence[ImagenEval], umbral_iou: float = 0.5, umbral_conf: float = 0.25, n_clases: int = 4) -> dict:
    """P/R/F1 a un umbral de confianza fijo: (a) solo localización, sin importar la clase
    (lo que importa para CONTAR cerezas) y (b) exigiendo además la clase correcta."""
    tp_a = fp_a = fn_a = tp_c = fp_c = fn_c = 0
    por_clase = {c: [0, 0, 0] for c in range(n_clases)}
    for im in imagenes:
        keep = im.pred_conf >= umbral_conf - 1e-9
        pc, pf, pk = im.pred_cajas[keep], im.pred_conf[keep], im.pred_clase[keep]
        n_tp = len(emparejar(pc, pf, im.gt_cajas, umbral_iou))
        tp_a, fp_a, fn_a = tp_a + n_tp, fp_a + len(pc) - n_tp, fn_a + len(im.gt_cajas) - n_tp
        pares = emparejar(pc, pf, im.gt_cajas, umbral_iou, pk, im.gt_clase)
        tp_c, fp_c, fn_c = tp_c + len(pares), fp_c + len(pc) - len(pares), fn_c + len(im.gt_cajas) - len(pares)
        emparejadas_pred = {i for i, _ in pares}
        emparejadas_gt = {j for _, j in pares}
        for c in range(n_clases):
            por_clase[c][0] += sum(1 for i, _ in pares if pk[i] == c)
            por_clase[c][1] += sum(1 for i in range(len(pk)) if pk[i] == c and i not in emparejadas_pred)
            por_clase[c][2] += sum(1 for j in range(len(im.gt_clase)) if im.gt_clase[j] == c and j not in emparejadas_gt)
    return {
        "umbral_iou": umbral_iou,
        "umbral_conf": umbral_conf,
        "localizacion": _prf(tp_a, fp_a, fn_a),
        "con_clase": _prf(tp_c, fp_c, fn_c),
        "por_clase": {c: _prf(*v) for c, v in por_clase.items()},
    }


def average_precision(confianzas, es_tp, n_gt: int) -> float:
    """AP con interpolación de 101 puntos de recall (COCO)."""
    if n_gt == 0:
        return float("nan")
    if len(confianzas) == 0:
        return 0.0
    orden = np.argsort(-np.asarray(confianzas, dtype=float), kind="stable")
    tp = np.asarray(es_tp, dtype=float)[orden]
    tp_acum, fp_acum = np.cumsum(tp), np.cumsum(1 - tp)
    recall = tp_acum / n_gt
    precision = tp_acum / (tp_acum + fp_acum)
    envolvente = np.maximum.accumulate(precision[::-1])[::-1]  # precisión máxima a recall >= r
    puntos = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, puntos, side="left")
    return float(np.mean([envolvente[i] if i < len(envolvente) else 0.0 for i in idx]))


def mean_average_precision(imagenes: Sequence[ImagenEval], n_clases: int = 4, umbrales_iou: Sequence[float] = (0.5,)) -> dict:
    """mAP (media sobre clases con gt) para cada umbral IoU y su promedio."""
    ap = np.full((len(umbrales_iou), n_clases), np.nan)
    for t, thr in enumerate(umbrales_iou):
        for c in range(n_clases):
            confs, tps, n_gt = [], [], 0
            for im in imagenes:
                mp, mg = im.pred_clase == c, im.gt_clase == c
                n_gt += int(mg.sum())
                pc, pf = im.pred_cajas[mp], im.pred_conf[mp]
                emparejadas = {i for i, _ in emparejar(pc, pf, im.gt_cajas[mg], thr)}
                confs.extend(pf.tolist())
                tps.extend(i in emparejadas for i in range(len(pf)))
            ap[t, c] = average_precision(confs, tps, n_gt)
    with np.errstate(all="ignore"):
        map_por_umbral = np.nanmean(ap, axis=1) if not np.all(np.isnan(ap)) else np.zeros(len(umbrales_iou))
    return {
        "umbrales_iou": list(umbrales_iou),
        "ap_por_clase": {c: [None if np.isnan(v) else float(v) for v in ap[:, c]] for c in range(n_clases)},
        "map_por_umbral": [float(v) for v in map_por_umbral],
        "map": float(np.mean(map_por_umbral)),
    }


def matriz_confusion_deteccion(imagenes: Sequence[ImagenEval], n_clases: int = 4, umbral_iou: float = 0.5, umbral_conf: float = 0.25) -> np.ndarray:
    """(n+1)x(n+1): filas = clase real (+ fondo), columnas = clase predicha (+ fondo)."""
    fondo = n_clases
    m = np.zeros((n_clases + 1, n_clases + 1), dtype=int)
    for im in imagenes:
        keep = im.pred_conf >= umbral_conf - 1e-9
        pc, pf, pk = im.pred_cajas[keep], im.pred_conf[keep], im.pred_clase[keep]
        pares = emparejar(pc, pf, im.gt_cajas, umbral_iou)
        for i, j in pares:
            m[im.gt_clase[j], pk[i]] += 1
        for j in set(range(len(im.gt_clase))) - {j for _, j in pares}:
            m[im.gt_clase[j], fondo] += 1
        for i in set(range(len(pk))) - {i for i, _ in pares}:
            m[fondo, pk[i]] += 1
    return m


# ------------------------------------------------------------------ clasificación (recortes: M0/M1)
def metricas_clasificacion(y_real, y_pred, n_clases: int = 4) -> dict:
    y_real, y_pred = np.asarray(y_real, dtype=int), np.asarray(y_pred, dtype=int)
    m = np.zeros((n_clases, n_clases), dtype=int)
    np.add.at(m, (y_real, y_pred), 1)
    por_clase = {}
    for c in range(n_clases):
        tp, fp, fn = int(m[c, c]), int(m[:, c].sum() - m[c, c]), int(m[c, :].sum() - m[c, c])
        por_clase[c] = {**_prf(tp, fp, fn), "soporte": int(m[c, :].sum())}
    con_soporte = [v for v in por_clase.values() if v["soporte"] > 0]
    return {
        "exactitud": float(np.trace(m) / m.sum()) if m.sum() else 0.0,
        "f1_macro": float(np.mean([v["f1"] for v in con_soporte])) if con_soporte else 0.0,
        "por_clase": por_clase,
        "matriz_confusion": m.tolist(),
    }


# ------------------------------------------------------------------ conteo
def metricas_conteo(manual, automatico) -> dict:
    y = np.asarray(manual, dtype=float)
    x = np.asarray(automatico, dtype=float)
    if y.shape != x.shape or y.size == 0:
        raise ValueError("manual y automatico deben tener la misma longitud (> 0)")
    err = x - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2: Optional[float] = 1 - ss_res / ss_tot if ss_tot > 0 else None
    pearson = float(np.corrcoef(x, y)[0, 1] ** 2) if y.size > 1 and x.std() > 0 and y.std() > 0 else None
    return {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "sesgo_medio": float(np.mean(err)),
        "r2": r2,
        "r2_pearson": pearson,
    }

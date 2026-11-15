"""
Entrenamiento y evaluacion del clasificador de madurez.

Uso:
    python -m ml.sintetico.training                      # datos sinteticos
    python -m ml.sintetico.training --imagenes datos/    # + fotos reales

Vive en ml/ (laboratorio), no en src/ (app): la app solo CARGA modelos y nunca
reentrena por su cuenta. Al guardar se escribe tambien el manifiesto con el
SHA-256 que la app verifica antes de cargar.

Fotos reales: en la carpeta, cada `foto.jpg` necesita su mascara `foto_mask.png`
(imagen de un canal; cada pixel vale 0=fondo, 1=verde, 2=pinton, 3=maduro,
4=sobremaduro). Se mezclan con las escenas sinteticas; use --sin-sinteticas
para entrenar solo con las reales cuando tenga suficientes.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from src.infrastructure.ml.features import CLASES, LADO_TRABAJO, extraer_caracteristicas, redimensionar_para_trabajo
from src.infrastructure.ml.modelo import ModeloMadurezPixel

from .synthetic_data import frutos_visibles, generar_escena

RUTA_MODELO_POR_DEFECTO = Path(__file__).resolve().parents[2] / "models" / "madurez_pixel.joblib"

_SEMILLA_VALIDACION = 1_000
_SEMILLA_PRUEBA = 2_000


def _muestrear_pixeles(rgb, etiquetas, rng, por_clase_max) -> Tuple[np.ndarray, np.ndarray]:
    X = extraer_caracteristicas(rgb)
    y = etiquetas.ravel()
    elegidos = []
    for clase in range(len(CLASES)):
        indices = np.flatnonzero(y == clase)
        if indices.size == 0:
            continue
        cupo = por_clase_max * (2 if clase == 0 else 1)  # el fondo es mas variado
        if indices.size > cupo:
            indices = rng.choice(indices, cupo, replace=False)
        elegidos.append(indices)
    elegidos = np.concatenate(elegidos)
    return X[elegidos], y[elegidos]


def _dataset_sintetico(n_escenas, semilla, por_clase_max):
    rng = np.random.default_rng(semilla)
    Xs, ys = [], []
    for _ in range(n_escenas):
        escena = generar_escena(rng)
        X, y = _muestrear_pixeles(escena.rgb, escena.etiquetas, rng, por_clase_max)
        Xs.append(X)
        ys.append(y)
    return Xs, ys


def _dataset_real(carpeta: Path, semilla, por_clase_max):
    rng = np.random.default_rng(semilla)
    Xs, ys = [], []
    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in (".jpg", ".jpeg", ".png") or ruta.stem.endswith("_mask"):
            continue
        ruta_mascara = ruta.with_name(f"{ruta.stem}_mask.png")
        if not ruta_mascara.is_file():
            print(f"  (se omite {ruta.name}: falta {ruta_mascara.name})")
            continue
        rgb, escala = redimensionar_para_trabajo(Image.open(ruta), LADO_TRABAJO)
        mascara = Image.open(ruta_mascara).convert("L").resize(rgb.shape[1::-1], Image.NEAREST)
        etiquetas = np.asarray(mascara, dtype=np.int64)
        if etiquetas.max() >= len(CLASES):
            raise ValueError(f"{ruta_mascara.name}: valores de mascara fuera de 0..{len(CLASES) - 1}")
        X, y = _muestrear_pixeles(rgb, etiquetas, rng, por_clase_max)
        Xs.append(X)
        ys.append(y)
    return Xs, ys


def entrenar(
    n_escenas: int = 600,
    semilla: int = 0,
    por_clase_max: int = 200,
    carpeta_real: Optional[Path] = None,
    usar_sinteticas: bool = True,
    peso_real: int = 3,
) -> ModeloMadurezPixel:
    Xs, ys = [], []
    if usar_sinteticas:
        Xs, ys = _dataset_sintetico(n_escenas, semilla, por_clase_max)
    n_real = 0
    if carpeta_real is not None:
        Xr, yr = _dataset_real(Path(carpeta_real), semilla, por_clase_max)
        n_real = len(Xr)
        for _ in range(peso_real):  # las fotos reales pesan mas que las sinteticas
            Xs += Xr
            ys += yr
    if not Xs:
        raise ValueError("No hay datos de entrenamiento.")

    X, y = np.concatenate(Xs), np.concatenate(ys)
    clasificador = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.1, random_state=semilla,
    )
    clasificador.fit(X, y)
    return ModeloMadurezPixel(
        clasificador,
        metadatos={
            "clases": list(CLASES),
            "escenas_sinteticas": n_escenas if usar_sinteticas else 0,
            "imagenes_reales": n_real,
            "muestras_pixel": int(len(y)),
            "semilla": semilla,
        },
    )


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluar(
    modelo: ModeloMadurezPixel,
    n_escenas: int = 80,
    semilla: int = _SEMILLA_PRUEBA,
    umbral: float = 0.40,
    iou_min: float = 0.30,
) -> Dict[str, float]:
    """
    Evalua el pipeline completo (clasificacion + segmentacion + deteccion) sobre
    escenas sinteticas NUEVAS (semilla distinta a la de entrenamiento).
    Es una medida optimista: las escenas salen del mismo generador que el entrenamiento.
    """
    from src.infrastructure.adapters.detectors.ml_detector_adapter import MLFruitDetectorAdapter

    detector = MLFruitDetectorAdapter(modelo)
    rng = np.random.default_rng(semilla)
    px_ok = px_total = tp = fp = fn = estado_ok = 0
    err_rel: List[float] = []
    total_det = total_gt = 0

    for _ in range(n_escenas):
        escena = generar_escena(rng)
        px_ok += int((modelo.predecir_proba(escena.rgb).argmax(-1) == escena.etiquetas).sum())
        px_total += escena.etiquetas.size

        gt = frutos_visibles(escena)
        dets = sorted(
            detector.detectar_frutos(Image.fromarray(escena.rgb), umbral),
            key=lambda d: -d.confianza,
        )
        libres = set(range(len(gt)))
        for d in dets:
            caja = (d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max)
            mejor, mejor_iou = None, iou_min
            for j in libres:
                v = _iou(caja, gt[j]["bbox"])
                if v >= mejor_iou:
                    mejor, mejor_iou = j, v
            if mejor is None:
                fp += 1
            else:
                libres.discard(mejor)
                tp += 1
                estado_ok += int(CLASES.index(d.estado.value) == gt[mejor]["estado"])
        fn += len(libres)
        total_det += len(dets)
        total_gt += len(gt)
        err_rel.append(abs(len(dets) - len(gt)) / max(len(gt), 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "exactitud_pixel": px_ok / px_total,
        "precision_deteccion": precision,
        "recall_deteccion": recall,
        "f1_deteccion": 2 * precision * recall / max(precision + recall, 1e-9),
        "exactitud_estado_en_detectados": estado_ok / max(tp, 1),
        "error_relativo_conteo_por_escena": float(np.mean(err_rel)),
        "frutos_detectados_sobre_reales": total_det / max(total_gt, 1),
        "escenas_evaluadas": n_escenas,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Entrena el clasificador de madurez del cafe.")
    p.add_argument("--escenas", type=int, default=600, help="escenas sinteticas de entrenamiento")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--imagenes", type=Path, default=None, help="carpeta con fotos reales + *_mask.png")
    p.add_argument("--sin-sinteticas", action="store_true", help="no usar escenas sinteticas")
    p.add_argument("--salida", type=Path, default=RUTA_MODELO_POR_DEFECTO)
    args = p.parse_args(argv)

    t0 = time.time()
    modelo = entrenar(
        n_escenas=args.escenas, semilla=args.semilla,
        carpeta_real=args.imagenes, usar_sinteticas=not args.sin_sinteticas,
    )
    print(f"Entrenado en {time.time() - t0:.1f}s con {modelo.metadatos['muestras_pixel']} muestras de pixel.")
    metricas = evaluar(modelo)
    modelo.metadatos["metricas_sinteticas"] = metricas
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f"Modelo guardado en: {modelo.guardar(args.salida)}")


if __name__ == "__main__":
    main()

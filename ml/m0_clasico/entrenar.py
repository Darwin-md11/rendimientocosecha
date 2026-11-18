"""
M0 — línea base clásica: color (HSV/Lab) + textura (LBP/HOG) + Random Forest o SVM, sobre los
recortes de las cajas reales (anotadas) del mismo dataset y la misma partición que M2.

Uso (desde la raíz del repo; CPU basta):
    python -m ml.m0_clasico.entrenar --datos /content/sikafe_yolo/data.yaml --salida <corridas> --modelo rf
    python -m ml.m0_clasico.entrenar --datos /content/sikafe_yolo/data.yaml --salida <corridas> --modelo svm
    python -m ml.m0_clasico.entrenar --corrida <corridas>/<id> --final          # test, UNA sola vez

Protocolo:
  - Hiperparámetros elegidos con validación cruzada POR GRUPOS (sesión de captura) dentro de train.
  - Train se submuestrea de forma estratificada (--max-train, por defecto 30 000 recortes) para que
    SVM y RF entrenen en minutos en CPU; queda registrado como limitación.
  - Se reporta en val; el test se toca una sola vez con --final (queda registrado).

Salidas en <corridas>/<id>: config.yaml, entorno.json, cv.json, modelo_m0.joblib + manifiesto,
metricas_val.json, resumen_val.md, matriz_confusion_val.png, tiempos.json.
Las características se guardan en caché (<corridas>/cache_m0/) y se reutilizan entre corridas.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import sklearn
import yaml
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.datos.lectura_yolo import NOMBRES, cajas_gt, grupos_por_archivo, imagenes_y_etiquetas, version_dataset
from ml.experiments.registro import capturar_entorno, guardar_json, id_corrida
from ml.m0_clasico.caracteristicas import DIMENSION, VERSION_DESCRIPTOR, caracteristicas_recorte, recortar
from src.evaluation import metricas as M
from src.infrastructure.ml.manifiesto import ruta_manifiesto, sha256_archivo, verificar_archivo

GRILLAS: Dict[str, List[dict]] = {
    "rf": [{"n_estimators": 150, "max_depth": d, "min_samples_leaf": hoja} for d in (None, 25) for hoja in (1, 3)],
    "svm": [{"C": c} for c in (1.0, 10.0)],
}


# ------------------------------------------------------------------ características
def extraer_split(data_yaml: Path, split: str, margen: float = 0.10, lado_min: int = 3) -> dict:
    grupos = grupos_por_archivo(data_yaml)
    X, y, g, imgs = [], [], [], []
    omitidos = 0
    t0 = time.perf_counter()
    for ruta, lbl, a_estado in imagenes_y_etiquetas(data_yaml, split):
        with Image.open(ruta) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        cajas, clases = cajas_gt(lbl, rgb.shape[1], rgb.shape[0], a_estado)
        for caja, c in zip(cajas, clases):
            rec = recortar(rgb, caja, margen)
            if min(rec.shape[:2]) < lado_min:
                omitidos += 1
                continue
            X.append(caracteristicas_recorte(rec))
            y.append(int(c))
            g.append(grupos.get(ruta.name, ruta.stem))
            imgs.append(ruta.name)
    dt = time.perf_counter() - t0
    X = np.stack(X) if X else np.zeros((0, DIMENSION), np.float32)
    return {
        "X": X,
        "y": np.asarray(y, int),
        "grupos": np.asarray(g),
        "imagenes": np.asarray(imgs),
        "omitidos": omitidos,
        "segundos": dt,
        "recortes_por_segundo": len(y) / dt if dt else None,
    }


def caracteristicas_con_cache(data_yaml: Path, split: str, carpeta_cache: Path) -> dict:
    nombre = f"{version_dataset(data_yaml)[:12]}_desc{VERSION_DESCRIPTOR}_{split}.npz"
    ruta = carpeta_cache / nombre
    if ruta.is_file():
        d = np.load(ruta, allow_pickle=False)
        print(f"Características de {split} desde caché: {ruta}")
        return {k: d[k] for k in ("X", "y", "grupos", "imagenes")} | {
            "omitidos": int(d["omitidos"]),
            "segundos": float(d["segundos"]),
            "recortes_por_segundo": float(d["recortes_por_segundo"]),
            "cache": True,
        }
    print(f"Extrayendo características de {split}…")
    r = extraer_split(data_yaml, split)
    carpeta_cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ruta,
        X=r["X"],
        y=r["y"],
        grupos=r["grupos"],
        imagenes=r["imagenes"],
        omitidos=r["omitidos"],
        segundos=r["segundos"],
        recortes_por_segundo=r["recortes_por_segundo"] or 0.0,
    )
    print(f"  {len(r['y'])} recortes en {r['segundos']:.1f} s ({r['recortes_por_segundo']:.0f} recortes/s); omitidos {r['omitidos']}")
    return r | {"cache": False}


# ------------------------------------------------------------------ modelo
def construir(modelo: str, params: dict, semilla: int):
    if modelo == "rf":
        return RandomForestClassifier(**params, class_weight="balanced_subsample", n_jobs=-1, random_state=semilla)
    if modelo == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced", random_state=semilla, **params))
    raise ValueError(f"modelo desconocido: {modelo}")


def submuestrear(y: np.ndarray, maximo: int, semilla: int) -> np.ndarray:
    """Índices de una submuestra estratificada por clase (todo si len(y) <= maximo)."""
    if len(y) <= maximo:
        return np.arange(len(y))
    rng = np.random.default_rng(semilla)
    idx = []
    for c in np.unique(y):
        de_c = np.flatnonzero(y == c)
        n = max(1, round(maximo * len(de_c) / len(y)))
        idx.append(rng.choice(de_c, size=min(n, len(de_c)), replace=False))
    return np.sort(np.concatenate(idx))


def validacion_cruzada(modelo: str, X, y, grupos, semilla: int, folds: int = 3) -> dict:
    """Elige hiperparámetros por F1 macro con GroupKFold (ningún grupo en dos pliegues)."""
    folds = min(folds, len(np.unique(grupos)))
    resultados = []
    for params in GRILLAS[modelo]:
        t0 = time.perf_counter()
        f1s = []
        for tr, va in GroupKFold(n_splits=folds).split(X, y, grupos):
            clf = construir(modelo, params, semilla).fit(X[tr], y[tr])
            f1s.append(f1_score(y[va], clf.predict(X[va]), average="macro", labels=sorted(np.unique(y))))
        resultados.append(
            {
                "params": params,
                "f1_macro_media": float(np.mean(f1s)),
                "f1_macro_desv": float(np.std(f1s)),
                "f1_por_pliegue": [float(v) for v in f1s],
                "segundos": time.perf_counter() - t0,
            }
        )
        print(f"  {params}: F1 macro {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    mejor = max(resultados, key=lambda r: r["f1_macro_media"])
    return {"pliegues": folds, "criterio": "f1_macro (GroupKFold por sesión)", "resultados": resultados, "mejor": mejor["params"]}


# ------------------------------------------------------------------ reporte
def _evaluar(clf, d: dict) -> dict:
    t0 = time.perf_counter()
    pred = clf.predict(d["X"]) if len(d["y"]) else np.zeros(0, int)
    dt = time.perf_counter() - t0
    met = M.metricas_clasificacion(d["y"], pred, len(NOMBRES))
    met["clases"] = NOMBRES
    met["recortes"] = int(len(d["y"]))
    met["ms_por_recorte_prediccion"] = 1000 * dt / max(1, len(d["y"]))
    return met


def _resumen_md(corrida: Path, split: str, met: dict, cfg: dict) -> str:
    filas = "\n".join(
        f"| {n} | {met['por_clase'][i]['precision']:.3f} | {met['por_clase'][i]['recall']:.3f} | {met['por_clase'][i]['f1']:.3f} | {met['por_clase'][i]['soporte']} |"
        for i, n in enumerate(NOMBRES)
    )
    return f"""# Evaluación M0 ({cfg["modelo"].upper()}) — corrida `{corrida.name}` — partición `{split}`

Recortes de cajas anotadas: {met["recortes"]} · descriptor v{VERSION_DESCRIPTOR} ({DIMENSION} dimensiones: HSV, Lab, LBP, HOG)
Hiperparámetros (validación cruzada por sesión en train): `{cfg["params"]}` · train submuestreado a {cfg["n_train"]} recortes

| Métrica | Valor |
|---|---|
| Exactitud | {met["exactitud"]:.3f} |
| F1 macro | {met["f1_macro"]:.3f} |
| Predicción (ms por recorte, CPU) | {met["ms_por_recorte_prediccion"]:.3f} |

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
{filas}

Generado por `python -m ml.m0_clasico.entrenar` (no editar a mano).
"""


def _grafico(corrida: Path, split: str, met: dict) -> None:
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
    ax.set_title(f"M0 – matriz de confusión ({split})")
    fig.tight_layout()
    fig.savefig(corrida / f"matriz_confusion_{split}.png", dpi=150)
    plt.close(fig)


def _guardar_reporte(corrida: Path, split: str, met: dict, cfg: dict) -> str:
    guardar_json(corrida / f"metricas_{split}.json", met)
    resumen = _resumen_md(corrida, split, met, cfg)
    (corrida / f"resumen_{split}.md").write_text(resumen, encoding="utf-8")
    _grafico(corrida, split, met)
    return resumen


# ------------------------------------------------------------------ flujo
def entrenar(datos: Path, salida: Path, modelo: str, semilla: int = 0, max_train: int = 30000, folds: int = 3) -> Path:
    cache = Path(salida) / "cache_m0"
    tr = caracteristicas_con_cache(datos, "train", cache)
    va = caracteristicas_con_cache(datos, "val", cache)
    if len(np.unique(tr["y"])) < 2:
        raise SystemExit("Train tiene menos de 2 clases: no se puede entrenar M0.")

    corrida = Path(salida) / id_corrida(f"m0_{modelo}", semilla)
    corrida.mkdir(parents=True, exist_ok=False)
    guardar_json(corrida / "entorno.json", capturar_entorno(Path(datos)))
    idx = submuestrear(tr["y"], max_train, semilla)
    X, y, g = tr["X"][idx], tr["y"][idx], tr["grupos"][idx]

    print(f"Validación cruzada por sesión ({modelo}, {len(y)} recortes, {len(np.unique(g))} grupos)…")
    t0 = time.perf_counter()
    cv = validacion_cruzada(modelo, X, y, g, semilla, folds)
    t_cv = time.perf_counter() - t0
    guardar_json(corrida / "cv.json", cv)

    t0 = time.perf_counter()
    clf = construir(modelo, cv["mejor"], semilla).fit(X, y)
    t_fit = time.perf_counter() - t0
    ruta_modelo = corrida / "modelo_m0.joblib"
    joblib.dump(clf, ruta_modelo, compress=3)

    cfg = {
        "modelo": modelo,
        "params": cv["mejor"],
        "semilla": semilla,
        "max_train": max_train,
        "n_train": int(len(y)),
        "folds": cv["pliegues"],
        "datos": str(datos),
        "descriptor_version": VERSION_DESCRIPTOR,
        "dimension": DIMENSION,
    }
    (corrida / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    guardar_json(
        ruta_manifiesto(ruta_modelo),
        {
            "archivo": ruta_modelo.name,
            "sha256": sha256_archivo(ruta_modelo),
            "familia": "M0",
            "version_sklearn": sklearn.__version__,
            "descriptor_version": VERSION_DESCRIPTOR,
            "clases": NOMBRES,
            "params": cv["mejor"],
            "datos_entrenamiento": f"recortes de cajas anotadas, partición train ({len(y)} submuestreados), dataset {version_dataset(datos)[:12]}",
            "validado_con_datos_reales": False,
            "corrida_id": corrida.name,
        },
    )
    guardar_json(
        corrida / "tiempos.json",
        {
            "extraccion_train_s": tr["segundos"],
            "extraccion_val_s": va["segundos"],
            "recortes_por_segundo_extraccion": tr["recortes_por_segundo"],
            "validacion_cruzada_s": t_cv,
            "entrenamiento_final_s": t_fit,
            "nota": "extracción en un solo núcleo de CPU (línea base para CUDA/MPI)",
        },
    )
    print(_guardar_reporte(corrida, "val", _evaluar(clf, va), cfg))
    return corrida


def evaluar_final(corrida: Path, forzar: bool = False) -> str:
    registro = corrida / "evaluacion_final.json"
    if registro.is_file() and not forzar:
        raise SystemExit(f"Ya existe {registro}: el test ya se usó. Repetir exige --forzar (queda registrado).")
    cfg = yaml.safe_load((corrida / "config.yaml").read_text(encoding="utf-8"))
    ruta_modelo = corrida / "modelo_m0.joblib"
    verificar_archivo(ruta_modelo, ("sha256",))  # joblib ejecuta código al cargar: solo archivos propios con hash verificado
    clf = joblib.load(ruta_modelo)
    te = caracteristicas_con_cache(Path(cfg["datos"]), "test", corrida.parent / "cache_m0")
    met = _evaluar(clf, te)
    resumen = _guardar_reporte(corrida, "test", met, cfg)
    previo = json.loads(registro.read_text(encoding="utf-8")) if registro.is_file() else {"ejecuciones": []}
    previo["ejecuciones"].append({"forzado": bool(forzar), "exactitud": met["exactitud"], "f1_macro": met["f1_macro"]})
    guardar_json(registro, previo)
    return resumen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", type=Path)
    p.add_argument("--salida", type=Path)
    p.add_argument("--modelo", choices=sorted(GRILLAS), default="rf")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--max-train", type=int, default=30000)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--corrida", type=Path)
    p.add_argument("--final", action="store_true")
    p.add_argument("--forzar", action="store_true")
    a = p.parse_args(argv)
    if a.final:
        if not a.corrida:
            p.error("--final requiere --corrida")
        print(evaluar_final(a.corrida, a.forzar))
        return 0
    if not (a.datos and a.salida):
        p.error("--datos y --salida son obligatorios")
    entrenar(a.datos, a.salida, a.modelo, a.semilla, a.max_train, a.folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

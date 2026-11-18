"""
Entrena M2 (YOLO, Ultralytics) desde una config YAML y deja una corrida reproducible.

Uso (en Colab, desde la raíz del repo):
    python -m ml.m2_yolo.entrenar --config ml/experiments/configs/m2_yolo11n_640.yaml \
        --datos /content/sikafe_yolo/data.yaml --salida /content/drive/MyDrive/sikafe/corridas
    # si Colab se desconecta, reanudar la misma corrida:
    python -m ml.m2_yolo.entrenar --reanudar /content/drive/MyDrive/sikafe/corridas/<id_corrida>

Deja en <salida>/<id_corrida>/:
    config.yaml, entorno.json (versiones, GPU, commit, hash del dataset),
    train/ (salida de Ultralytics: curvas, results.csv, weights/best.pt, last.pt),
    m2_yolo.pt + m2_yolo.manifest.json (listo para la app; validado=false hasta evaluar --final),
    metricas_val_ultralytics.json
El conjunto de prueba NO se toca aquí (ver ml/m2_yolo/evaluar.py --final).
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ml.experiments.registro import capturar_entorno, fijar_semillas, guardar_json, id_corrida
from src.infrastructure.ml.manifiesto import sha256_archivo


def _metricas_ultralytics(res, nombres) -> dict:
    caja = res.box
    return {
        "map50": float(caja.map50),
        "map50_95": float(caja.map),
        "precision_media": float(caja.mp),
        "recall_medio": float(caja.mr),
        "map50_95_por_clase": {nombres[i]: float(v) for i, v in enumerate(caja.maps)},
        "velocidad_ms_por_imagen": dict(getattr(res, "speed", {}) or {}),
    }


def escribir_manifiesto_app(corrida: Path, cfg: dict, datos_yaml: Path, nombres: list, entorno: dict) -> Path:
    pesos = corrida / "m2_yolo.pt"
    resumen_datos = {}
    man_datos = Path(datos_yaml).parent / "manifiesto.json"
    if man_datos.is_file():
        resumen_datos = json.loads(man_datos.read_text(encoding="utf-8"))["resumen"]
    manifiesto = {
        "archivo": pesos.name,
        "sha256": sha256_archivo(pesos),
        "familia": "M2",
        "nombre": f"YOLO ({cfg['modelo_base']}, {cfg['imgsz']} px) – {resumen_datos.get('dataset', 'dataset')}",
        "framework": "ultralytics",
        "version_framework": entorno["paquetes"].get("ultralytics"),
        "clases": nombres,
        "imgsz": cfg["imgsz"],
        "iou_nms": cfg.get("iou_nms", 0.6),
        "datos_entrenamiento": (
            f"{resumen_datos.get('dataset', '?')} ({resumen_datos.get('licencia', 'licencia ?')}), "
            f"partición train, versión {str(resumen_datos.get('version_dataset_sha256', '?'))[:12]}"
        ),
        "validado_con_datos_reales": False,
        "dominio_validacion": None,
        "validado_en_dominio_objetivo": False,
        "corrida_id": corrida.name,
        "commit": entorno["git"]["commit"],
        "semilla": cfg["semilla"],
        "fecha_entrenamiento": datetime.now(timezone.utc).date().isoformat(),
        "url_descarga": None,
    }
    return guardar_json(corrida / "m2_yolo.manifest.json", manifiesto)


def entrenar(config: Path, datos: Path, salida: Path, semilla=None) -> Path:
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if semilla is not None:
        cfg["semilla"] = semilla
    corrida = Path(salida) / id_corrida(cfg["nombre"], cfg["semilla"])
    corrida.mkdir(parents=True, exist_ok=False)
    (corrida / "config.yaml").write_text(yaml.safe_dump({**cfg, "datos": str(datos)}, allow_unicode=True), encoding="utf-8")
    entorno = guardar_json(corrida / "entorno.json", capturar_entorno(Path(datos)))
    print(f"Corrida: {corrida}\nEntorno: {entorno.read_text(encoding='utf-8')}")

    fijar_semillas(cfg["semilla"])
    modelo = YOLO(cfg["modelo_base"])
    modelo.train(
        data=str(datos),
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        batch=cfg["batch"],
        seed=cfg["semilla"],
        deterministic=True,
        workers=cfg.get("workers", 2),
        project=str(corrida),
        name="train",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    return finalizar(corrida)


def reanudar(corrida: Path) -> Path:
    from ultralytics import YOLO

    ultimo = Path(corrida) / "train" / "weights" / "last.pt"
    if not ultimo.is_file():
        raise SystemExit(f"No hay {ultimo}: no se puede reanudar.")
    YOLO(str(ultimo)).train(resume=True)
    return finalizar(Path(corrida))


def finalizar(corrida: Path) -> Path:
    """Copia best.pt como m2_yolo.pt, escribe su manifiesto y valida en la partición val."""
    from ultralytics import YOLO

    cfg = yaml.safe_load((corrida / "config.yaml").read_text(encoding="utf-8"))
    entorno = json.loads((corrida / "entorno.json").read_text(encoding="utf-8"))
    mejor = corrida / "train" / "weights" / "best.pt"
    shutil.copy2(mejor, corrida / "m2_yolo.pt")
    modelo = YOLO(str(corrida / "m2_yolo.pt"))
    nombres = [modelo.names[i] for i in sorted(modelo.names)]
    escribir_manifiesto_app(corrida, cfg, Path(cfg["datos"]), nombres, entorno)
    res = modelo.val(data=cfg["datos"], split="val", imgsz=cfg["imgsz"], conf=0.001, iou=0.6, project=str(corrida), name="val", exist_ok=True)
    metricas = guardar_json(corrida / "metricas_val_ultralytics.json", _metricas_ultralytics(res, nombres))
    print("\n=== VALIDACIÓN (Ultralytics, partición val) ===")
    print(metricas.read_text(encoding="utf-8"))
    print(f"\nPesos para la app: {corrida / 'm2_yolo.pt'} (+ m2_yolo.manifest.json)")
    return corrida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--datos", type=Path, help="data.yaml generado por ml.datos.preparar_yolo")
    p.add_argument("--salida", type=Path, help="carpeta de corridas (idealmente en Google Drive)")
    p.add_argument("--semilla", type=int)
    p.add_argument("--reanudar", type=Path, help="carpeta de una corrida interrumpida")
    a = p.parse_args(argv)
    if a.reanudar:
        reanudar(a.reanudar)
    else:
        if not (a.config and a.datos and a.salida):
            p.error("--config, --datos y --salida son obligatorios (o use --reanudar)")
        entrenar(a.config, a.datos, a.salida, a.semilla)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
CUDA en M2: tiempo de inferencia de YOLO en CPU y en GPU, con el MISMO adaptador que usa la app.

Uso (Colab con T4 GPU):
    python -m ml.benchmark.yolo_cpu_gpu --corrida <carpeta_corrida> --n 200 --salida <carpeta>

Las imágenes se cargan en memoria antes de medir (no se mide la lectura de disco). Se descartan las
primeras --calentamiento inferencias de cada dispositivo. Se reporta mediana, media y percentil 90 por
imagen, y si CPU y GPU encuentran el mismo número de cerezas.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from ml.datos.lectura_yolo import imagenes_y_etiquetas
from ml.experiments.registro import capturar_entorno, guardar_json
from src.infrastructure.adapters.detectors.yolo_detector_adapter import YoloFruitDetectorAdapter

UMBRAL = 0.25


def gpu_disponible() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def cargar_imagenes(data_yaml: Path, split: str, n: int):
    imgs = []
    for ruta, _, _ in imagenes_y_etiquetas(data_yaml, split):
        with Image.open(ruta) as im:
            imgs.append(ImageOps.exif_transpose(im).convert("RGB"))
        if len(imgs) >= n:
            break
    return imgs


def medir(detector, imgs, calentamiento: int = 5) -> dict:
    for img in imgs[:calentamiento]:
        detector.detectar_frutos(img, UMBRAL)
    tiempos, conteos = [], []
    for img in imgs:
        t0 = time.perf_counter()
        dets = detector.detectar_frutos(img, UMBRAL)
        tiempos.append(time.perf_counter() - t0)
        conteos.append(len(dets))
    ms = sorted(1000 * t for t in tiempos)
    return {
        "n": len(imgs),
        "ms_mediana": statistics.median(ms),
        "ms_media": statistics.mean(ms),
        "ms_p90": ms[int(0.9 * (len(ms) - 1))],
        "conteos": conteos,
    }


def comparar(pesos: Path, imgs, calentamiento: int = 5) -> dict:
    res = {"umbral": UMBRAL, "imagenes": len(imgs), "calentamiento": calentamiento}
    res["cpu"] = medir(YoloFruitDetectorAdapter.desde_archivo(pesos, device="cpu"), imgs, calentamiento)
    if gpu_disponible():
        res["gpu"] = medir(YoloFruitDetectorAdapter.desde_archivo(pesos, device=0), imgs, calentamiento)
        c, g = np.asarray(res["cpu"]["conteos"]), np.asarray(res["gpu"]["conteos"])
        res["consistencia"] = {"pct_imagenes_mismo_conteo": float(100 * np.mean(c == g)), "dif_media_absoluta": float(np.mean(np.abs(c - g)))}
        res["aceleracion_mediana"] = res["cpu"]["ms_mediana"] / res["gpu"]["ms_mediana"]
    else:
        res["gpu"] = None
    return res


def tabla_md(res: dict, entorno: dict) -> str:
    filas = [f"| CPU | {res['cpu']['ms_mediana']:.1f} | {res['cpu']['ms_media']:.1f} | {res['cpu']['ms_p90']:.1f} | 1× |"]
    if res["gpu"]:
        g = res["gpu"]
        filas.append(
            f"| GPU ({entorno.get('gpu')}) | {g['ms_mediana']:.1f} | {g['ms_media']:.1f} | {g['ms_p90']:.1f} | {res['aceleracion_mediana']:.1f}× |"
        )
        cons = (
            f"\nMismo número de cerezas en CPU y GPU en el {res['consistencia']['pct_imagenes_mismo_conteo']:.1f} % de las imágenes "
            f"(diferencia media {res['consistencia']['dif_media_absoluta']:.2f} cerezas).\n"
        )
    else:
        cons = "\nGPU no disponible: solo se midió la CPU.\n"
    return f"""# CUDA en M2: inferencia de YOLO, CPU vs GPU

{res["imagenes"]} imágenes de validación (480×640), umbral {res["umbral"]}, a través del adaptador de la app (incluye pre y postproceso).
Se descartan las primeras {res["calentamiento"]} inferencias de cada dispositivo. CPU: {entorno.get("plataforma")}.

| Dispositivo | ms/imagen (mediana) | media | p90 | Aceleración |
|---|---|---|---|---|
{chr(10).join(filas)}
{cons}
Generado por `python -m ml.benchmark.yolo_cpu_gpu` (no editar a mano).
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrida", type=Path, required=True)
    p.add_argument("--datos", type=Path, help="data.yaml (por defecto, el de la config de la corrida)")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--calentamiento", type=int, default=5)
    p.add_argument("--salida", type=Path, required=True)
    a = p.parse_args(argv)
    datos = a.datos or Path(yaml.safe_load((a.corrida / "config.yaml").read_text(encoding="utf-8"))["datos"])
    imgs = cargar_imagenes(datos, a.split, a.n)
    res = comparar(a.corrida / "m2_yolo.pt", imgs, a.calentamiento)
    entorno = capturar_entorno(datos)
    res["entorno"] = entorno
    res["corrida"] = a.corrida.name
    a.salida.mkdir(parents=True, exist_ok=True)
    base = a.salida / f"yolo_cpu_gpu_{a.split}_{len(imgs)}"
    guardar_json(base.with_suffix(".json"), res)
    md = tabla_md(res, entorno)
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

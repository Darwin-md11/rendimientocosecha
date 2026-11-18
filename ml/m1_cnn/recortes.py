"""
Caché de recortes para M1: cada cereza anotada, recortada con margen y reescalada a LADO x LADO,
en un .npy de uint8 (N, LADO, LADO, 3) que se abre como memmap (no se carga entero en RAM).

Mismos recortes que M0 (misma función de recorte y margen, misma partición), así M0, M1 y la
clasificación de M2 en cajas emparejadas se comparan sobre las mismas cerezas.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ml.datos.lectura_yolo import cajas_gt, grupos_por_archivo, imagenes_y_etiquetas, version_dataset
from ml.m0_clasico.caracteristicas import limites_recorte

MARGEN = 0.10
LADO_MIN = 3


def _cajas_validas(ancho, alto, cajas, margen=MARGEN):
    for caja, c in cajas:
        x0, y0, x1, y1 = limites_recorte(ancho, alto, caja, margen)
        if min(x1 - x0, y1 - y0) >= LADO_MIN:
            yield (x0, y0, x1, y1), c


def construir_cache(data_yaml: Path, split: str, carpeta: Path, lado: int = 96) -> dict:
    """Devuelve {"X": ruta .npy, "meta": ruta .npz, "n": N}. Reutiliza la caché si ya existe."""
    base = Path(carpeta) / f"{version_dataset(data_yaml)[:12]}_{lado}px_{split}"
    ruta_x, ruta_meta = base.with_suffix(".npy"), base.with_suffix(".npz")
    if ruta_x.is_file() and ruta_meta.is_file():
        return {"X": ruta_x, "meta": ruta_meta, "n": len(np.load(ruta_meta)["y"]), "cache": True}
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    grupos = grupos_por_archivo(data_yaml)

    # Pasada 1: contar (solo cabeceras y etiquetas) para reservar el archivo
    lista = []
    for ruta, lbl, a_estado in imagenes_y_etiquetas(data_yaml, split):
        with Image.open(ruta) as im:
            ancho, alto = im.size
            if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # orientaciones que rotan 90°: se intercambian
                ancho, alto = alto, ancho
        cajas, clases = cajas_gt(lbl, ancho, alto, a_estado)
        validas = list(_cajas_validas(ancho, alto, zip(cajas, clases)))
        if validas:
            lista.append((ruta, validas))
    n = sum(len(v) for _, v in lista)

    # Pasada 2: recortar, reescalar y escribir (en un archivo temporal: una caché a medias nunca se reutiliza)
    tmp_x = base.with_suffix(".parcial.npy")
    X = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.uint8, shape=(n, lado, lado, 3))
    y, g, imgs = np.zeros(n, np.int64), [], []
    k = 0
    for ruta, validas in lista:
        with Image.open(ruta) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        for (x0, y0, x1, y1), c in validas:
            X[k] = np.asarray(rgb.crop((x0, y0, x1, y1)).resize((lado, lado), Image.BILINEAR), dtype=np.uint8)
            y[k] = c
            g.append(grupos.get(ruta.name, ruta.stem))
            imgs.append(ruta.name)
            k += 1
    X.flush()
    del X
    np.savez(ruta_meta, y=y, grupos=np.asarray(g), imagenes=np.asarray(imgs))
    os.replace(tmp_x, ruta_x)
    return {"X": ruta_x, "meta": ruta_meta, "n": n, "cache": False}


def cargar(cache: dict):
    """(X memmap de solo lectura, y, grupos)."""
    meta = np.load(cache["meta"])
    return np.load(cache["X"], mmap_mode="r"), meta["y"], meta["grupos"]


def pesos_por_clase(y: np.ndarray, n_clases: int = 4) -> np.ndarray:
    """Peso inverso a la frecuencia, normalizado para que la media ponderada sea 1 (clases ausentes: 0)."""
    conteo = np.bincount(y, minlength=n_clases).astype(float)
    pesos = np.where(conteo > 0, len(y) / (n_clases * np.maximum(conteo, 1)), 0.0)
    return pesos

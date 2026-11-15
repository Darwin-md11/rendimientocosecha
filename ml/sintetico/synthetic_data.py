"""
Generador de escenas sinteticas de ramas de cafeto para entrenar el clasificador.

Cada escena trae la imagen RGB, una mascara de clase por pixel (ver CLASES en
features.py) y un mapa de instancias (un id por fruto) para poder evaluar el
conteo. Incluye hojas que tapan frutos, racimos que se solapan, hojas secas y
ramas con colores parecidos a los frutos, y cambios de iluminacion y ruido.

IMPORTANTE: son imagenes sinteticas, no fotos reales. Sirven para arrancar el
sistema y para probar el pipeline, pero la precision en campo se debe medir y
mejorar reentrenando con fotos reales etiquetadas (ver training.py).
"""
import colorsys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ANCHO = 320
ALTO = 240

# Rangos (tono en grados, saturacion, brillo) por clase de fruto.
_RANGOS_FRUTO = {
    1: dict(h=(55, 100), s=(0.40, 0.80), v=(0.40, 0.85)),    # VERDE
    2: dict(h=(18, 55), s=(0.55, 0.95), v=(0.60, 0.98)),     # PINTON
    3: dict(h=(-12, 12), s=(0.65, 0.98), v=(0.45, 0.92)),    # MADURO
    4: dict(h=(-40, 10), s=(0.35, 0.80), v=(0.12, 0.40)),    # SOBREMADURO
}
_RANGO_HOJA = dict(h=(70, 140), s=(0.35, 0.90), v=(0.12, 0.55))
_RANGO_HOJA_SECA = dict(h=(15, 45), s=(0.50, 0.85), v=(0.25, 0.60))
_RANGO_RAMA = dict(h=(15, 40), s=(0.40, 0.70), v=(0.20, 0.40))


@dataclass
class Escena:
    rgb: np.ndarray          # (alto, ancho, 3) uint8
    etiquetas: np.ndarray    # (alto, ancho) int, valores 0..4
    instancias: np.ndarray   # (alto, ancho) int, 0 = sin fruto, >0 = id de fruto


def _color(rng: np.random.Generator, rango: Dict, brillo_extra: float = 1.0):
    h = rng.uniform(*rango["h"]) % 360.0
    s = float(np.clip(rng.uniform(*rango["s"]), 0.0, 1.0))
    v = float(np.clip(rng.uniform(*rango["v"]) * brillo_extra, 0.0, 1.0))
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _poligono_hoja(rng, cx, cy, largo, ancho):
    ang = rng.uniform(0, np.pi)
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    x = (largo / 2.0) * np.cos(t)
    y = (ancho / 2.0) * np.sin(t)
    xr = cx + x * np.cos(ang) - y * np.sin(ang)
    yr = cy + x * np.sin(ang) + y * np.cos(ang)
    return list(zip(xr.tolist(), yr.tolist()))


def _dibujar_fruto(d_img, d_lab, d_inst, cx, cy, rx, ry, estado, idx, rng):
    rango = _RANGOS_FRUTO[estado]
    h = rng.uniform(*rango["h"]) % 360.0
    s = rng.uniform(*rango["s"])
    v = rng.uniform(*rango["v"])
    caja = [cx - rx, cy - ry, cx + rx, cy + ry]
    d_lab.ellipse(caja, fill=estado)
    d_inst.ellipse(caja, fill=idx)

    pasos = 5  # elipses concentricas: mas claras hacia el centro, simulan volumen
    for k in range(pasos):
        f = k / (pasos - 1)
        escala = 1.0 - 0.55 * f
        dx, dy = -0.22 * rx * f, -0.22 * ry * f
        vv = float(np.clip(v * (0.75 + 0.35 * f), 0, 1))
        ss = float(np.clip(s * (1.0 - 0.15 * f), 0, 1))
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, ss, vv)
        d_img.ellipse(
            [cx + dx - rx * escala, cy + dy - ry * escala,
             cx + dx + rx * escala, cy + dy + ry * escala],
            fill=(int(r * 255), int(g * 255), int(b * 255)),
        )
    if rng.random() < 0.7:  # brillo especular
        hx, hy = cx - 0.35 * rx, cy - 0.35 * ry
        e = 0.16 * max(rx, ry)
        d_img.ellipse([hx - e, hy - e, hx + e, hy + e], fill=(235, 235, 225))


def generar_escena(rng: np.random.Generator, ancho: int = ANCHO, alto: int = ALTO) -> Escena:
    img = Image.new("RGB", (ancho, alto), _color(rng, _RANGO_HOJA, 0.7))
    lab = Image.new("I", (ancho, alto), 0)
    inst = Image.new("I", (ancho, alto), 0)
    d_img, d_lab, d_inst = ImageDraw.Draw(img), ImageDraw.Draw(lab), ImageDraw.Draw(inst)

    # Fondo: hojas (algunas secas) y ramas. No llevan etiqueta de fruto (quedan en 0).
    for _ in range(int(rng.integers(50, 90))):
        rango = _RANGO_HOJA_SECA if rng.random() < 0.04 else _RANGO_HOJA
        d_img.polygon(
            _poligono_hoja(rng, rng.uniform(0, ancho), rng.uniform(0, alto),
                           rng.uniform(25, 60), rng.uniform(10, 24)),
            fill=_color(rng, rango),
        )
    for _ in range(int(rng.integers(1, 4))):
        p0 = (rng.uniform(0, ancho), rng.uniform(0, alto))
        p1 = (rng.uniform(0, ancho), rng.uniform(0, alto))
        d_img.line([p0, p1], fill=_color(rng, _RANGO_RAMA), width=int(rng.integers(2, 5)))

    # Frutos en racimos mixtos.
    idx = 0
    for _ in range(int(rng.integers(1, 5))):
        ccx, ccy = rng.uniform(40, ancho - 40), rng.uniform(40, alto - 40)
        dominante = int(rng.integers(1, 5))
        for _ in range(int(rng.integers(1, 6))):
            estado = dominante if rng.random() < 0.6 else int(rng.integers(1, 5))
            r = rng.uniform(6, 16)
            idx += 1
            _dibujar_fruto(
                d_img, d_lab, d_inst,
                ccx + rng.normal(0, 18), ccy + rng.normal(0, 18),
                r * rng.uniform(0.85, 1.15), r * rng.uniform(0.85, 1.15),
                estado, idx, rng,
            )

    # Hojas delante: ocultan parte de los frutos (oclusion).
    for _ in range(int(rng.integers(4, 18))):
        poli = _poligono_hoja(rng, rng.uniform(0, ancho), rng.uniform(0, alto),
                              rng.uniform(25, 60), rng.uniform(10, 22))
        d_img.polygon(poli, fill=_color(rng, _RANGO_HOJA))
        d_lab.polygon(poli, fill=0)
        d_inst.polygon(poli, fill=0)

    # Iluminacion, ruido y desenfoque.
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * rng.uniform(0.55, 1.25) * rng.uniform(0.9, 1.1, size=3)
    arr = arr + rng.normal(0.0, rng.uniform(2.0, 8.0), size=arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    radio = rng.uniform(0.0, 1.0)
    if radio > 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radio))

    return Escena(
        rgb=np.asarray(img, dtype=np.uint8),
        etiquetas=np.asarray(lab, dtype=np.int64),
        instancias=np.asarray(inst, dtype=np.int64),
    )


def frutos_visibles(escena: Escena, area_min: int = 40) -> List[Dict]:
    """Verdad de terreno: frutos con al menos `area_min` pixeles visibles."""
    resultado = []
    for id_inst in np.unique(escena.instancias):
        if id_inst == 0:
            continue
        mascara = escena.instancias == id_inst
        area = int(mascara.sum())
        if area < area_min:
            continue
        ys, xs = np.nonzero(mascara)
        estado = int(np.bincount(escena.etiquetas[mascara]).argmax())
        resultado.append({
            "estado": estado,
            "area": area,
            "bbox": (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
        })
    return resultado

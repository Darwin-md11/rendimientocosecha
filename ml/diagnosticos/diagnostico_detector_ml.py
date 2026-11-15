"""
Diagnóstico del detector ML de SIKAFE (MLFruitDetectorAdapter).

Responde dos preguntas que las métricas del README no responden:
  1) ¿Qué tan frágil es ante cambios de luz, foco, ruido y compresión?
     (se perturban las MISMAS escenas sintéticas y se usa la misma métrica del repo)
  2) ¿Qué hace con imágenes que NO tienen cerezas de café?
     (lo correcto es 0 detecciones)

Uso (desde la raíz del repositorio, con las dependencias instaladas):
    python ml/diagnosticos/diagnostico_detector_ml.py
    python ml/diagnosticos/diagnostico_detector_ml.py --escenas 60      # más escenas = menos ruido estadístico

No modifica nada del repositorio. Las imágenes de prueba sin café salen de scikit-image
(incluidas en la librería) y, si hay internet, de dos fotos públicas del repo de Ultralytics.
"""
import argparse
import collections
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repositorio
from ml.sintetico.synthetic_data import frutos_visibles, generar_escena  # noqa: E402
from ml.sintetico.training import _SEMILLA_PRUEBA, _iou  # noqa: E402
from src.infrastructure.adapters.detectors.ml_detector_adapter import MLFruitDetectorAdapter  # noqa: E402
from src.infrastructure.composition import RUTA_MODELO_POR_DEFECTO  # noqa: E402
from src.infrastructure.ml.features import CLASES  # noqa: E402
from src.infrastructure.ml.modelo import ModeloMadurezPixel  # noqa: E402

UMBRAL, IOU_MIN = 0.40, 0.30  # los mismos valores que usa training.evaluar


# ------------------------------------------------------------------ perturbaciones
def _jpeg(img, calidad):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=calidad)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _canales(img, r, g, b):
    a = np.asarray(img).astype(np.float32) * np.array([r, g, b], dtype=np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _ruido(img, sigma):
    a = np.asarray(img).astype(np.float32)
    a += np.random.default_rng(1).normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _baja_resolucion(img, f):
    w, h = img.size
    return img.resize((int(w * f), int(h * f)), Image.BILINEAR).resize((w, h), Image.BILINEAR)


CONDICIONES = {
    "sin cambios (≈ README)": lambda i: i,
    "brillo x0.6 (sombra)": lambda i: ImageEnhance.Brightness(i).enhance(0.6),
    "brillo x1.4 (sol fuerte)": lambda i: ImageEnhance.Brightness(i).enhance(1.4),
    "balance cálido (R+15%, B-15%)": lambda i: _canales(i, 1.15, 1.0, 0.85),
    "balance frío (R-15%, B+15%)": lambda i: _canales(i, 0.85, 1.0, 1.15),
    "saturación x0.6 (nublado)": lambda i: ImageEnhance.Color(i).enhance(0.6),
    "desenfoque gauss σ=2": lambda i: i.filter(ImageFilter.GaussianBlur(2)),
    "ruido gauss σ=12": lambda i: _ruido(i, 12),
    "JPEG calidad 25": lambda i: _jpeg(i, 25),
    "resolución a la mitad": lambda i: _baja_resolucion(i, 0.5),
}


def evaluar_condicion(detector, transformar, n_escenas):
    rng = np.random.default_rng(_SEMILLA_PRUEBA)
    tp = fp = fn = estado_ok = total_det = total_gt = 0
    errores = []
    for _ in range(n_escenas):
        escena = generar_escena(rng)
        gt = frutos_visibles(escena)
        imagen = transformar(Image.fromarray(escena.rgb).convert("RGB"))
        dets = sorted(detector.detectar_frutos(imagen, UMBRAL), key=lambda d: -d.confianza)
        libres = set(range(len(gt)))
        for d in dets:
            caja = (d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max)
            mejor, mejor_iou = None, IOU_MIN
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
        errores.append(abs(len(dets) - len(gt)) / max(len(gt), 1))
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-9),
        "estado_ok": estado_ok / max(tp, 1),
        "det/gt": total_det / max(total_gt, 1),
        "err_conteo": float(np.mean(errores)),
    }


# --------------------------------------------------------- imágenes sin cerezas
def imagenes_sin_cafe():
    import skimage.data as sd

    imgs = {
        "skimage: taza de café": Image.fromarray(sd.coffee()),
        "skimage: astronauta": Image.fromarray(sd.astronaut()),
        "skimage: gato": Image.fromarray(sd.chelsea()),
        "skimage: cohete": Image.fromarray(sd.rocket()),
    }
    base = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/"
    for nombre in ("bus.jpg", "zidane.jpg"):
        try:
            datos = urllib.request.urlopen(base + nombre, timeout=15).read()
            imgs[f"ultralytics: {nombre}"] = Image.open(io.BytesIO(datos)).convert("RGB")
        except Exception:
            pass  # sin internet: se omiten
    return imgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--escenas", type=int, default=30, help="escenas sintéticas por condición (def. 30)")
    n = ap.parse_args().escenas

    detector = MLFruitDetectorAdapter(ModeloMadurezPixel.cargar(RUTA_MODELO_POR_DEFECTO))

    print(f"\n1) ROBUSTEZ sobre {n} escenas sintéticas perturbadas (umbral {UMBRAL}, IoU>={IOU_MIN})")
    print(f"{'condición':<32}{'P':>6}{'R':>6}{'F1':>6}{'estado':>8}{'det/gt':>8}{'err.cont':>10}")
    for nombre, f in CONDICIONES.items():
        r = evaluar_condicion(detector, f, n)
        print(f"{nombre:<32}{r['P']:>6.2f}{r['R']:>6.2f}{r['F1']:>6.2f}{r['estado_ok']:>8.2f}{r['det/gt']:>8.2f}{r['err_conteo']:>10.2f}")

    print("\n2) FALSOS POSITIVOS en imágenes SIN cerezas de café (lo correcto es 0)")
    for nombre, img in imagenes_sin_cafe().items():
        dets = detector.detectar_frutos(img, UMBRAL)
        por_estado = dict(collections.Counter(d.estado.value for d in dets))
        print(f"  {nombre:<26}{img.size[0]}x{img.size[1]:<6} -> {len(dets):3d} 'cerezas' {por_estado}")

    print("\n3) TIEMPO por imagen (mediana de 3, tu CPU)")
    base = Image.fromarray(np.random.default_rng(0).integers(0, 255, (3000, 4000, 3), dtype=np.uint8))
    for lado in [(640, 480), (1920, 1080), (4000, 3000)]:
        img = base.resize(lado)
        detector.detectar_frutos(img, UMBRAL)  # calentamiento
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            detector.detectar_frutos(img, UMBRAL)
            t.append(time.perf_counter() - t0)
        print(f"  {lado[0]}x{lado[1]:<6} {sorted(t)[1]:.2f} s")


if __name__ == "__main__":
    main()

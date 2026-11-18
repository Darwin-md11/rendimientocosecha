"""
Descriptor de M0, versión 2: especificación propia, implementada dos veces con el MISMO algoritmo:

  - descriptor_lote_numpy(imgs): referencia en CPU, vectorizada con numpy sobre un lote.
  - descriptor_secuencial(img, ...): un recorte, solo operaciones escalares y bucles. Es el código que
    se compila para la GPU (ml/m0_clasico/descriptor_cuda.py, un hilo de CUDA por recorte). En CPU se
    puede ejecutar en Python puro, y así se verifica contra la referencia sin necesidad de GPU.

Entrada: recorte RGB uint8 reescalado a 48 x 48 (preparar_recorte, en CPU con PIL).
Salida (442 valores, float):
  [0:48)    histograma HSV (H, S, V; 16 bins cada uno, en [0, 1])
  [48:96)   histograma Lab (L en [0, 100], a y b en [-128, 127]; 16 bins cada uno)
  [96:108)  medias de H, S, V, desviaciones de H, S, V, medias de L, a, b, desviaciones de L, a, b
  [108:118) LBP uniforme de 8 vecinos enteros (radio 1) sobre los píxeles interiores: 10 bins
  [118:442) HOG: gradiente central, 9 orientaciones sin signo (bins de 20°, sin interpolación),
            celdas de 12 x 12 px (4 x 4 celdas), bloques de 2 x 2 celdas (3 x 3 bloques), norma L2-Hys
Los histogramas se normalizan por el número de píxeles que cuentan (2304 o 2116 para LBP).

Precisión: la escala de grises y el LBP usan aritmética entera (idénticos en CPU y GPU). El resto es
float64; las funciones pow/atan2 y el orden de las sumas pueden diferir en el último bit entre CPU y
GPU, lo que en casos raros mueve un píxel de bin. Las pruebas y el script de comparación miden esa
diferencia y exigen que sea mínima.
"""

import math

import numpy as np
from PIL import Image

VERSION_DESCRIPTOR = 2
LADO = 48
NPIX = LADO * LADO  # 2304
NPIX_LBP = (LADO - 2) * (LADO - 2)  # 2116
BINS = 16
CELDA = 12
NCELDAS = LADO // CELDA  # 4
ORIENT = 9
NBLOQUES = NCELDAS - 1  # 3
DIMENSION = 48 + 48 + 12 + 10 + NBLOQUES * NBLOQUES * 4 * ORIENT  # 442
O_HSV, O_LAB, O_EST, O_LBP, O_HOG = 0, 48, 96, 108, 118

# sRGB (D65) -> XYZ (las mismas constantes que skimage.color) y blanco de referencia D65
_M = ((0.412453, 0.357580, 0.180423), (0.212671, 0.715160, 0.072169), (0.019334, 0.119193, 0.950227))
_BLANCO = (0.95047, 1.0, 1.08883)
# Vecinos del LBP en orden circular (dy, dx): derecha, arriba-derecha, arriba, ... (antihorario)
_VECINOS = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))
_EPS2 = 1e-10  # eps = 1e-5 de la norma L2-Hys


def tabla_lineal() -> np.ndarray:
    """sRGB uint8 -> intensidad lineal (256 valores). Se calcula en CPU y se comparte con la GPU."""
    c = np.arange(256, dtype=np.float64) / 255.0
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


TABLA_LINEAL = tabla_lineal()


def preparar_recorte(recorte_rgb: np.ndarray) -> np.ndarray:
    img = Image.fromarray(np.ascontiguousarray(recorte_rgb, dtype=np.uint8)).resize((LADO, LADO), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ====================================================================== referencia numpy (lote)
def _hist16(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """x: (N, P). Histograma de 16 bins por fila con la regla t=(x-lo)/(hi-lo); k=min(int(16 t), 15); fuera de [0,1] no cuenta."""
    n = x.shape[0]
    t = (x - lo) / (hi - lo)
    dentro = (t >= 0) & (t <= 1)
    k = np.minimum((t * BINS).astype(np.int64), BINS - 1)
    fila = np.broadcast_to(np.arange(n)[:, None], x.shape)
    return np.bincount((fila * BINS + k)[dentro], minlength=n * BINS).reshape(n, BINS) / NPIX


def _hsv_lab(imgs: np.ndarray):
    r, g, b = (imgs[..., i].astype(np.int64).reshape(len(imgs), -1) for i in range(3))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    v = mx / 255.0
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0.0)
    dd = np.where(d > 0, d, 1)
    h = np.select([d == 0, mx == r, mx == g], [0.0, (g - b) / dd, 2.0 + (b - r) / dd], 4.0 + (r - g) / dd)
    h = np.where(h < 0, h + 6.0, h) / 6.0
    h = np.where(h >= 1.0, h - 1.0, h)
    lr, lg, lb = TABLA_LINEAL[r], TABLA_LINEAL[g], TABLA_LINEAL[b]
    xyz = [(_M[i][0] * lr + _M[i][1] * lg + _M[i][2] * lb) / _BLANCO[i] for i in range(3)]
    f = [np.where(t > 0.008856, t ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0) for t in xyz]
    lab = (116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2]))
    return (h, s, v), lab


def descriptor_lote_numpy(imgs: np.ndarray) -> np.ndarray:
    """imgs: uint8 (N, 48, 48, 3). Devuelve float64 (N, 442)."""
    imgs = np.asarray(imgs, dtype=np.uint8)
    n = len(imgs)
    out = np.zeros((n, DIMENSION))
    if n == 0:
        return out
    (h, s, v), lab = _hsv_lab(imgs)
    for j, canal in enumerate((h, s, v)):
        out[:, O_HSV + 16 * j : O_HSV + 16 * (j + 1)] = _hist16(canal, 0.0, 1.0)
    for j, (canal, lo, hi) in enumerate(zip(lab, (0.0, -128.0, -128.0), (100.0, 127.0, 127.0))):
        out[:, O_LAB + 16 * j : O_LAB + 16 * (j + 1)] = _hist16(canal, lo, hi)
    for j, canal in enumerate((h, s, v)):
        out[:, O_EST + j] = canal.mean(1)
        out[:, O_EST + 3 + j] = canal.std(1)
    for j, canal in enumerate(lab):
        out[:, O_EST + 6 + j] = canal.mean(1)
        out[:, O_EST + 9 + j] = canal.std(1)

    i64 = imgs.astype(np.int64)
    gris = (299 * i64[..., 0] + 587 * i64[..., 1] + 114 * i64[..., 2] + 500) // 1000  # (N, 48, 48), entero exacto
    c = gris[:, 1:-1, 1:-1]
    bits = np.stack([gris[:, 1 + dy : LADO - 1 + dy, 1 + dx : LADO - 1 + dx] >= c for dy, dx in _VECINOS], axis=-1)
    unos = bits.sum(-1)
    cambios = (bits != np.roll(bits, -1, axis=-1)).sum(-1)
    codigo = np.where(cambios <= 2, unos, 9).reshape(n, -1)
    fila = np.broadcast_to(np.arange(n)[:, None], codigo.shape)
    out[:, O_LBP : O_LBP + 10] = np.bincount((fila * 10 + codigo).ravel(), minlength=n * 10).reshape(n, 10) / NPIX_LBP

    gf = gris / 255.0
    gx = np.zeros_like(gf)
    gy = np.zeros_like(gf)
    gx[:, :, 1:-1] = gf[:, :, 2:] - gf[:, :, :-2]
    gy[:, 1:-1, :] = gf[:, 2:, :] - gf[:, :-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) * (180.0 / math.pi)
    ang = np.where(ang < 0, ang + 180.0, ang)
    ang = np.where(ang >= 180.0, ang - 180.0, ang)
    bin_o = np.minimum((ang / 20.0).astype(np.int64), ORIENT - 1)
    yy, xx = np.meshgrid(np.arange(LADO), np.arange(LADO), indexing="ij")
    celda = (yy // CELDA) * NCELDAS + (xx // CELDA)
    idx = (np.arange(n)[:, None, None] * (NCELDAS * NCELDAS) + celda[None]) * ORIENT + bin_o
    celdas = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=n * NCELDAS * NCELDAS * ORIENT)
    celdas = celdas.reshape(n, NCELDAS, NCELDAS, ORIENT) / (CELDA * CELDA)
    k = 0
    for by in range(NBLOQUES):
        for bx in range(NBLOQUES):
            bloque = celdas[:, by : by + 2, bx : bx + 2, :].reshape(n, -1)
            bloque = bloque / np.sqrt((bloque**2).sum(1, keepdims=True) + _EPS2)
            bloque = np.minimum(bloque, 0.2)
            bloque = bloque / np.sqrt((bloque**2).sum(1, keepdims=True) + _EPS2)
            out[:, O_HOG + k : O_HOG + k + 36] = bloque
            k += 36
    return out


# ====================================================================== versión secuencial (= kernel CUDA)
def descriptor_secuencial(img, lut, gris, celdas, out):  # noqa: C901 (una sola función: requisito de Numba CUDA)
    """Un recorte. img uint8 (48,48,3); lut float64 (256); gris int (2304) y celdas float64 (144) son memoria
    de trabajo; out float64 (442) recibe el resultado. Solo escalares, bucles e índices: compatible con
    numba.cuda (sin numpy, listas ni funciones auxiliares)."""
    for j in range(442):
        out[j] = 0.0
    for j in range(144):
        celdas[j] = 0.0

    # ---- pasada 1: HSV/Lab por píxel -> histogramas y sumas; escala de grises entera
    sh = 0.0
    ss = 0.0
    sv = 0.0
    sl = 0.0
    sa = 0.0
    sb = 0.0
    for y in range(48):
        for x in range(48):
            ri = int(img[y, x, 0])
            gi = int(img[y, x, 1])
            bi = int(img[y, x, 2])
            gris[y * 48 + x] = (299 * ri + 587 * gi + 114 * bi + 500) // 1000
            # HSV en coma flotante: en la GPU, int(uint8) puede ser un entero SIN signo y g - b daría la vuelta
            r = float(ri)
            g = float(gi)
            b = float(bi)
            mx = r if r > g else g
            mx = mx if mx > b else b
            mn = r if r < g else g
            mn = mn if mn < b else b
            d = mx - mn
            vv = mx / 255.0
            s_ = d / mx if mx > 0.0 else 0.0
            if d == 0.0:
                hh = 0.0
            else:
                if mx == r:
                    hh = (g - b) / d
                elif mx == g:
                    hh = 2.0 + (b - r) / d
                else:
                    hh = 4.0 + (r - g) / d
                if hh < 0:
                    hh = hh + 6.0
                hh = hh / 6.0
                if hh >= 1.0:
                    hh = hh - 1.0
            lr = lut[ri]
            lg = lut[gi]
            lb = lut[bi]
            tx = (0.412453 * lr + 0.357580 * lg + 0.180423 * lb) / 0.95047
            ty = (0.212671 * lr + 0.715160 * lg + 0.072169 * lb) / 1.0
            tz = (0.019334 * lr + 0.119193 * lg + 0.950227 * lb) / 1.08883
            fx = tx ** (1.0 / 3.0) if tx > 0.008856 else 7.787 * tx + 16.0 / 116.0
            fy = ty ** (1.0 / 3.0) if ty > 0.008856 else 7.787 * ty + 16.0 / 116.0
            fz = tz ** (1.0 / 3.0) if tz > 0.008856 else 7.787 * tz + 16.0 / 116.0
            ll = 116.0 * fy - 16.0
            aa = 500.0 * (fx - fy)
            bb = 200.0 * (fy - fz)
            # histogramas: t = (x - lo) / (hi - lo); k = min(int(16 t), 15); fuera de [0, 1] no cuenta
            for canal in range(6):
                if canal == 0:
                    val = hh
                    lo = 0.0
                    hi = 1.0
                elif canal == 1:
                    val = s_
                    lo = 0.0
                    hi = 1.0
                elif canal == 2:
                    val = vv
                    lo = 0.0
                    hi = 1.0
                elif canal == 3:
                    val = ll
                    lo = 0.0
                    hi = 100.0
                elif canal == 4:
                    val = aa
                    lo = -128.0
                    hi = 127.0
                else:
                    val = bb
                    lo = -128.0
                    hi = 127.0
                t = (val - lo) / (hi - lo)
                if t >= 0.0 and t <= 1.0:
                    k = int(t * 16.0)
                    if k > 15:
                        k = 15
                    out[16 * canal + k] += 1.0
            sh += hh
            ss += s_
            sv += vv
            sl += ll
            sa += aa
            sb += bb
    for j in range(96):
        out[j] = out[j] / 2304.0
    mh = sh / 2304.0
    ms = ss / 2304.0
    mv = sv / 2304.0
    ml = sl / 2304.0
    ma = sa / 2304.0
    mb = sb / 2304.0

    # ---- pasada 2: desviaciones (se recalcula HSV/Lab para no guardar 6 valores por píxel)
    dh = 0.0
    ds = 0.0
    dv = 0.0
    dl = 0.0
    da = 0.0
    db = 0.0
    for y in range(48):
        for x in range(48):
            ri = int(img[y, x, 0])
            gi = int(img[y, x, 1])
            bi = int(img[y, x, 2])
            r = float(ri)
            g = float(gi)
            b = float(bi)
            mx = r if r > g else g
            mx = mx if mx > b else b
            mn = r if r < g else g
            mn = mn if mn < b else b
            d = mx - mn
            vv = mx / 255.0
            s_ = d / mx if mx > 0.0 else 0.0
            if d == 0.0:
                hh = 0.0
            else:
                if mx == r:
                    hh = (g - b) / d
                elif mx == g:
                    hh = 2.0 + (b - r) / d
                else:
                    hh = 4.0 + (r - g) / d
                if hh < 0:
                    hh = hh + 6.0
                hh = hh / 6.0
                if hh >= 1.0:
                    hh = hh - 1.0
            lr = lut[ri]
            lg = lut[gi]
            lb = lut[bi]
            tx = (0.412453 * lr + 0.357580 * lg + 0.180423 * lb) / 0.95047
            ty = (0.212671 * lr + 0.715160 * lg + 0.072169 * lb) / 1.0
            tz = (0.019334 * lr + 0.119193 * lg + 0.950227 * lb) / 1.08883
            fx = tx ** (1.0 / 3.0) if tx > 0.008856 else 7.787 * tx + 16.0 / 116.0
            fy = ty ** (1.0 / 3.0) if ty > 0.008856 else 7.787 * ty + 16.0 / 116.0
            fz = tz ** (1.0 / 3.0) if tz > 0.008856 else 7.787 * tz + 16.0 / 116.0
            ll = 116.0 * fy - 16.0
            aa = 500.0 * (fx - fy)
            bb = 200.0 * (fy - fz)
            dh += (hh - mh) * (hh - mh)
            ds += (s_ - ms) * (s_ - ms)
            dv += (vv - mv) * (vv - mv)
            dl += (ll - ml) * (ll - ml)
            da += (aa - ma) * (aa - ma)
            db += (bb - mb) * (bb - mb)
    out[96] = mh
    out[97] = ms
    out[98] = mv
    out[99] = math.sqrt(dh / 2304.0)
    out[100] = math.sqrt(ds / 2304.0)
    out[101] = math.sqrt(dv / 2304.0)
    out[102] = ml
    out[103] = ma
    out[104] = mb
    out[105] = math.sqrt(dl / 2304.0)
    out[106] = math.sqrt(da / 2304.0)
    out[107] = math.sqrt(db / 2304.0)

    # ---- LBP uniforme: 8 vecinos enteros en orden antihorario desde la derecha
    for y in range(1, 47):
        for x in range(1, 47):
            c = gris[y * 48 + x]
            b0 = 1 if gris[y * 48 + x + 1] >= c else 0
            b1 = 1 if gris[(y - 1) * 48 + x + 1] >= c else 0
            b2 = 1 if gris[(y - 1) * 48 + x] >= c else 0
            b3 = 1 if gris[(y - 1) * 48 + x - 1] >= c else 0
            b4 = 1 if gris[y * 48 + x - 1] >= c else 0
            b5 = 1 if gris[(y + 1) * 48 + x - 1] >= c else 0
            b6 = 1 if gris[(y + 1) * 48 + x] >= c else 0
            b7 = 1 if gris[(y + 1) * 48 + x + 1] >= c else 0
            unos = b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7
            cambios = (b0 ^ b1) + (b1 ^ b2) + (b2 ^ b3) + (b3 ^ b4) + (b4 ^ b5) + (b5 ^ b6) + (b6 ^ b7) + (b7 ^ b0)
            codigo = unos if cambios <= 2 else 9
            out[108 + codigo] += 1.0
    for j in range(10):
        out[108 + j] = out[108 + j] / 2116.0

    # ---- HOG
    for y in range(48):
        for x in range(48):
            gx = 0.0
            gy = 0.0
            if x >= 1 and x <= 46:
                gx = gris[y * 48 + x + 1] / 255.0 - gris[y * 48 + x - 1] / 255.0
            if y >= 1 and y <= 46:
                gy = gris[(y + 1) * 48 + x] / 255.0 - gris[(y - 1) * 48 + x] / 255.0
            mag = math.sqrt(gx * gx + gy * gy)
            ang = math.atan2(gy, gx) * (180.0 / math.pi)
            if ang < 0.0:
                ang = ang + 180.0
            if ang >= 180.0:
                ang = ang - 180.0
            bo = int(ang / 20.0)
            if bo > 8:
                bo = 8
            celdas[((y // 12) * 4 + (x // 12)) * 9 + bo] += mag
    for j in range(144):
        celdas[j] = celdas[j] / 144.0
    k = 118
    for by in range(3):
        for bx in range(3):
            # bloque = celdas (by,bx), (by,bx+1), (by+1,bx), (by+1,bx+1); 9 orientaciones cada una
            s2 = 0.0
            for q in range(4):
                cy = by + q // 2
                cx = bx + q % 2
                for o in range(9):
                    w = celdas[(cy * 4 + cx) * 9 + o]
                    s2 += w * w
            norma = math.sqrt(s2 + 1e-10)
            s2b = 0.0
            for q in range(4):
                cy = by + q // 2
                cx = bx + q % 2
                for o in range(9):
                    w = celdas[(cy * 4 + cx) * 9 + o] / norma
                    if w > 0.2:
                        w = 0.2
                    out[k + q * 9 + o] = w
                    s2b += w * w
            norma2 = math.sqrt(s2b + 1e-10)
            for j in range(36):
                out[k + j] = out[k + j] / norma2
            k += 36


def descriptor_secuencial_cpu(img: np.ndarray) -> np.ndarray:
    """Ejecuta la versión secuencial en Python puro (lenta; para verificar el código del kernel)."""
    out = np.zeros(DIMENSION)
    descriptor_secuencial(np.asarray(img, np.uint8), TABLA_LINEAL, np.zeros(NPIX, np.int64), np.zeros(144), out)
    return out

"""
CUDA en M0: compara el descriptor de color + textura en CPU y en GPU sobre recortes REALES del dataset,
verifica que ambos den el mismo resultado y mide la aceleración.

Uso (Colab con T4 GPU, desde la raíz del repo):
    python -m ml.benchmark.cuda_m0 --datos /content/sikafe_yolo/data.yaml --split val --n 20000 --salida <carpeta>

Mide (mediana de --repeticiones):
  - CPU por recorte: el descriptor numpy llamado recorte a recorte (como lo hace un bucle simple).
  - CPU por lote: el descriptor numpy vectorizado sobre lotes de 4096 recortes (la mejor versión en CPU).
  - GPU kernel: solo la ejecución del kernel CUDA (un hilo por recorte).
  - GPU total: kernel + copias CPU→GPU y GPU→CPU.
La lectura de imágenes, el recorte y el reescalado a 48x48 se hacen una vez en CPU y no entran en la comparación
(se reportan aparte).
"""

import argparse
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ml.datos.lectura_yolo import cajas_gt, imagenes_y_etiquetas
from ml.experiments.registro import capturar_entorno, guardar_json
from ml.m0_clasico.caracteristicas import preparar_recorte, recortar
from ml.m0_clasico.descriptor import descriptor_lote_numpy
from ml.m0_clasico.descriptor_cuda import HILOS_POR_BLOQUE, cuda_disponible, descriptor_lote_cuda

BLOQUES_DESCRIPTOR = {"hsv": (0, 48), "lab": (48, 96), "estadisticas": (96, 108), "lbp": (108, 118), "hog": (118, 442)}
# Tolerancias de equivalencia CPU/GPU: un píxel que cambia de bin mueve 1/2304 un histograma; el resto, redondeo.
TOL_HISTOGRAMA = 2.5 / 2116
TOL_CONTINUO = 1e-6
MAX_FRACCION_RECORTES_CON_CAMBIO = 0.01


def recolectar_recortes(data_yaml: Path, split: str, n: int, margen: float = 0.10):
    t0 = time.perf_counter()
    recortes = []
    for ruta, lbl, a_estado in imagenes_y_etiquetas(data_yaml, split):
        with Image.open(ruta) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        cajas, _ = cajas_gt(lbl, rgb.shape[1], rgb.shape[0], a_estado)
        for caja in cajas:
            rec = recortar(rgb, caja, margen)
            if min(rec.shape[:2]) >= 3:
                recortes.append(preparar_recorte(rec))
            if len(recortes) >= n:
                break
        if len(recortes) >= n:
            break
    return np.stack(recortes), time.perf_counter() - t0


def _mediana(f, repeticiones: int):
    tiempos, resultado = [], None
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        resultado = f()
        tiempos.append(time.perf_counter() - t0)
    return statistics.median(tiempos), resultado


def verificar_equivalencia(cpu: np.ndarray, gpu: np.ndarray) -> dict:
    dif = np.abs(cpu - gpu)
    por_bloque = {
        k: {"max_dif": float(dif[:, a:b].max()), "pct_identicos": float(100 * np.mean(cpu[:, a:b] == gpu[:, a:b]))}
        for k, (a, b) in BLOQUES_DESCRIPTOR.items()
    }
    hist = np.concatenate([dif[:, 0:96], dif[:, 108:118]], axis=1)
    cont = np.concatenate([dif[:, 96:108], dif[:, 118:442]], axis=1)
    frac = float(np.mean(hist.max(1) > 0))
    ok = hist.max() <= TOL_HISTOGRAMA and cont.max() <= TOL_CONTINUO and frac <= MAX_FRACCION_RECORTES_CON_CAMBIO
    return {
        "equivalente": bool(ok),
        "por_bloque": por_bloque,
        "fraccion_recortes_con_algun_cambio_de_bin": frac,
        "tolerancias": {"histograma": TOL_HISTOGRAMA, "continuo": TOL_CONTINUO, "max_fraccion_recortes": MAX_FRACCION_RECORTES_CON_CAMBIO},
    }


def comparar(imgs48: np.ndarray, n_por_recorte: int = 2000, repeticiones: int = 3, lote: int = 4096) -> dict:
    n = len(imgs48)
    res = {"n_recortes": n, "repeticiones": repeticiones, "lote_cpu": lote}
    sub = imgs48[: min(n_por_recorte, n)]
    t, _ = _mediana(lambda: [descriptor_lote_numpy(x[None]) for x in sub], repeticiones)
    res["cpu_por_recorte"] = {"ms_por_recorte": 1000 * t / len(sub), "medido_sobre": len(sub)}
    t, ref = _mediana(lambda: np.concatenate([descriptor_lote_numpy(imgs48[i : i + lote]) for i in range(0, n, lote)]), repeticiones)
    res["cpu_lote"] = {"s_total": t, "ms_por_recorte": 1000 * t / n}

    if not cuda_disponible():
        res["gpu"] = {"disponible": False, "motivo": "no hay GPU CUDA o falta numba"}
        return res
    t0 = time.perf_counter()
    with warnings.catch_warnings():  # la llamada de compilación usa pocos recortes: el aviso de baja ocupación no aplica
        warnings.simplefilter("ignore")
        descriptor_lote_cuda(imgs48[:256])  # primera llamada: compila el kernel
    compilacion = time.perf_counter() - t0
    corridas = []
    for _ in range(repeticiones):
        tiempos = {}
        gpu = descriptor_lote_cuda(imgs48, tiempos=tiempos)
        corridas.append(tiempos)
    kernel = statistics.median(c["kernel_s"] for c in corridas)
    total = statistics.median(c["total_s"] for c in corridas)
    res["gpu"] = {
        "disponible": True,
        "compilacion_s": compilacion,
        "hilos_por_bloque": HILOS_POR_BLOQUE,
        "bloques": corridas[0]["bloques"],
        "kernel_s": kernel,
        "total_s": total,
        "copia_a_gpu_s": statistics.median(c["copia_a_gpu_s"] for c in corridas),
        "copia_a_cpu_s": statistics.median(c["copia_a_cpu_s"] for c in corridas),
        "ms_por_recorte_kernel": 1000 * kernel / n,
        "ms_por_recorte_total": 1000 * total / n,
    }
    res["equivalencia"] = verificar_equivalencia(ref, gpu)
    cpu_rec = res["cpu_por_recorte"]["ms_por_recorte"] * n / 1000
    res["aceleracion"] = {
        "cpu_por_recorte_vs_gpu_kernel": cpu_rec / kernel,
        "cpu_lote_vs_gpu_kernel": res["cpu_lote"]["s_total"] / kernel,
        "cpu_lote_vs_gpu_total": res["cpu_lote"]["s_total"] / total,
    }
    return res


def tabla_md(res: dict, entorno: dict) -> str:
    n = res["n_recortes"]
    filas = [
        f"| CPU, recorte a recorte (numpy) | {res['cpu_por_recorte']['ms_por_recorte']:.4f} | {res['cpu_por_recorte']['ms_por_recorte'] * n / 1000:.2f} (estimado) | 1× |",
        f"| CPU, por lotes de {res['lote_cpu']} (numpy vectorizado) | {res['cpu_lote']['ms_por_recorte']:.4f} | {res['cpu_lote']['s_total']:.2f} | "
        f"{res['cpu_por_recorte']['ms_por_recorte'] / res['cpu_lote']['ms_por_recorte']:.1f}× |",
    ]
    g = res["gpu"]
    if g["disponible"]:
        base = res["cpu_por_recorte"]["ms_por_recorte"]
        filas += [
            f"| GPU, solo kernel CUDA | {g['ms_por_recorte_kernel']:.4f} | {g['kernel_s']:.3f} | {base / g['ms_por_recorte_kernel']:.1f}× |",
            f"| GPU, kernel + copias CPU↔GPU | {g['ms_por_recorte_total']:.4f} | {g['total_s']:.3f} | {base / g['ms_por_recorte_total']:.1f}× |",
        ]
        e = res["equivalencia"]
        eq = "\n".join(f"| {k} | {v['max_dif']:.2e} | {v['pct_identicos']:.2f} % |" for k, v in e["por_bloque"].items())
        extra = f"""
**Equivalencia CPU/GPU: {"OK" if e["equivalente"] else "FALLA"}** (recortes con algún píxel en otro bin: {100 * e["fraccion_recortes_con_algun_cambio_de_bin"]:.3f} %)

| Bloque | Máx. diferencia | Componentes idénticos |
|---|---|---|
{eq}

Kernel: un hilo por recorte, {g["hilos_por_bloque"]} hilos por bloque, {g["bloques"]} bloques; compilación (una vez): {g["compilacion_s"]:.1f} s.
Aceleración frente a CPU por lotes: {res["aceleracion"]["cpu_lote_vs_gpu_kernel"]:.1f}× (solo kernel), {res["aceleracion"]["cpu_lote_vs_gpu_total"]:.1f}× (con copias).
"""
    else:
        extra = f"\nGPU no disponible: {g['motivo']}.\n"
    return f"""# CUDA en M0: descriptor de color y textura, CPU vs GPU

{n} recortes reales de cerezas (48×48 px), mediana de {res["repeticiones"]} repeticiones.
Hardware: GPU {entorno.get("gpu") or "ninguna"} · {entorno.get("plataforma")} · numba {entorno.get("paquetes", {}).get("numba")}

| Implementación | ms por recorte | Tiempo total (s) | Aceleración vs CPU recorte a recorte |
|---|---|---|---|
{chr(10).join(filas)}
{extra}
Generado por `python -m ml.benchmark.cuda_m0` (no editar a mano).
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--repeticiones", type=int, default=3)
    p.add_argument("--salida", type=Path, required=True)
    a = p.parse_args(argv)
    imgs48, t_prep = recolectar_recortes(a.datos, a.split, a.n)
    print(f"{len(imgs48)} recortes preparados en {t_prep:.1f} s (lectura + recorte + reescalado, CPU)")
    res = comparar(imgs48, repeticiones=a.repeticiones)
    res["preparacion_s"] = t_prep
    entorno = capturar_entorno(a.datos)
    res["entorno"] = entorno
    a.salida.mkdir(parents=True, exist_ok=True)
    base = a.salida / f"cuda_m0_{a.split}_{len(imgs48)}"
    guardar_json(base.with_suffix(".json"), res)
    md = tabla_md(res, entorno)
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    if res["gpu"]["disponible"] and not res["equivalencia"]["equivalente"]:
        print("ERROR: la GPU no reproduce el descriptor de la CPU dentro de la tolerancia. Pegue esta salida en el chat.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Descriptor de M0 en GPU con CUDA (Numba). Paralelismo: UN HILO DE CUDA POR RECORTE.

El cuerpo del kernel es descriptor_secuencial (ml/m0_clasico/descriptor.py), compilado con
numba.cuda.jit(device=True): la GPU ejecuta exactamente el algoritmo que las pruebas verifican en CPU.
Cada hilo usa memoria local para su escala de grises (2304 enteros) y sus 16 celdas HOG (144 valores),
y escribe su fila de 442 valores en memoria global; ningún hilo escribe en la fila de otro, así que no
hacen falta operaciones atómicas ni sincronización.

Requiere una GPU NVIDIA y numba (vienen en Google Colab con "T4 GPU").
"""

import time

import numpy as np

from .descriptor import DIMENSION, TABLA_LINEAL, descriptor_secuencial

HILOS_POR_BLOQUE = 128
_KERNEL = None


class CudaNoDisponible(RuntimeError):
    pass


def cuda_disponible() -> bool:
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except ImportError:
        return False


def _compilar():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    try:
        from numba import cuda, float64, int32
    except ImportError as e:
        raise CudaNoDisponible("Falta la librería numba (pip install numba).") from e
    if not cuda.is_available():
        raise CudaNoDisponible("No hay GPU CUDA disponible (en Colab: Entorno de ejecución → Cambiar tipo → T4 GPU).")

    descriptor_dispositivo = cuda.jit(device=True)(descriptor_secuencial)

    @cuda.jit
    def kernel(imgs, lut, salidas):
        i = cuda.grid(1)  # índice global del hilo = índice del recorte
        if i < imgs.shape[0]:
            gris = cuda.local.array(2304, int32)
            celdas = cuda.local.array(144, float64)
            descriptor_dispositivo(imgs[i], lut, gris, celdas, salidas[i])

    _KERNEL = kernel
    return kernel


def descriptor_lote_cuda(imgs: np.ndarray, hilos_por_bloque: int = HILOS_POR_BLOQUE, tiempos: dict = None) -> np.ndarray:
    """imgs uint8 (N, 48, 48, 3) -> float64 (N, 442) calculado en la GPU.
    Si se pasa `tiempos`, se rellena con la duración de cada etapa (copia a GPU, kernel, copia a CPU)."""
    kernel = _compilar()  # primero: si falta numba o la GPU, lanza CudaNoDisponible con un mensaje claro
    from numba import cuda

    imgs = np.ascontiguousarray(imgs, dtype=np.uint8)
    n = len(imgs)
    if n == 0:
        return np.zeros((0, DIMENSION))
    t0 = time.perf_counter()
    d_imgs = cuda.to_device(imgs)
    d_lut = cuda.to_device(TABLA_LINEAL)
    d_out = cuda.device_array((n, DIMENSION), dtype=np.float64)
    cuda.synchronize()
    t1 = time.perf_counter()
    bloques = (n + hilos_por_bloque - 1) // hilos_por_bloque
    kernel[bloques, hilos_por_bloque](d_imgs, d_lut, d_out)
    cuda.synchronize()
    t2 = time.perf_counter()
    out = d_out.copy_to_host()
    t3 = time.perf_counter()
    if tiempos is not None:
        tiempos.update(
            {
                "copia_a_gpu_s": t1 - t0,
                "kernel_s": t2 - t1,
                "copia_a_cpu_s": t3 - t2,
                "total_s": t3 - t0,
                "bloques": bloques,
                "hilos_por_bloque": hilos_por_bloque,
            }
        )
    return out


def simular_malla_en_cpu(imgs: np.ndarray) -> np.ndarray:
    """Reproduce en CPU, de forma secuencial, lo que hace la malla de hilos del kernel: el "hilo" i calcula
    el recorte i con su propia memoria de trabajo y escribe solo en salidas[i]. Sirve para verificar el
    reparto de trabajo sin GPU (no mide rendimiento)."""
    imgs = np.ascontiguousarray(imgs, dtype=np.uint8)
    salidas = np.zeros((len(imgs), DIMENSION))
    for i in range(len(imgs)):  # en la GPU: i = cuda.grid(1), todos a la vez
        gris = np.zeros(2304, np.int32)
        celdas = np.zeros(144, np.float64)
        descriptor_secuencial(imgs[i], TABLA_LINEAL, gris, celdas, salidas[i])
    return salidas

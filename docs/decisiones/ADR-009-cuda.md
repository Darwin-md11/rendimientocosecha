# ADR-009 — CUDA: descriptor de M0 en GPU y comparación CPU/GPU de YOLO

- **Estado:** aceptada (2026-11-17), opción elegida por el equipo según el requisito del curso (usar CUDA o MPI).
- **Decisión:**
  1. **Kernel CUDA propio** para el descriptor de M0 (color HSV/Lab, LBP, HOG; 442 dimensiones), escrito con Numba. Paralelismo de datos: **un hilo por recorte**; cada hilo usa memoria local para su escala de grises y sus celdas HOG y escribe solo su fila del resultado (sin atómicas ni sincronización).
  2. **Comparación CPU vs GPU de la inferencia de YOLO** con el mismo adaptador que usa la app.
- **Por qué un descriptor v2:** el v1 usaba scikit-image, cuyo LBP y HOG están compilados (Cython) y no se pueden replicar con garantía en CUDA. El v2 tiene especificación propia (docstring de `ml/m0_clasico/descriptor.py`) y se implementa dos veces con el mismo algoritmo: numpy vectorizado (referencia CPU) y una función secuencial que es el cuerpo del kernel. M0 no se había ejecutado con datos reales, así que el cambio no invalida resultados.
- **Verificación:**
  - En CPU (pruebas): la función del kernel reproduce la referencia numpy (histogramas idénticos, resto con diferencias ≤ 1e-9 por el orden de las sumas). El reparto "un hilo por recorte" se simula en CPU.
  - En GPU (Colab, `ml/benchmark/cuda_m0`): se compara con la CPU sobre recortes reales. Tolerancias: ≤ 2,5/2116 por bin de histograma, ≤ 1e-6 en el resto y < 1 % de recortes con algún píxel en otro bin (pow/atan2 de la GPU pueden diferir en el último bit). El script termina con error si no se cumple.
- **Medición:** mediana de repeticiones; se reporta CPU recorte a recorte, CPU vectorizada por lotes (la mejor CPU), GPU solo kernel y GPU con copias. La preparación de recortes (lectura, recorte, reescalado) es igual para todos y se reporta aparte. Primera llamada: compilación, excluida.
- **Limitaciones:** el kernel usa float64 (en la T4 rinde menos que float32) para priorizar la equivalencia con la CPU; la alternativa de un bloque por recorte con memoria compartida y atómicas queda como trabajo futuro.

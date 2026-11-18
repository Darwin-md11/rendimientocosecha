# Bitácora del proyecto SIKAFE

> Documento vivo: se actualiza en cada paso. Sirve para retomar el trabajo en otra conversación o por otro integrante.
> Metodología: CRISP-DM (ADR-001). Entrega final: **viernes 2026-11-20**.

## Estado actual

| Fase | Estado | Rama |
|---|---|---|
| 0 — Línea base, honestidad y orden del repo | ✔ código y pruebas listos; **pendiente:** Release del modelo y *push* | `fase-0-honestidad` |
| 1 — Módulo de rendimiento (versión mínima) | ✔ factores con fuente en `configs/agronomia.yaml`, `TODO(fuente)` visible; **pendiente:** fuentes reales y unidad de muestreo | `fase-2-3-croppie-yolo` |
| 2 — Datos (Croppie) | ✔ script de preparación listo; **pendiente:** ejecutarlo en Colab | `fase-2-3-croppie-yolo` |
| 3 — M2 YOLO | ✔ entrenado (corrida `m2_yolo11n_640_s0_20261116-1754`); **pendiente:** evaluación propia en val, test final | `fase-2-3-croppie-yolo` |
| 3 — M0 | ✔ script listo (RF y SVM, CV por sesión); **pendiente:** ejecutar en Colab (CPU) | `fase-2-3-croppie-yolo` |
| 3 — M1 | ✔ entrenado (MobileNetV2, corrida `m1_mobilenetv2_96_s0_20261118-0456`); **pendiente:** test final | `fase-2-3-croppie-yolo` |
| CUDA (requisito del curso) | ✔ ejecutado en T4: equivalencia OK, 43,6× (kernel vs mejor CPU); YOLO 7,8× | `fase-2-3-croppie-yolo` |
| 4 — Evaluación | ✔ métricas comunes listas; tablas de M2 se generan al evaluar | `fase-2-3-croppie-yolo` |
| 5–6 — Prototipo final y cierre | — | — |

## Plan ajustado a 5 días (propuesto 2026-11-15, falta confirmación del equipo)

| Día | Software | Equipo |
|---|---|---|
| Dom 15 – Lun 16 | Fase 0; Fase 1 mínima (cadena de rendimiento con `TODO(fuente)`) | Elegir y descargar dataset, verificar licencia; confirmar Colab con GPU |
| Lun 16 – Mar 17 | Scripts/notebook Colab de M2 (YOLO), M0 y M1 sobre recortes; partición por grupo | Entrenar en Colab y pegar logs |
| Mié 18 | Adaptador YOLO + contrato; métricas e informe generado por script | Fotos propias con conteo manual (si es posible) |
| Jue 19 | Integración UI; tablas y figuras para 8.2–8.4 | Redacción del informe |
| Vie 20 | Entrega | Entrega |

**Fuera de alcance por el plazo (se declara como limitación):** experimentos (b) y (d), ≥3 semillas (salvo M0), estimación multi-foto con bootstrap.

## Decisiones tomadas

| Fecha | Decisión | Dónde |
|---|---|---|
| 2026-11-15 | Rendimiento en kg de café **cereza**; pergamino solo si hay factor con fuente; factores sin fuente marcados `TODO(fuente)` y avisados en el reporte | Fase 1 |
| 2026-11-15 | Detector sintético: solo demo/regresión, etiquetado "SINTÉTICO – NO VALIDADO" | ADR-003 |
| 2026-11-15 | Pesos fuera de git, Release + manifiesto SHA-256 + versión de scikit-learn; fallo claro si no coincide (sin reentreno automático) | ADR-004 |
| 2026-11-15 | Se mantiene Gradio | ADR-006 |
| 2026-11-15 | `src/` (app) separado de `ml/` (laboratorio) | ADR-005 |
| 2026-11-15 | Dataset para M2: Croppie (CC BY-SA 4.0); partición 70/15/15 por foto original; low_visibility excluida y enmascarada | docs/datasets.md |
| 2026-11-15 | Métricas comunes propias (COCO 101 puntos) como cifra principal; mAP de Ultralytics como referencia | ADR-007 (propuesta) |
| 2026-11-15 | Procedencia en tres niveles: no validado / validado en otro dominio / validado en la región | ADR-008 |

## Pendientes / preguntas abiertas

- [ ] **Unidad de muestreo** (qué cubre una foto: rama, lado de planta, planta): sin definir. Bloquea la cadena de rendimiento completa de la Fase 1.
- [ ] **Estimación con varias fotos**: sin definir (requiere visto bueno del docente). Mientras tanto, una imagen a la vez.
- [ ] Documento fuente de los factores agronómicos (Cenicafé/FNC): peso por fruto, frutos por rama, ramas por planta, densidad, factor cereza→pergamino.
- [ ] **Confirmar mapeo amarilla → PINTÓN** (Croppie `yellow_cherry`).
- [ ] Ejecutar `ml/notebooks/m2_yolo_colab.ipynb` en Colab y pegar las salidas 📋.
- [ ] Fotos propias con conteo manual (15–20) para la evaluación en la región y el R² frente a conteo manual.
- [ ] PDF del artículo de revisión sistemática (no se recibió).
- [ ] Inconsistencias del documento base: §7.2 dice "cuatro etapas" pero la tabla tiene seis fases; ref. [7] con "⚠ COMPLETAR"; cifra de pérdidas >30 % sin fuente.

## Bitácora

### 2026-11-15 — Fase 0 (rama `fase-0-honestidad`)

- 0.1 Repo clonado en `b3456b6`: 17 pruebas pasan. Diagnóstico reproducido → `docs/resultados/diagnostico-sintetico.md` (19–32 falsos positivos en fotos sin café).
- 0.2 `ModeloInfo` + `info()`/`es_simulado` en el puerto; procedencia y advertencias en la respuesta, la imagen anotada (franja), la UI (banner permanente, sin "✅") y la CLI. Suite de contrato reutilizable.
- 0.3 Manifiesto con SHA-256 verificado antes de `joblib.load`; sin reentreno automático ni cambio silencioso de detector.
- 0.4 `requirements.txt` / `-dev` / `-ml`; ruff; CI de GitHub Actions (Python 3.12).
- 0.5 Entrenamiento y generador sintético movidos a `ml/sintetico/`; la app ya no los importa.
- 0.6 ADR-001 a ADR-006, README actualizado.
- **Bloqueo:** sin permiso de *push* desde la sesión de desarrollo → los cambios se entregan como parches (`git am`). Antes de fusionar: publicar la Release `modelo-sintetico-v1` (ADR-004).

### 2026-11-15 (noche) — Fases 2–4 para M2 (rama `fase-2-3-croppie-yolo`, sale de `fase-0-honestidad`)

- Dataset elegido: Croppie (ficha: CC BY-SA 4.0, 633 fotos originales, ~61 050 cajas, formato YOLO). `configs/clases.yaml` con el mapeo.
- `ml/datos/preparar_yolo.py`: mapeo, enmascarado de excluidas, agrupamiento de recortes por foto original con chequeo de plausibilidad, partición 70/15/15 por grupo, prueba de fuga, data.yaml, manifiesto con hashes, EDA (tabla y gráfico).
- `src/evaluation/metricas.py`: métricas comunes probadas a mano.
- `YoloFruitDetectorAdapter` + composición (`SIKAFE_DETECTOR=yolo`, por defecto si existe su manifiesto).
- `ml/m2_yolo/entrenar.py` y `evaluar.py` (test una sola vez con `--final`), configs 640/960, `ml/notebooks/m2_yolo_colab.ipynb`.
- **No ejecutado con Ultralytics real** (sin GPU/PyTorch en el entorno de desarrollo): probado con un modelo simulado. Las cifras reales saldrán del notebook.

### 2026-11-16/17 — Datos preparados y M2 entrenado (Colab, T4)

- Inspección de Croppie: 8 333 recortes 480×640 de 1 262 fotos (marca de tiempo en el nombre), 17 días (2023-06-22 a 2023-07-20).
  **La partición original del dataset tiene fuga:** 616 de 1 262 fotos tienen recortes en su train y su val → no se usa.
- Partición propia por **sesión de captura** (umbral automático 15 min: 60 sesiones, la mayor con 7,8 % de las imágenes):
  train 5 960 / val 1 115 / test 1 258 imágenes (40/12/8 sesiones); prueba de fuga OK; dataset `9dbcf58b…`.
  La mezcla de clases cambia entre particiones (test: 50 % VERDE, 2,8 % SOBREMADURO; train: 31 % y 4,8 %) → reportar por clase.
- M2 YOLO11n 640 px, 60 épocas (1,6 h en T4; la parada temprana no se activó), Ultralytics 8.4.215, torch 2.14.1+cu130.
  Validación según Ultralytics (cifras pegadas por el equipo; fuente: `metricas_val_ultralytics.json` de la corrida):
  mAP@0,5 0,673 · mAP@0,5:0,95 0,450 · P 0,702 · R 0,658; por clase mAP@0,5: MADURO 0,894, VERDE 0,826, SOBREMADURO 0,508, PINTÓN 0,466.
  Optimista (el mejor modelo se eligió en val). Pendiente: métricas propias (`evaluar --split val`) y test final.
- M0: `ml/m0_clasico/` (descriptor HSV/Lab/LBP/HOG de 442 dimensiones, RF y SVM, CV por sesión, test único). Pendiente ejecutarlo.
- Métricas: clasificación de madurez en cajas emparejadas para M2 (comparable con M0/M1 sobre recortes).

### 2026-11-17 — CUDA (opciones 1 + 2, ADR-009)

- Descriptor de M0 v2 (especificación propia) en CPU (numpy) y GPU (kernel CUDA con Numba, un hilo por recorte); verificado en CPU.
- `ml/benchmark/cuda_m0.py` (equivalencia + aceleración sobre recortes reales) y `ml/benchmark/yolo_cpu_gpu.py`.
- **No ejecutado en GPU** (el entorno de desarrollo no tiene GPU ni Numba): pendiente correr las celdas CUDA en Colab.
- M2, procedencia confirmada por el equipo: commit 803c327, dataset 9dbcf58b…, pesos SHA-256 c5b8838a…, T4.

### 2026-11-17 (noche) — M1

- `ml/m1_cnn/`: caché de recortes 96×96 en disco (las MISMAS cerezas y el mismo orden que M0, verificado), MobileNetV2 preentrenada
  en ImageNet: 2 épocas de cabeza (red congelada) + 10 de ajuste completo (AdamW, coseno), pérdida con pesos por clase, aumento en
  GPU sin tocar tono ni saturación, precisión mixta, mejor época por F1 macro en val, test único con `--final`.
- Configs: `m1_mobilenetv2_96` (principal), `m1_efficientnet_b0_96` y `m1_mobilenetv2_96_sin_preentrenar` (experimento (d), opcionales).
- **No ejecutado** (sin PyTorch en el entorno de desarrollo): probado lo que no depende de PyTorch; en Colab, primero `--rapido`.
- Reporte de clasificación común para M0 y M1 (`ml/experiments/reporte_clasificacion.py`).
- Alcance: M0 y M1 clasifican recortes de cajas reales (no detectan); se comparan con M2 por la clasificación en cajas
  emparejadas. El pipeline de dos etapas (localizador + clasificador) para usar M0/M1 en la app queda como trabajo futuro.

### 2026-11-17/18 — Resultados de CUDA y M1 (Colab, T4; salidas pegadas por el equipo)

- **CUDA, descriptor M0** (20 000 recortes reales de val, mediana de 3): primera corrida **falló la equivalencia** (HSV y
  estadísticas): en la GPU `int(uint8)` es sin signo y las restas del tono daban la vuelta. Corregido (commit 90b5111) con prueba que lo reproduce.
  Segunda corrida: **equivalencia OK** (HSV, Lab, LBP 100 % idénticos; estadísticas ≤ 1,4e-12; HOG ≤ 3,3e-16; 0 recortes con cambio de bin).
  Tiempos: CPU recorte a recorte 1,888 ms; CPU por lotes 1,407 ms (28,1 s); GPU kernel 0,032 ms (0,645 s); GPU con copias 0,035 ms (0,707 s).
  Aceleración: 58,6× (vs recorte a recorte), **43,6× (vs CPU por lotes)**, 39,8× con copias. La preparación de recortes en CPU
  (5,9 s) domina de punta a punta (≈ 34 s → ≈ 6,6 s, ≈ 5×): ley de Amdahl. Compilación del kernel: 4,8 s (una vez).
- **CUDA, YOLO** (200 imágenes de val, umbral 0,25, adaptador de la app): CPU 95,5 ms/imagen (mediana), GPU 12,2 ms → **7,8×**;
  mismo conteo en el 100 % de las imágenes. CPU de Colab: 2 vCPU.
- **M1** MobileNetV2 96 px (corrida `m1_mobilenetv2_96_s0_20261118-0456`, 9,6 min de entrenamiento): val exactitud 0,871, F1 macro 0,768;
  F1 por clase: MADURO 0,930, VERDE 0,902, SOBREMADURO 0,682, PINTÓN 0,559 (precisión 0,459 / recall 0,715: los pesos por clase
  favorecen el recall de las minoritarias). Mejor época = la última (mejora marginal). Optimista (época elegida en val).
- Avisos de M1 (float(loss) y orden del planificador con AMP) corregidos después de esta corrida; no cambian los resultados de forma relevante.
- Módulo de rendimiento mínimo: `configs/agronomia.yaml` (única fuente de valores por defecto), supuestos con fuente en la respuesta,
  UI y CLI; pergamino solo con factor con fuente; `docs/modelo-rendimiento.md`.

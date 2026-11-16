# ADR-007 — Una sola implementación de métricas para M0, M1 y M2

- **Estado:** propuesta (2026-11-15); consultar con el docente si exige la mAP de Ultralytics.
- **Contexto:** M0/M1 clasifican recortes y M2 detecta; si cada uno usa su propia herramienta de métricas, las cifras no son comparables.
- **Decisión:** `src/evaluation/metricas.py` (numpy, probado contra ejemplos calculados a mano) es la referencia para el informe: emparejamiento tipo COCO, AP con interpolación de 101 puntos, mAP@0,5 y mAP@0,5:0,95, P/R/F1 a un umbral fijo (0,25 por defecto, en la config), matriz de confusión con "fondo", métricas de clasificación sobre recortes y de conteo (MAE, RMSE, sesgo, R² y r² de Pearson). La mAP que imprime Ultralytics se guarda como referencia cruzada (`metricas_val_ultralytics.json`), no como cifra principal.
- **Consecuencias:** las diferencias pequeñas con Ultralytics (que interpola y empareja de forma algo distinta) son esperables y se explican en el informe. El conteo anotado del dataset es un *proxy* del conteo manual; el R² frente a conteo manual en campo requiere fotos propias.

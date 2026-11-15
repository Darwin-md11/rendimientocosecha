# ADR-005 — Separación entre la app (`src/`) y el laboratorio (`ml/`)

- **Estado:** aceptada (2026-11-15)
- **Decisión:** `src/` contiene solo lo necesario para ejecutar la app (dependencias livianas de `requirements.txt`). `ml/` contiene entrenamiento, datos sintéticos, diagnósticos y, en la Fase 3, los scripts/notebooks de M0, M1 y M2 (`requirements-ml.txt`, pensado para Colab/Kaggle). `ml/` puede importar `src/`; `src/` nunca importa `ml/` ni PyTorch/Ultralytics a nivel de módulo (import diferido en el adaptador que lo necesite).
- **Consecuencias:** lo vigilan dos pruebas en `tests/test_carga_modelo_y_arquitectura.py`.

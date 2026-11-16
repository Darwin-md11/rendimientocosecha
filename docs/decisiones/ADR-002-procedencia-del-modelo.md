# ADR-002 — Procedencia obligatoria del modelo y advertencias visibles

- **Estado:** aceptada (2026-11-15)
- **Contexto:** la UI mostraba "✅ Inferencia completada" con un modelo entrenado solo con escenas sintéticas, que detecta de 19 a 32 "cerezas" en fotos sin café (`docs/resultados/diagnostico-sintetico.md`).
- **Decisión:** todo detector expone `info() -> ModeloInfo` (nombre, familia, versión, datos de entrenamiento, `validado_con_datos_reales`, SHA-256 de los pesos, fecha). La respuesta del caso de uso incluye `modelo` y `advertencias`. Si el modelo no está validado: banner permanente en Gradio, bloque de aviso en la CLI, franja en la imagen anotada y nunca "✅". Un modelo de familia SINTETICO o MOCK no puede declararse validado (lo impide el propio `ModeloInfo`).
- **Consecuencias:** `validado_con_datos_reales=True` solo se pondrá a un modelo evaluado sobre la partición de prueba real (Fase 4), con su ID de corrida. Las pruebas `tests/test_procedencia_y_contrato.py` fallan si se presenta un resultado no validado sin advertencia.

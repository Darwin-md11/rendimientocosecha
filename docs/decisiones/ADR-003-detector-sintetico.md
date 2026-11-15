# ADR-003 — El detector sintético se conserva solo como demo y regresión

- **Estado:** aceptada por el equipo (2026-11-15)
- **Decisión:** el clasificador de píxeles + watershed entrenado con escenas sintéticas se conserva etiquetado "SINTÉTICO – NO VALIDADO" (`SIKAFE_DETECTOR=sintetico`, alias `ml`). Hoy es el detector por defecto porque no existe otro; cuando haya un modelo validado con datos reales, ese pasará a ser el valor por defecto y el sintético quedará solo para demo y pruebas de regresión.
- **Consecuencias:** su código de entrenamiento y el generador sintético viven en `ml/sintetico/` (laboratorio), no en la app. Sus métricas sobre escenas sintéticas no se reportan como desempeño del sistema.

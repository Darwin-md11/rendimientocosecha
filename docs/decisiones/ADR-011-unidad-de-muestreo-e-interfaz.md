# ADR-011 — Unidad de muestreo (una foto = una planta completa) e interfaz web

- **Estado:** aceptada por el equipo (2026-11-18), por el plazo de entrega.
- **Contexto:** la fórmula de rendimiento escala de lo fotografiado al lote con `plantas_totales_lote / plantas_muestra`, pero la
  unidad de muestreo estaba sin definir. Si se fotografía una rama y se deja `plantas_muestra = 1`, la app trata la rama como una
  planta entera y subestima mucho el lote. Además, la interfaz mostraba todo el reporte como un solo bloque de texto, y el umbral por
  defecto (0,40) no era el de la evaluación de M2 (0,25).
- **Decisión:**
  1. **Cada foto muestra una planta completa.** `plantas_muestra = 1` con fuente "decisión del equipo"; la interfaz ya no lo pide.
  2. Se declaran dos simplificaciones nuevas en cada resultado: la foto solo ve una cara de la planta (el conteo es un mínimo mientras
     la oclusión siga en 1,0), y una sola planta representa a todo el lote.
  3. Umbral de confianza por defecto **0,25** en la interfaz y la CLI: con él se midieron el R² y el MAE del conteo.
  4. Interfaz: tarjetas (en la planta / en el lote), barra de madurez con nombres y números, advertencias siempre visibles con la
     lista completa desplegable, pestañas para la foto, el desglose y el cálculo paso a paso con sus fuentes. Parámetros técnicos en
     "Parámetros avanzados". Colores de madurez validados para daltonismo y contraste en modo claro y oscuro.
- **Alternativas descartadas por el plazo:** unidad = rama con un factor "ramas productivas por planta" (requiere fuente o conteo en
  campo); varias fotos por planta o por lote con promedio (estimación multifoto).
- **Consecuencias:** el kg del lote solo tiene sentido si la foto muestra la planta completa; con fotos de rama, usar el conteo y la
  madurez, no el kg. Queda como trabajo futuro calibrar la oclusión y muestrear varias plantas.

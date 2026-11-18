# ADR-010 — Parámetros agronómicos de Cenicafé y "listo para cosechar"

- **Estado:** aceptada por el equipo (2026-11-18).
- **Contexto:** el rendimiento usaba 1,8 g por cereza y un factor de oclusión de 1,25 sin fuente, y no calculaba pergamino. El
  equipo aportó una investigación (`docs/investigacion/parametros_agronomicos_cenicafe.md`); los valores clave se verificaron
  contra el PDF de Montilla-Pérez et al. (2008), Tablas 17, 18 y 20.
- **Decisión:**
  1. Peso medio por cereza **1,85 g** (café sin seleccionar, variedad Colombia), referencia bibliográfica pendiente de calibración local.
  2. Factor cereza→pergamino seco **0,20** (aproximado), mismo estudio; el pergamino se calcula.
  3. Factor de oclusión **1,0 (sin corrección)**, nuevo tipo `calibracion`: no hay valor bibliográfico y se advierte siempre.
  4. Se reportan **potencial** (todas las cerezas) y **listo para cosechar** (MADURO + SOBREMADURO; Cenicafé, Cartilla 19).
  5. Cada resultado advierte: estimación indicativa no calibrada, un solo peso/factor para todos los estados de madurez, y los
     tres contextos distintos (Caldas, Uganda, Norte de Santander). Los parámetros llevan versión (`version_parametros`).
- **Descartado:** factores por estado de madurez (combinaría estudios con diseños distintos); usar el sesgo de conteo de M2 frente
  a las anotaciones como "oclusión" (corrige al detector, no las cerezas ocultas).
- **Consecuencia:** frente a la versión anterior, el kg estimado baja (1,85/1,8 × 1,0/1,25 ≈ 0,82).

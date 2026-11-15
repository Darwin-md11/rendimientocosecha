# Bitácora del proyecto SIKAFE

> Documento vivo: se actualiza en cada paso. Sirve para retomar el trabajo en otra conversación o por otro integrante.
> Metodología: CRISP-DM (ADR-001). Entrega final: **viernes 2026-11-20**.

## Estado actual

| Fase | Estado | Rama |
|---|---|---|
| 0 — Línea base, honestidad y orden del repo | ✔ código y pruebas listos; **pendiente:** Release del modelo y *push* | `fase-0-honestidad` |
| 1 — Módulo de rendimiento (versión mínima) | siguiente | — |
| 2 — Datos | bloqueada: elegir dataset (equipo) | — |
| 3 — Modelos M0/M1/M2 | bloqueada por la Fase 2 y por GPU en Colab | — |
| 4 — Evaluación | bloqueada por la Fase 3 | — |
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

## Pendientes / preguntas abiertas

- [ ] **Unidad de muestreo** (qué cubre una foto: rama, lado de planta, planta): sin definir. Bloquea la cadena de rendimiento completa de la Fase 1.
- [ ] **Estimación con varias fotos**: sin definir (requiere visto bueno del docente). Mientras tanto, una imagen a la vez.
- [ ] Documento fuente de los factores agronómicos (Cenicafé/FNC): peso por fruto, frutos por rama, ramas por planta, densidad, factor cereza→pergamino.
- [ ] Dataset abierto elegido + licencia verificada. Candidatos encontrados (sin verificar): Croppie (Hugging Face), Roboflow `carlosmm/coffee_cherry`, Sanya et al. (Mendeley, dron).
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

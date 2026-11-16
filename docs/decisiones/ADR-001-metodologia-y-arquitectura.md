# ADR-001 — CRISP-DM como metodología y arquitectura hexagonal

- **Estado:** aceptada (2026-11-15)
- **Contexto:** el docente exige una metodología de desarrollo validada; el proyecto compara modelos (M0, M1, M2) que deben poder intercambiarse sin tocar la app.
- **Decisión:** CRISP-DM (6 fases, ver documento base §7.2) organiza el trabajo y la bitácora (`docs/PROGRESO.md`). El software mantiene la arquitectura hexagonal existente: dominio sin dependencias externas, caso de uso, y adaptadores de detección/UI inyectados solo en `src/infrastructure/composition.py`.
- **Consecuencias:** cada modelo nuevo es un adaptador de `FruitDetectorPort` (o de un futuro `ClasificadorMadurezPort` envuelto en un pipeline de dos etapas) y debe pasar la suite de contrato `tests/contrato_detector.py`.

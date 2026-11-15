# ADR-006 — Se mantiene Gradio para el prototipo

- **Estado:** aceptada (2026-11-15)
- **Contexto:** el documento base admite Gradio o Streamlit; el repositorio ya tiene una UI funcional en Gradio y el plazo es de 5 días.
- **Decisión:** mantener Gradio. Los textos que dependen de la validación del modelo viven en `src/infrastructure/adapters/ui/presentacion.py` (sin dependencia de Gradio) para poder probarlos.

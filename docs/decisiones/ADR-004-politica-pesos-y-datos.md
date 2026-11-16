# ADR-004 — Pesos y datos fuera de git, verificados por SHA-256

- **Estado:** aceptada por el equipo (2026-11-15)
- **Decisión:** los pesos (`models/*.joblib`, `*.pt`, …) y los datos no se versionan en git. Cada modelo tiene un manifiesto versionado (`models/<nombre>.manifest.json`) con `sha256`, versión de scikit-learn (o del framework), procedencia y `url_descarga` (GitHub Release). La app verifica el hash **antes** de deserializar (joblib/pickle ejecuta código) y falla con un mensaje claro si: falta el archivo, falta el manifiesto, el hash no coincide o la versión de scikit-learn instalada no es la registrada. No se reentrena al arrancar.
- **Pasos pendientes del equipo (antes de fusionar a `main`):**
  1. En GitHub: *Releases → Draft a new release*, etiqueta `modelo-sintetico-v1`.
  2. Adjuntar `models/madurez_pixel.joblib` tal como está en `main` (commit `b3456b6`). Su SHA-256 debe ser `c1a18fa9ebfd249bdd57839aeee66313372703feffa81a59c8f6df1cbbd0cc14` (`sha256sum models/madurez_pixel.joblib`).
  3. Publicar. Comprobar con `python scripts/descargar_modelo.py` en un clon limpio.
- **Consecuencias:** la CI descarga los pesos en cada ejecución; sin la Release la CI falla con un mensaje explícito. Los datos de campo (fotos propias) quedan en Drive con su manifiesto de hashes (Fase 2).

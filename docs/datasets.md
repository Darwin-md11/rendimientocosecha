# Inventario de datasets

> Las licencias y cifras salen de la ficha pública de cada dataset (fecha de consulta indicada). **El equipo debe verificarlas en la fuente antes de citarlas en el informe.**
> Las estadísticas del dataset ya preparado (particiones, clases, tamaños de caja) se generan con `python -m ml.datos.preparar_yolo` → `eda.md` / `estadisticas.json`.

| Dataset | Fuente | Licencia | Imágenes / instancias | Clases originales | Uso en SIKAFE | Consultado |
|---|---|---|---|---|---|---|
| **Croppie** | https://huggingface.co/datasets/rgautroncgiar/croppie_coffee_ug | CC BY-SA 4.0 | 633 fotos originales (celular, fincas pequeñas, Arábica y Robusta) con ~61 050 cajas; publicado como 5 836 recortes de entrenamiento + 2 497 de validación de 480×640 | dark_brown, green, low_visibility_unsure, red, yellow | **M2 (en curso)**: train/val/test 70/15/15 por foto original | 2026-11-15 |
| coffee_cherry (CarlosMM) | https://universe.roboflow.com/carlosmm/coffee_cherry-wd4iy | CC BY 4.0 (según la página) | `TODO(fuente)` | madura, semimadura, sobremadura, verde | candidato: evaluación entre dominios | 2026-11-15 (página no accesible desde la sesión) |
| Coffee and Cashew Nut (Sanya et al.) | https://data.mendeley.com/datasets/r46c6bpfpf/1 (DOI 10.17632/r46c6bpfpf.1) | `TODO(fuente)` | 3 000 imágenes de café (dron) | unripe, ripening, ripe, spoilt, coffee_tree | candidato (dominio distinto: dron) | 2026-11-15 |

**Atribución Croppie (obligatoria por CC BY-SA):** "Croppie © 2024 by Producers Direct and Alliance Bioversity & CIAT", financiado por GIZ (Fair Forward – AI for All). La licencia *ShareAlike* implica que los datos derivados (p. ej. el dataset re-particionado) deben compartirse con la misma licencia; los pesos del modelo entrenado: consultar con el docente (ver `docs/legal.md`, pendiente).

**Mapeo de clases** (`configs/clases.yaml`): green→VERDE, yellow→PINTÓN (**supuesto pendiente de confirmar**: en variedades de fruto amarillo, la cereza amarilla puede estar madura), red→MADURO, dark_brown→SOBREMADURO, low_visibility_unsure→excluida (su región se enmascara en la imagen).

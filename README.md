# SIKAFE — Estimación de rendimiento de cosecha de café

SIKAFE analiza una fotografía de un cafeto y:

1. detecta las cerezas;
2. clasifica cada una por estado de madurez (VERDE, PINTÓN, MADURO, SOBREMADURO);
3. estima el rendimiento en **kg de café cereza** para el lote, y en pergamino seco con un factor de Cenicafé.

El proyecto es de Sistemas Inteligentes, en la Universidad de Pamplona. Sigue la metodología CRISP-DM y una arquitectura hexagonal (puertos y adaptadores).

> **Estado (2026-11-18).**
> - La app usa por defecto **M2: YOLO11n entrenado con fotos reales** del dataset Croppie (fincas de Uganda).
> - El modelo se evaluó **una sola vez** en una partición de prueba separada por sesiones de captura. Allí obtuvo mAP@0,5 = 0,603 y R² del conteo = 0,959 frente a las anotaciones.
> - **No se ha evaluado con fotos de Norte de Santander.** La app lo indica en cada resultado: "VALIDADO SOLO EN OTRO DOMINIO".
> - El rendimiento en kg es **indicativo**: usa parámetros bibliográficos (Cenicafé), no calibrados localmente, ni validados con una cosecha real.

## Resultados (partición de prueba)

| Modelo | Tarea | F1 macro de madurez | Otras métricas |
|---|---|---|---|
| M0 — color + textura + SVM (mejor M0) | clasifica cerezas recortadas | 0,659 | exactitud 0,822 |
| M1 — MobileNetV2 con transfer learning | clasifica cerezas recortadas | 0,744 | exactitud 0,865 (sin preentrenar: 0,690) |
| **M2 — YOLO11n (el de la app)** | detecta, clasifica y cuenta | 0,776* | mAP@0,5 **0,603** [IC 95 % por sesiones 0,555–0,658] · conteo R² **0,959** [0,947–0,971] · MAE 1,92 cerezas/imagen |

\* Medido solo en las cerezas que M2 localiza, por lo que tiene sesgo de selección.

**Contraste con la hipótesis:**
- El aprendizaje profundo supera a la línea base clásica: **sí**.
- Meta mAP@0,5 ≥ 0,80: **no se cumple**.
- Meta R² del conteo ≥ 0,80: **se cumple frente a las anotaciones**. No hay conteo manual en campo.

Detalle: [docs/resultados/resultados_test.md](docs/resultados/resultados_test.md). Las tablas y figuras del capítulo 8 se generan con la celda 14 del cuaderno de Colab (ver [Laboratorio](#laboratorio-reproducir-los-resultados)).

## Instalación

Requiere Python ≥ 3.10; se probó con 3.12. Se recomienda usar un entorno virtual.

```bash
python -m venv .venv
.venv\Scripts\activate                                  # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt -r requirements-ml.txt   # app + YOLO (ultralytics instala PyTorch: descarga grande)
python scripts/descargar_modelo.py                       # descarga todos los modelos de la Release y verifica su SHA-256
```

Cada archivo de requisitos tiene un uso:

| Archivo | Para qué |
|---|---|
| `requirements.txt` | La app sin PyTorch. Solo sirve con el detector sintético de demostración. |
| `requirements-ml.txt` | YOLO (M2) y el laboratorio: ultralytics, PyTorch, numba. |
| `requirements-dev.txt` | Pruebas y lint. |

Los pesos **no están en git** ([ADR-004](docs/decisiones/ADR-004-politica-pesos-y-datos.md)). Cada modelo tiene un manifiesto versionado, `models/*.manifest.json`, con:
- el SHA-256 de los pesos;
- su procedencia y la url de su GitHub Release (`modelo-m2-v1`, `modelo-sintetico-v1`).

La app **no arranca** si falta el archivo, si el hash no coincide o si falta una dependencia. En esos casos muestra un mensaje que dice cómo resolverlo. Nunca reentrena ni cambia de detector en silencio.

## Uso

```bash
# Interfaz web: abrir http://127.0.0.1:7860
python -m src.infrastructure.adapters.ui.gradio_app

# Línea de comandos
python -m src.infrastructure.adapters.ui.cli_app --imagen foto.jpg --guardar-anotada resultado.jpg
```

Cada resultado muestra:
- las cerezas por estado de madurez;
- el rendimiento **potencial** (todas las cerezas) y el **listo para cosechar** (MADURO + SOBREMADURO), en kg de café cereza y de pergamino seco;
- los **supuestos del cálculo**: cada factor con su fuente y su estado de calibración;
- la **procedencia del modelo**: nombre, datos de entrenamiento, dominio de validación y SHA-256;
- todas las advertencias.

Las fórmulas y los parámetros están en [docs/modelo-rendimiento.md](docs/modelo-rendimiento.md). Por ejemplo, 10 000 cerezas × 1,85 g = 18,5 kg de café cereza, que a 0,20 equivalen a 3,70 kg de pergamino seco.

El detector se elige con la variable `SIKAFE_DETECTOR`, o con `--detector` en la CLI. En PowerShell: `$env:SIKAFE_DETECTOR="sintetico"`.

| Valor | Detector | Cómo se presenta |
|---|---|---|
| `yolo` (por defecto si existe `models/m2_yolo.manifest.json`) | M2: YOLO11n entrenado con Croppie | VALIDADO SOLO EN OTRO DOMINIO |
| `sintetico` (alias `ml`) | clasificador de píxeles entrenado con escenas sintéticas; no necesita PyTorch | SINTÉTICO – NO VALIDADO |
| `mock` | cajas aleatorias (solo pruebas) | SIMULADO – NO VALIDADO |

Velocidad de M2 medida en Colab: 95,5 ms por imagen en CPU y 12,2 ms en GPU T4. En otro computador variará.

**Problemas frecuentes:**

| Mensaje | Solución |
|---|---|
| "No existe el modelo …" | `python scripts/descargar_modelo.py` |
| "necesita la librería 'ultralytics'" | `pip install -r requirements-ml.txt` (o use `SIKAFE_DETECTOR=sintetico`) |
| "El SHA-256 … no coincide" | Borre el archivo de `models/` y vuelva a descargarlo. |

**Demo sin instalar nada:** la celda 13 del cuaderno de Colab abre la app con un enlace público temporal (`SIKAFE_COMPARTIR=1`). Por defecto la app solo escucha en el propio computador.

**Privacidad (Ley 1581 de 2012):** use fotos de plantas **sin personas**.
- SIKAFE no guarda las fotos por su cuenta. Gradio sí deja las subidas en su carpeta temporal, y la CLI escribe la imagen anotada solo si se le pide con `--guardar-anotada`.
- Al modelo le llegan solo los píxeles, y la imagen anotada sale **sin metadatos EXIF**, que pueden incluir la ubicación GPS. Lo comprueba `tests/test_privacidad.py`.

## Publicar una nueva versión del modelo M2 (equipo)

1. Ejecute `python scripts/publicar_modelo_m2.py --corrida <carpeta con m2_yolo.pt y m2_yolo.manifest.json>`. El script:
   - verifica el SHA-256;
   - verifica que la corrida sea la oficial de `configs/informe.yaml`;
   - verifica que su evaluación final en test ya esté hecha;
   - escribe `models/m2_yolo.manifest.json` con la url de la Release.
2. En GitHub, cree la Release con el tag del manifiesto y adjunte `m2_yolo.pt`.
3. Haga commit **solo del manifiesto**, y siempre **después** de publicar la Release: el CI descarga los pesos desde ella.

## Laboratorio: reproducir los resultados

Todo el entrenamiento y la evaluación se hacen en Google Colab con una GPU T4, con [`ml/notebooks/m2_yolo_colab.ipynb`](ml/notebooks/m2_yolo_colab.ipynb). Hay una versión para Kaggle. Las corridas se guardan en Google Drive (`MyDrive/sikafe/`), así que sobreviven si Colab se desconecta.

| Paso | Comando (desde la raíz del repositorio) |
|---|---|
| Datos: inspeccionar y preparar | `python -m ml.datos.preparar_yolo --origen <croppie> --destino <datos> --dataset croppie --semilla 0` |
| M0: color + textura + RF / SVM (CPU) | `python -m ml.m0_clasico.entrenar --datos <datos>/data.yaml --salida <corridas> --modelo rf` (o `svm`) |
| M1: CNN con transfer learning (GPU) | `python -m ml.m1_cnn.entrenar --config ml/experiments/configs/m1_mobilenetv2_96.yaml --datos <datos>/data.yaml --salida <corridas>` |
| M2: YOLO11n | `python -m ml.m2_yolo.entrenar --config ml/experiments/configs/m2_yolo11n_640.yaml --datos <datos>/data.yaml --salida <corridas>` |
| Evaluar en validación (se puede repetir) | `python -m ml.m2_yolo.evaluar --corrida <corridas>/<id> --split val` |
| Evaluación final en test (**una sola vez**, con el modelo congelado) | `python -m ml.m2_yolo.evaluar --corrida <corridas>/<id> --split test --final` |
| CUDA: kernel propio del descriptor de M0, y YOLO en CPU vs GPU | `python -m ml.benchmark.cuda_m0 …` · `python -m ml.benchmark.yolo_cpu_gpu …` |
| Tablas y figuras del capítulo 8, con intervalos de confianza de M2 | `python -m ml.informe.generar --corridas <corridas> --datos <datos>/data.yaml --benchmarks <benchmarks> --salida <informe>` |

**Reproducibilidad:**
- **Semilla:** 0 en todo.
- **Partición:** 70/15/15 **por sesión de captura**. Las fotos separadas por más de 15 minutos son sesiones distintas: 60 sesiones, que se reparten 40 / 12 / 8. Al preparar el dataset, una prueba de fuga verifica que ninguna sesión quede en dos particiones.
- **Dataset preparado:** hash `9dbcf58b…`.
- **Por corrida:** cada una guarda `config.yaml` y `entorno.json` (versiones, GPU, commit y hash del dataset).
- **Corridas oficiales:** sus ID exactos están en [`configs/informe.yaml`](configs/informe.yaml).
- **Métricas:** son las mismas para todos los modelos, en `src/evaluation/metricas.py` ([ADR-007](docs/decisiones/ADR-007-metricas-comunes.md)).

## Estructura

```
src/                       LA APP (no importa nada de ml/)
  domain/                  entidades (ModeloInfo, supuestos del rendimiento), puertos, cálculo de rendimiento
  application/             caso de uso AnalizarImagenCosechaUseCase y DTOs
  infrastructure/
    adapters/detectors/    YOLO (M2), sintético, mock
    adapters/ui/           Gradio, CLI y textos de presentación
    config/                carga y validación de configs/agronomia.yaml
    ml/                    manifiestos y carga verificada de modelos
    composition.py         único lugar que decide qué detector se inyecta
  evaluation/              métricas comunes (numpy)
ml/                        EL LABORATORIO (Colab / Kaggle)
  datos/                   preparación del dataset (mapeo de clases, partición por sesión, prueba de fuga, EDA)
  m0_clasico/              descriptor de color y textura (CPU y kernel CUDA con numba) + RF / SVM
  m1_cnn/                  CNN sobre recortes (MobileNetV2 / EfficientNet / ResNet)
  m2_yolo/                 entrenamiento y evaluación de YOLO
  benchmark/               CPU vs GPU (CUDA)
  informe/                 tablas, figuras e intervalos de confianza del capítulo 8
  experiments/             configuraciones de cada corrida y registro del entorno
  notebooks/               cuadernos de Colab y Kaggle
  sintetico/, diagnosticos/  detector sintético (demostración) y su diagnóstico
configs/                   clases.yaml (mapeo), agronomia.yaml (factores con fuente), informe.yaml (corridas oficiales)
models/                    manifiestos de los modelos (los pesos se descargan)
scripts/                   descarga verificada y publicación del modelo M2
docs/                      progreso, decisiones (ADR), resultados, datasets, modelo de rendimiento
tests/                     pruebas (incluye la suite de contrato de los detectores)
```

Pruebas y lint: `pip install -r requirements-dev.txt`, luego `python -m pytest` y `ruff check .`. El CI de GitHub Actions las ejecuta en Ubuntu y en Windows.

## Documentación

| Documento | Contenido |
|---|---|
| [docs/PROGRESO.md](docs/PROGRESO.md) | Bitácora, decisiones del equipo y resultados de cada paso. |
| [docs/decisiones/](docs/decisiones/) | ADR-001 a ADR-010: arquitectura, procedencia, pesos, métricas, YOLO, CUDA, parámetros agronómicos. |
| [docs/resultados/resultados_test.md](docs/resultados/resultados_test.md) | Resultados finales en test y su lectura. |
| [docs/modelo-rendimiento.md](docs/modelo-rendimiento.md) | Fórmulas del rendimiento, parámetros con fuente, limitaciones y referencias. |
| [docs/datasets.md](docs/datasets.md) | Datasets, licencias y atribución. |
| [docs/resultados/diagnostico-sintetico.md](docs/resultados/diagnostico-sintetico.md) | Por qué el detector sintético no sirve con fotos reales. |

## Limitaciones conocidas

- **Dominio:** se entrenó y se evaluó solo con Croppie, fotos de celular de fincas de Uganda (2023). Falta validarlo con fotos de Norte de Santander.
- **Conteo:** se comparó con las cajas anotadas del dataset, no con un conteo manual en campo. Hay cerezas ocultas por hojas: el factor de oclusión es 1,0, es decir, no se corrige, y está pendiente de calibrar.
- **Rendimiento:** usa un solo peso medio de cereza (1,85 g) y un solo factor a pergamino (0,20) para todos los estados de madurez. La unidad de muestreo y la estimación con varias fotos por planta **no están definidas**.
- **PINTÓN** es la clase más difícil: AP de 0,257 en M2. La cereza amarilla de Croppie se tomó como PINTÓN por decisión del equipo.
- Cada modelo se entrenó con una sola semilla. M0 y M1 no tienen intervalos de confianza, porque no se guardaron sus predicciones por recorte.

## Datos y licencias

- **Croppie:** © 2024 Producers Direct and Alliance Bioversity & CIAT, licencia **CC BY-SA 4.0** ([ficha del dataset](https://huggingface.co/datasets/rgautroncgiar/croppie_coffee_ug)). Exige atribución, y que lo derivado se comparta con la misma licencia (*ShareAlike*). Ver [docs/datasets.md](docs/datasets.md).
- **Ultralytics (YOLO) y licencia del repositorio:** `TODO(fuente)`. Están pendientes de verificar y documentar en `docs/legal.md`, porque la licencia de Ultralytics condiciona cómo se distribuyen la app y los pesos. Mientras tanto, el manifiesto de M2 lleva la licencia marcada como pendiente.
- Modelo de rendimiento: parámetros de Montilla-Pérez et al. (2008), *Cenicafé* 59(2), y de Cenicafé (2004), Cartilla 19. Referencias completas en [docs/modelo-rendimiento.md](docs/modelo-rendimiento.md).

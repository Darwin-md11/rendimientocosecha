# SIKAFE - Estimación de rendimiento de cosecha de café

Sistema que estima el rendimiento de cosecha (kg de café cereza) a partir de una fotografía del cafeto:
detecta las cerezas, clasifica su estado de madurez y escala el resultado al lote.
Arquitectura hexagonal (puertos y adaptadores). Metodología CRISP-DM.

> ⚠️ **Estado actual:** el único modelo incluido se entrenó **solo con escenas sintéticas** y está etiquetado
> **"SINTÉTICO – NO VALIDADO"**. En fotos sin café detecta de 19 a 32 "cerezas"
> ([diagnóstico](docs/resultados/diagnostico-sintetico.md)). La UI y la CLI lo advierten en cada resultado.
> El detector con fotos reales (**M2, YOLO entrenado con el dataset Croppie**) está listo para entrenarse en Colab:
> [`ml/notebooks/m2_yolo_colab.ipynb`](ml/notebooks/m2_yolo_colab.ipynb). Estado en [docs/PROGRESO.md](docs/PROGRESO.md).

## Instalación

```bash
pip install -r requirements.txt          # app
pip install -r requirements-dev.txt      # + pruebas y lint
python scripts/descargar_modelo.py       # pesos desde la Release, con verificación SHA-256
```

Los pesos no se guardan en git ([ADR-004](docs/decisiones/ADR-004-politica-pesos-y-datos.md)): cada modelo tiene un
manifiesto versionado (`models/*.manifest.json`) con su SHA-256 y la versión de scikit-learn con la que se guardó.
La app **no arranca** (con un mensaje que explica cómo resolverlo) si falta el archivo, el hash no coincide o la
versión de scikit-learn instalada es otra. Nunca reentrena ni cambia de detector en silencio.

## Uso

```bash
# Línea de comandos (sin --imagen usa una escena sintética de demostración)
python -m src.infrastructure.adapters.ui.cli_app
python -m src.infrastructure.adapters.ui.cli_app --imagen foto.jpg --guardar-anotada resultado.jpg

# Interfaz web (Gradio)
python -m src.infrastructure.adapters.ui.gradio_app

# Pruebas y lint
python -m pytest
ruff check .
```

Selección de detector con `SIKAFE_DETECTOR` (o `--detector` en la CLI):

| Valor | Detector | Procedencia |
|---|---|---|
| `yolo` (por defecto si existe `models/m2_yolo.manifest.json`) | M2: YOLO entrenado con fotos reales (Croppie) | la de su manifiesto (p. ej. "VALIDADO SOLO EN OTRO DOMINIO") |
| `sintetico` (por defecto si no hay YOLO; alias `ml`) | clasificador de píxeles + watershed | SINTÉTICO – NO VALIDADO |
| `mock` | cajas aleatorias, ignora la imagen | SIMULADO – NO VALIDADO |

Para usar YOLO: copie `m2_yolo.pt` y `m2_yolo.manifest.json` (salida del notebook) a `models/` e instale `pip install -r requirements-ml.txt`.

Cada resultado incluye la **procedencia del modelo** (nombre, familia, versión, datos de entrenamiento,
si fue validado con datos reales, SHA-256 de los pesos) y sus advertencias.

## Cómo funciona el detector sintético

`MLFruitDetectorAdapter` (implementa `FruitDetectorPort`):

1. Reduce la imagen a un lado mayor de 320 px.
2. Un clasificador de píxeles (`HistGradientBoostingClassifier`) asigna a cada píxel: fondo, verde, pintón, maduro o sobremaduro.
3. Limpia la máscara y separa cerezas que se tocan con *watershed* sobre la transformada de distancia.
4. Descarta regiones pequeñas, alargadas o irregulares (hojas) y las de baja confianza.
5. Cada región es una detección: estado = clase con mayor probabilidad media, confianza = esa probabilidad.

Sobre 80 escenas sintéticas nuevas obtuvo precisión 0,88 y recall 0,76. Es una medida **optimista**: la prueba
sale del mismo generador que el entrenamiento. Con perturbaciones leves (brillo, desenfoque, compresión) el error
de conteo llega a ~100 %. **No hay medición con fotos reales.**

Reentrenarlo (laboratorio, no forma parte de la app):

```bash
python -m ml.sintetico.training                       # escribe el .joblib y su manifiesto
python -m ml.sintetico.training --imagenes datos/     # + fotos con máscara foto_mask.png (0..4 por píxel)
python ml/diagnosticos/diagnostico_detector_ml.py     # robustez y falsos positivos
```

## Laboratorio: datos, entrenamiento y evaluación de M2

```bash
python -m ml.datos.preparar_yolo --origen <croppie_raw> --inspeccionar
python -m ml.datos.preparar_yolo --origen <croppie_raw> --destino <sikafe_yolo> --dataset croppie --semilla 0
python -m ml.m2_yolo.entrenar --config ml/experiments/configs/m2_yolo11n_640.yaml --datos <sikafe_yolo>/data.yaml --salida <corridas>
python -m ml.m2_yolo.evaluar --corrida <corridas>/<id> --split val
python -m ml.m2_yolo.evaluar --corrida <corridas>/<id> --split test --final     # una sola vez
```

Las métricas (mAP@0,5, mAP@0,5:0,95, P/R/F1, matriz de confusión, MAE/RMSE/R² de conteo, tiempo) salen de
`src/evaluation/metricas.py`, común a todos los modelos ([ADR-007](docs/decisiones/ADR-007-metricas-comunes.md)).
Datasets y licencias: [docs/datasets.md](docs/datasets.md).

## Estructura

```
src/                      LA APP (dependencias livianas)
  domain/                 modelos (incl. ModeloInfo), puertos, servicio de rendimiento, excepciones
  application/            caso de uso AnalizarImagenCosechaUseCase y DTOs
  infrastructure/
    adapters/detectors/   YOLO (M2), sintético (MLFruitDetectorAdapter), mock
    adapters/ui/          CLI, Gradio y textos de presentación
    ml/                   características y carga verificada del modelo
    composition.py        único lugar que decide qué detector se inyecta
  evaluation/             métricas comunes (numpy)
ml/                       EL LABORATORIO (Colab/Kaggle)
  datos/                  preparación de datasets YOLO (mapeo, partición por grupo, fuga, EDA)
  m2_yolo/                entrenamiento y evaluación de YOLO
  experiments/            configs por corrida y registro de entorno
  notebooks/              notebook de Colab
  sintetico/, diagnosticos/
configs/                  clases.yaml (mapeo de clases por dataset)
models/                   manifiestos de los modelos (los pesos se descargan)
scripts/                  descarga de pesos con verificación
docs/                     PROGRESO.md, decisiones (ADR), resultados
tests/                    pruebas (incl. suite de contrato para detectores)
```

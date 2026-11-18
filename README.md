# SIKAFE - Estimación de rendimiento de cosecha de café

Sistema que estima el rendimiento de cosecha (kg de café cereza) a partir de una fotografía del cafeto:
detecta las cerezas, clasifica su estado de madurez y escala el resultado al lote.
Arquitectura hexagonal (puertos y adaptadores). Metodología CRISP-DM.

> ⚠️ **Estado actual:** el único modelo incluido se entrenó **solo con escenas sintéticas** y está etiquetado
> **"SINTÉTICO – NO VALIDADO"**. En fotos sin café detecta de 19 a 32 "cerezas"
> ([diagnóstico](docs/resultados/diagnostico-sintetico.md)). La UI y la CLI lo advierten en cada resultado.
> Los modelos con datos reales (M0, M1, M2) están en desarrollo: ver [docs/PROGRESO.md](docs/PROGRESO.md).

## Instalación

```bash
pip install -r requirements.txt          # app
pip install -r requirements-dev.txt      # + pruebas y lint
python scripts/descargar_modelo.py       # pesos desde la Release, con verificación SHA-256

1. Usarla en cualquier PC
Instale las dependencias, de preferencia en un entorno virtual:
pip install -r requirements.txt -r requirements-ml.txt

Esto instala PyTorch para CPU, que es una descarga grande.

2. Descargue los modelos:
python scripts/descargar_modelo.py

Baja los modelos y verifica el hash. En el PC donde publicó dirá “Ya existe y el hash coincide”. Si quiere probar la descarga real, borre antes models\m2_yolo.pt.

3. Abra la interfaz web:
python -m src.infrastructure.adapters.ui.gradio_app

Luego entre a http://127.0.0.1:7860. YOLO es el detector por defecto. En la CPU de Colab tardó unos 95 ms por imagen; en su PC el tiempo será otro.
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
| `sintetico` (por defecto; alias `ml`) | clasificador de píxeles + watershed | SINTÉTICO – NO VALIDADO |
| `mock` | cajas aleatorias, ignora la imagen | SIMULADO – NO VALIDADO |

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

## Estructura

```
src/                      LA APP (dependencias livianas)
  domain/                 modelos (incl. ModeloInfo), puertos, servicio de rendimiento, excepciones
  application/            caso de uso AnalizarImagenCosechaUseCase y DTOs
  infrastructure/
    adapters/detectors/   sintético (MLFruitDetectorAdapter), mock
    adapters/ui/          CLI, Gradio y textos de presentación
    ml/                   características y carga verificada del modelo
    composition.py        único lugar que decide qué detector se inyecta
ml/                       EL LABORATORIO (entrenamiento, datos, diagnósticos; Colab/Kaggle)
models/                   manifiestos de los modelos (los pesos se descargan)
scripts/                  descarga de pesos con verificación
docs/                     PROGRESO.md, decisiones (ADR), resultados
tests/                    pruebas (incl. suite de contrato para detectores)
```

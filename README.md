# SIKAFE - Estimación de rendimiento de cosecha de café

Sistema que estima el rendimiento de cosecha (kg de café cereza) a partir de una fotografía del cafeto:
detecta las cerezas, clasifica su estado de madurez y escala el resultado al lote.
Arquitectura Hexagonal (puertos y adaptadores).

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Línea de comandos (sin --imagen usa una escena sintética de demostración)
python -m src.infrastructure.adapters.ui.cli_app
python -m src.infrastructure.adapters.ui.cli_app --imagen foto.jpg --guardar-anotada resultado.jpg

# Interfaz web (Gradio)
python -m src.infrastructure.adapters.ui.gradio_app

# Pruebas
python -m pytest
```

Por defecto se usa el detector con machine learning. `SIKAFE_DETECTOR=mock` (o `--detector mock` en la CLI)
activa el detector simulado, que genera cajas aleatorias y solo sirve para probar la arquitectura.

## Cómo funciona el detector con ML

`MLFruitDetectorAdapter` (implementa `FruitDetectorPort`):

1. Reduce la imagen a un lado mayor de 320 px.
2. Un clasificador de píxeles (`HistGradientBoostingClassifier` de scikit-learn) asigna a cada píxel una clase:
   fondo, verde, pintón, maduro o sobremaduro. Usa color (RGB/HSV/croma), contexto a varias escalas, textura y bordes.
3. Limpia la máscara y separa cerezas que se tocan con *watershed* sobre la transformada de distancia.
4. Descarta regiones pequeñas, alargadas o irregulares (hojas) y las de baja confianza.
5. Cada región es una detección: estado = clase con mayor probabilidad media, confianza = esa probabilidad.

El cálculo de rendimiento (`CalculadoraRendimientoService`) no cambió.

## Calidad medida (¡leer antes de confiar en los números!)

El modelo incluido en `models/madurez_pixel.joblib` se entrenó **solo con escenas sintéticas**
(`src/infrastructure/ml/synthetic_data.py`), porque el repositorio no tiene fotos reales etiquetadas.
Resultados sobre 80 escenas sintéticas nuevas (`python -m src.infrastructure.ml.training`):

| Métrica | Valor |
|---|---|
| Exactitud por píxel | 0.97 |
| Precisión de detección | 0.88 |
| Recall de detección | 0.76 |
| Estado correcto en frutos detectados | 0.96 |
| Error relativo de conteo por escena | 0.20 |

Estas cifras son **optimistas**: el conjunto de evaluación sale del mismo generador que el de entrenamiento.
Con fotos reales de campo (otra luz, otras variedades, frutos muy ocultos) el desempeño será menor y
**no está medido**. Los filtros de forma y tamaño del detector se calibraron con escenas sintéticas de validación.

## Reentrenar con fotos reales (recomendado)

Cree una carpeta donde cada foto `lote1.jpg` tenga su máscara `lote1_mask.png`, una imagen de un canal
donde cada píxel vale `0`=fondo, `1`=verde, `2`=pintón, `3`=maduro, `4`=sobremaduro. Luego:

```bash
python -m src.infrastructure.ml.training --imagenes datos/ --salida models/madurez_pixel.joblib
# con suficientes fotos reales, también se puede entrenar solo con ellas:
python -m src.infrastructure.ml.training --imagenes datos/ --sin-sinteticas
```

Si el modelo guardado no existe o fue creado con otra versión de scikit-learn, el sistema entrena uno nuevo
con datos sintéticos automáticamente (~30 s) la primera vez que se usa.

## Estructura

```
src/
  domain/          modelos, puertos (FruitDetectorPort), servicio de rendimiento, excepciones
  application/     caso de uso AnalizarImagenCosechaUseCase y DTOs
  infrastructure/
    adapters/detectors/   MLFruitDetectorAdapter, MockFruitDetectorAdapter
    adapters/ui/          CLI y Gradio
    ml/                   características, datos sintéticos, modelo, entrenamiento/evaluación
    composition.py        elige qué detector se inyecta
models/            modelo entrenado (.joblib)
tests/             pruebas con pytest
```

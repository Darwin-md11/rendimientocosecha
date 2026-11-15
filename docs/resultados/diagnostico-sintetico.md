# Diagnóstico del detector sintético (línea base de la Fase 0)

> **Modelo:** clasificador de píxeles + watershed, familia **SINTÉTICO – NO VALIDADO** (600 escenas sintéticas, 0 imágenes reales).
> Estas cifras describen la **fragilidad** del modelo de demostración; **no** son el desempeño del sistema en campo.

## Reproducibilidad

| Campo | Valor |
|---|---|
| Comando | `python ml/diagnosticos/diagnostico_detector_ml.py --escenas 30` |
| Código evaluado | commit `b3456b6` (antes de los cambios de la Fase 0; el detector no cambió) |
| Modelo | `models/madurez_pixel.joblib`, SHA-256 `c1a18fa9ebfd249bdd57839aeee66313372703feffa81a59c8f6df1cbbd0cc14` |
| Semillas | escenas: `_SEMILLA_PRUEBA = 2000` (de `ml/sintetico/training.py`); ruido: 1 |
| Umbral / IoU | 0,40 / ≥ 0,30 (los mismos de `training.evaluar`) |
| Versiones | Python 3.13.18, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, scikit-image 0.26.0, Pillow 12.3.0 |
| Hardware | sandbox en la nube, 2 vCPU Intel Xeon @ 2,10 GHz |
| Fecha | 2026-11-15 |

Las dos fotos de Ultralytics (`bus.jpg`, `zidane.jpg`) se omitieron: el entorno no tenía acceso a internet desde Python.

## Salida completa (sin editar)

```

1) ROBUSTEZ sobre 30 escenas sintéticas perturbadas (umbral 0.4, IoU>=0.3)
condición                            P     R    F1  estado  det/gt  err.cont
sin cambios (≈ README)            0.88  0.77  0.82    0.96    0.88      0.17
brillo x0.6 (sombra)              0.92  0.72  0.80    0.96    0.78      0.19
brillo x1.4 (sol fuerte)          0.49  0.73  0.59    0.96    1.51      1.08
balance cálido (R+15%, B-15%)     0.78  0.73  0.76    0.89    0.94      0.32
balance frío (R-15%, B+15%)       0.92  0.76  0.83    0.86    0.83      0.18
saturación x0.6 (nublado)         0.82  0.73  0.78    0.86    0.89      0.22
desenfoque gauss σ=2              0.51  0.69  0.59    0.93    1.34      0.92
ruido gauss σ=12                  0.69  0.64  0.66    0.95    0.94      0.43
JPEG calidad 25                   0.62  0.74  0.67    0.94    1.20      0.52
resolución a la mitad             0.76  0.72  0.74    0.97    0.95      0.35

2) FALSOS POSITIVOS en imágenes SIN cerezas de café (lo correcto es 0)
  skimage: taza de café     600x400    ->  22 'cerezas' {'MADURO': 4, 'SOBREMADURO': 2, 'PINTON': 16}
  skimage: astronauta       512x512    ->  29 'cerezas' {'MADURO': 4, 'SOBREMADURO': 18, 'VERDE': 7}
  skimage: gato             451x300    ->  32 'cerezas' {'SOBREMADURO': 25, 'PINTON': 7}
  skimage: cohete           640x427    ->  19 'cerezas' {'PINTON': 1, 'SOBREMADURO': 11, 'VERDE': 7}

3) TIEMPO por imagen (mediana de 3, tu CPU)
  640x480    0.49 s
  1920x1080   0.33 s
  4000x3000   0.51 s

real	2m44.677s
user	5m1.331s
sys	0m0.617s
```

## Lectura

- Sin perturbaciones reproduce el README (P 0,88, R 0,77). Es optimista: el conjunto de prueba sale del mismo generador que el de entrenamiento.
- Cambios leves de toma rompen el conteo: con brillo ×1,4 el error de conteo es 108 % y con desenfoque σ=2 es 92 %; JPEG calidad 25 y ruido σ=12 bajan el F1 a 0,67 y 0,66.
- **En imágenes sin café detecta de 19 a 32 "cerezas"**: segmenta por color y forma, no tiene noción de "esto no es café".
- Motivó ADR-002 (procedencia y advertencias) y ADR-003 (el sintético queda solo como demo y regresión).

# Resultados finales en la partición de prueba (test)

> **Origen de cada cifra:** salida de `--final` de cada corrida (`metricas_test.json` y `resumen_test.md` en
> `MyDrive/sikafe/corridas/<id>/`), ejecutada **una sola vez** el 2026-11-18 con los modelos congelados (decisión del equipo).
> Dataset Croppie preparado `9dbcf58b…`, partición por sesión de captura (15 min): test = 8 sesiones, 1 258 imágenes, 22 219 cerezas
> anotadas. Métricas: `src/evaluation/metricas.py` (común a todos los modelos, ADR-007). Una sola semilla (0) por modelo.

## 1. Clasificación de madurez (las mismas 22 219 cerezas para M0 y M1)

| Modelo | Corrida | Exactitud | **F1 macro** | F1 VERDE | F1 PINTÓN | F1 MADURO | F1 SOBREMADURO | ms/recorte |
|---|---|---|---|---|---|---|---|---|
| M0 — color+textura + Random Forest | `m0_rf_s0_20261118-0342` | 0,839 | 0,633 | 0,904 | 0,391 | 0,873 | 0,365 | 0,036 (CPU) |
| M0 — color+textura + SVM (RBF) | `m0_svm_s0_20261118-0403` | 0,822 | 0,659 | 0,896 | 0,442 | 0,862 | 0,434 | 9,36 (CPU) |
| M1 — MobileNetV2 **sin** preentrenar | `m1_mobilenetv2_96_sin_preentrenar_s0_20261118-0540` | 0,836 | 0,690 | 0,907 | 0,498 | 0,892 | 0,464 | 0,240 (GPU) |
| M1 — MobileNetV2 con transfer learning | `m1_mobilenetv2_96_s0_20261118-0456` | 0,865 | **0,744** | 0,918 | 0,496 | 0,922 | 0,640 | 0,160 (GPU) |
| M2 — YOLO11n, en las cerezas que localiza* | `m2_yolo11n_640_s0_20261116-1754` | 0,908 | 0,776 | — | — | — | — | — |

\* M2 detecta y clasifica en una pasada; su exactitud/F1 se mide solo sobre las cerezas que localiza (IoU ≥ 0,5, confianza ≥ 0,25),
con **sesgo de selección** (las que encuentra suelen ser las más visibles). No es idéntica a la tarea de M0/M1 (todas las cerezas anotadas).

Soporte por clase en test: VERDE 11 219 · PINTÓN 1 631 · MADURO 8 747 · SOBREMADURO 622. La exactitud está dominada por VERDE (50 %):
el **F1 macro** es la métrica de comparación.

## 2. Detección y conteo (M2)

| Métrica (test) | Valor |
|---|---|
| mAP@0,5 | **0,603** |
| mAP@0,5:0,95 | 0,424 |
| AP@0,5 por clase: VERDE / PINTÓN / MADURO / SOBREMADURO | 0,811 / 0,257 / 0,867 / 0,477 |
| Precisión / Recall / F1 con clase (confianza 0,25) | 0,840 / 0,780 / 0,809 |
| Precisión / Recall / F1 solo localización | 0,924 / 0,858 / 0,890 |
| Conteo por imagen frente al conteo **anotado**: MAE / RMSE / sesgo | 1,92 / 5,64 / −1,27 cerezas |
| Conteo por imagen: R² / r² de Pearson | **0,959** / 0,980 |
| Tiempo de inferencia (mediana, T4, adaptador de la app) | 15 ms/imagen |

## 3. Contraste con la hipótesis y las metas

| Afirmación | Resultado en test | ¿Se cumple? |
|---|---|---|
| El aprendizaje profundo supera a la línea base clásica en F1 | F1 macro: M1 0,744 y M2 0,776 (cerezas localizadas) frente a 0,659 del mejor M0 | **Sí** (1 semilla, sin intervalo de confianza) |
| ... y en mAP | M0 no detecta (clasifica recortes): la comparación de mAP no aplica | No comparable |
| mAP@0,5 ≥ 0,80 | 0,603 | **No** |
| R² del conteo ≥ 0,80 | 0,959 frente al conteo **anotado** del dataset | Sí frente a anotaciones; **sin validar frente a conteo manual en campo** |
| Transfer learning mejora a la CNN (experimento d) | F1 macro 0,744 vs 0,690 (+0,054); exactitud 0,865 vs 0,836 | Sí (1 semilla) |

## 4. Lectura y limitaciones (para el capítulo 8.4)

- **PINTÓN es la clase más difícil en todos los modelos** (F1 0,39–0,50; AP 0,257 en M2): es la transición entre verde y rojo y
  además depende del supuesto amarilla→PINTÓN del mapeo de clases (pendiente de confirmar). SOBREMADURO es escasa (2,8 % del test).
- M1 tiene **recall alto y precisión baja** en las clases minoritarias (PINTÓN 0,715 / 0,380): efecto de los pesos por clase.
- Las cifras de test son iguales o algo mejores que las de validación: no hay señal de que elegir hiperparámetros en val haya
  inflado los resultados. La mezcla de clases cambia entre particiones (sesiones de distintas fechas de la cosecha).
- **Dominio:** todo es Croppie (fotos de celular en fincas de Uganda, 2023). **No se ha evaluado con fotos de Norte de Santander**;
  la app lo indica ("VALIDADO SOLO EN OTRO DOMINIO").
- **Conteo:** R² frente a las cajas anotadas, no frente a un conteo manual en campo; el rendimiento en kg no está validado.
- **Comparación de mAP:** la mAP propia (0,603) usa NMS agnóstico (una etiqueta por cereza, lo que exige el conteo); Ultralytics, con
  NMS por clase, reportó 0,673 en validación. Se informa la propia como cifra principal (ADR-007).
- Una sola semilla por configuración y sin intervalos de confianza (tiempo de cómputo). M0 entrenado con 30 000 recortes submuestreados.

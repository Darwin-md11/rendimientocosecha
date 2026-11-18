# Investigación: parámetros agronómicos para estimar el rendimiento del café a partir del conteo de cerezas

**Tema:** peso promedio del fruto de café y conversión de café cereza a café pergamino seco  
**Aplicación:** sistema de reconocimiento y conteo de frutos mediante visión artificial  
**Fuentes principales:** Centro Nacional de Investigaciones de Café (Cenicafé) y Federación Nacional de Cafeteros de Colombia (FNC)  
**Fecha de consulta:** 18 de noviembre de 2026

---

## 1. Propósito de la investigación

El sistema de visión artificial cuenta frutos de café detectados en imágenes. Para convertir ese conteo en una estimación de masa —por ejemplo, kilogramos de café cereza o de café pergamino seco— se necesitan parámetros agronómicos que no se obtienen directamente del modelo de visión artificial.

Los parámetros investigados son:

1. **Peso promedio de una cereza de café**, expresado en gramos por fruto.
2. **Factor de conversión de café cereza a café pergamino seco**, expresado en kilogramos de pergamino seco por kilogramo de café cereza.
3. **Factor de oclusión o corrección del conteo**, que no es un valor bibliográfico universal: debe estimarse mediante validación con conteos manuales en las condiciones del proyecto.

El objetivo de este documento es reunir evidencia técnica, explicar las diferencias entre estudios y recomendar cómo registrar e implementar los parámetros en el software sin presentar estimaciones como mediciones reales.

## 2. Hallazgos principales

La investigación más útil para establecer una primera referencia es el artículo de Montilla-Pérez et al. (2008), publicado en la revista *Cenicafé*. Estudió café variedad Colombia y reportó pesos medios de 1,99 g por fruto para una muestra seleccionada y de 1,85 g para una muestra sin seleccionar. El mismo trabajo reportó relaciones de 4,94 kg de café cereza por kg de pergamino seco para la muestra seleccionada y de 4,89:1 para la muestra sin seleccionar. En su tabla de factores de conversión, Cenicafé presenta 0,20 para ambas muestras.

Por tanto, el valor de **1,8 g** que aparece actualmente en el código está cerca del resultado de 1,85 g de la muestra sin seleccionar, pero no debe conservarse como si ya tuviera una fuente: el código original no documentaba su procedencia. Puede sustituirse por 1,85 g como referencia bibliográfica provisional cuando la definición de la muestra y las condiciones del cultivo sean razonablemente comparables.

El factor **0,20** también tiene respaldo bibliográfico como valor aproximado para convertir café cereza en pergamino seco. Sin embargo, otros estudios muestran que el estado de madurez y las condiciones del beneficio cambian la relación. En consecuencia, ni 1,85 g ni 0,20 deben tratarse como constantes universales para todas las variedades, fincas, cosechas y procesos.

## 3. Peso promedio de una cereza de café

### 3.1. Estudio principal: Montilla-Pérez et al. (2008)

**Referencia:** *Caracterización de algunas propiedades físicas y factores de conversión del café durante el proceso de beneficio húmedo tradicional*. Revista Cenicafé, 59(2), 120–142.

**Variedad estudiada:** Colombia (*Coffea arabica L.*).

**Localización:** fincas de Chinchiná, Manizales, Villamaría y Palestina, departamento de Caldas.

**Periodo:** tres épocas de la cosecha del segundo semestre de 2005.

**Diseño de las muestras:** de la masa cosechada se prepararon dos tipos de muestra:

- **Café seleccionado:** cereza madura seleccionada, sin defectos.
- **Café sin seleccionar:** muestra tomada tal como fue cosechada, sin realizar la misma selección previa.

Para la medición del peso y de los diámetros se tomaron 100 frutos de la muestra de 1 kg y se midieron individualmente. El estudio también siguió el proceso de beneficio húmedo y secado para determinar características físicas y factores de conversión.

**Resultados de peso por fruto:**

| Referencia de la tabla del estudio | Peso de un fruto |
|---|---:|
| Uribe (1977), valor de referencia incluido por los autores | 2,00 g |
| Muestra seleccionada | 1,99 g |
| Muestra sin seleccionar | 1,85 g |

**Interpretación:**

- Si el sistema cuenta exclusivamente frutos maduros y sanos, el valor de 1,99 g puede ser la referencia bibliográfica más cercana entre las muestras de este estudio.
- Si el sistema cuenta una masa cosechada sin selección previa, 1,85 g es una referencia más cercana a esa condición experimental.
- La selección de uno u otro valor debe depender de qué está contando realmente el modelo. No basta con escoger el valor que más se acerque al parámetro que ya está en el código.
- El estudio se hizo con variedad Colombia y fincas de Caldas. No demuestra que el mismo promedio se aplique a otras variedades o a la zona de validación del proyecto.

**Comparación con el parámetro actual:**

El sistema utiliza actualmente 1,8 g por cereza, sin fuente documentada. La diferencia frente al valor de 1,85 g es de 0,05 g por fruto, aproximadamente 2,7 % respecto a 1,85 g. La cercanía numérica no sustituye la necesidad de documentar y validar el parámetro.

### 3.2. Estudio complementario: Salazar et al. (1994)

**Referencia:** *Crecimiento del fruto de café Coffea arabica L. var. Colombia*. Revista Cenicafé, 45(2), 41–50.

Este trabajo hizo seguimiento al crecimiento del fruto de café variedad Colombia desde la floración hasta la maduración, midiendo el peso fresco y el peso seco. Los autores reportaron pesos máximos observados de:

- **Peso fresco máximo:** 1,66 g por fruto.
- **Peso seco máximo:** 0,56 g por fruto.

El estudio describe tres fases de crecimiento y reporta que la madurez completa se alcanza alrededor de los 240 días después de la floración en las condiciones observadas. También analiza la evolución del contenido de humedad durante el desarrollo del fruto.

**Cómo interpretar este resultado:** el valor de 1,66 g no debe reemplazar automáticamente los 1,85 g o 1,99 g del estudio de 2008. Son investigaciones diferentes, con objetivos y procedimientos distintos. El trabajo de 1994 estudia la curva de crecimiento del fruto; el de 2008 caracteriza frutos de muestras cosechadas y compara una muestra seleccionada con otra sin seleccionar. Por ello, los valores no son directamente intercambiables.

## 4. Conversión de café cereza a pergamino seco

### 4.1. Estudio principal: Montilla-Pérez et al. (2008)

El artículo de Cenicafé de 2008 presenta dos maneras complementarias de expresar la conversión.

**Relación de conversión:** cantidad de kilogramos de café cereza necesarios para obtener un kilogramo de café pergamino seco.

| Tipo de muestra | Relación café cereza : pergamino seco |
|---|---:|
| Valor de referencia de Uribe (1977) incluido en el estudio | 4,50 : 1 |
| Café seleccionado | 4,94 : 1 |
| Café sin seleccionar | 4,89 : 1 |

**Factor de conversión:** cantidad de kilogramos de pergamino seco estimados por cada kilogramo de café cereza.

| Tipo de muestra | Factor cereza → pergamino |
|---|---:|
| Valor de referencia de Uribe (1977) | 0,22 |
| Café seleccionado | 0,20 |
| Café sin seleccionar | 0,20 |

El valor 0,20 es el factor reportado en la tabla de Cenicafé. Es un valor redondeado: los inversos matemáticos de 4,94 y 4,89 son aproximadamente 0,2024 y 0,2045, respectivamente. Para reproducir el valor bibliográfico del artículo y mantener una configuración sencilla, puede utilizarse 0,20, siempre identificándolo como aproximado.

### 4.2. Efecto del estado de madurez: Marín-López et al. (2003)

**Referencia:** *Relación entre el estado de madurez del fruto del café y las características de beneficio, rendimiento y calidad de la bebida*. Revista Cenicafé, 54(4), 297–315.

**Material estudiado:** frutos rojos de café variedad Colombia en diferentes grados de madurez. Las muestras se evaluaron semanalmente entre 182 y 231 días después de la floración. El beneficio se hizo por vía húmeda, con desmucilaginado mecánico y secado solar.

El estudio reportó la siguiente relación entre kilogramos de café cereza y kilogramos de café pergamino seco:

| Estado del fruto | Días después de floración (ddf) | Relación cereza : pergamino seco |
|---|---:|---:|
| Verde 1 | 182 | 11,43 : 1 |
| Verde 2 | 189 | 11,47 : 1 |
| Verde 3 | 196 | 25,34 : 1 |
| Verde amarillo | 203 | 9,87 : 1 |
| Pintón | 210 | 6,27 : 1 |
| Maduro | 217 | 5,04 : 1 |
| Sobremaduro | 224 | 5,25 : 1 |
| Seco | 231 | 68,41 : 1 |

Los mejores resultados de conversión entre los estados comparados correspondieron al fruto maduro (5,04:1) y al sobremaduro (5,25:1). El estudio también reportó proporciones de pergamino seco de 94,59 % para el estado maduro y 88,64 % para el sobremaduro.

**Interpretación:** la madurez tiene un efecto importante en el rendimiento. Por ejemplo, la relación 25,34:1 del estado verde 3 significa que se requiere mucha más masa de café cereza para obtener un kilogramo de pergamino seco que en el estado maduro. Por ello, aplicar un único factor a un conteo que mezcle frutos verdes, pintones, maduros, sobremaduros y secos puede introducir un error considerable.

Los autores también señalan que la relación de conversión depende de la calidad del café cereza y de la tecnología utilizada para beneficiarlo, incluido el tipo y la calibración de los equipos.

### 4.3. Efecto de los frutos sobremaduros: Sanz-Uribe et al. (2018)

**Referencia:** *Retención de pases: Una opción para mejorar la productividad de la mano de obra en la cosecha de café*. Avances Técnicos Cenicafé, 488, 1–8.

El documento explica que un fruto sobremaduro tiene en promedio un 12 % menos de masa que un fruto maduro y que, en condiciones normales, ambos producen los mismos dos granos. También presenta una comparación de conversión de cereza a pergamino seco para muestras con distintas proporciones de frutos sobremaduros:

| Porcentaje de frutos sobremaduros en la muestra | Café cereza requerido para 1 kg de pergamino seco |
|---:|---:|
| 4,01 % | 5,01 kg |
| 81,7 % | 4,51 kg |

El documento indica que en la muestra con 81,7 % de frutos sobremaduros se necesitó alrededor de 10 % menos de café cereza para obtener un kilogramo de pergamino seco. La nota al pie del documento atribuye los datos de esta comparación a un informe técnico de Sanz-Uribe de 2016.

**Interpretación y cautela:** este resultado demuestra que la composición de la masa cosechada puede modificar el factor de conversión. No debe utilizarse 4,51:1 como factor universal ni combinarse mediante un promedio simple con los valores de otros experimentos. La comparación corresponde a muestras y condiciones concretas, y el propio documento la presenta en el contexto de la retención de pases.

### 4.4. Recomendación de cosecha de la FNC/Cenicafé

La **Cartilla 19: Recolección del café**, publicada por Cenicafé dentro de las cartillas cafeteras, recomienda recolectar frutos maduros para obtener mejores conversiones de café cereza a café pergamino seco. También advierte que recolectar frutos verdes y secos afecta la calidad del café cosechado y de la bebida.

Esta recomendación refuerza la importancia de considerar el estado de madurez cuando el sistema calcula rendimiento. No proporciona, por sí sola, un peso promedio universal ni un factor numérico aplicable a todas las fincas.

## 5. Ecuaciones para implementar la estimación

### 5.1. De número de cerezas a masa de café cereza

Si el modelo detecta `N` frutos y se adopta un peso promedio `P_c` expresado en gramos por fruto, la masa estimada de café cereza en kilogramos es:

\[
M_{cc} = \frac{N \times P_c}{1000}
\]

Donde:

- \(M_{cc}\): masa estimada de café cereza, en kg.
- \(N\): número de cerezas detectadas o conteo corregido.
- \(P_c\): peso promedio de una cereza, en g/fruto.
- 1000: conversión de gramos a kilogramos.

### 5.2. De masa de café cereza a pergamino seco

Si se utiliza el factor `F` reportado como kilogramos de pergamino seco por kilogramo de café cereza:

\[
M_{cps} = M_{cc} \times F
\]

Donde:

- \(M_{cps}\): masa estimada de café pergamino seco, en kg.
- \(M_{cc}\): masa estimada de café cereza, en kg.
- \(F\): factor de conversión cereza → pergamino seco.

Con los parámetros bibliográficos provisionales de 1,85 g/fruto y 0,20:

\[
M_{cps} = \frac{N \times 1,85}{1000} \times 0,20
\]

### 5.3. Ejemplo de cálculo

Supóngase que el modelo detecta 10.000 cerezas. Para ilustrar el cálculo se utilizan 1,85 g por fruto y un factor de conversión de 0,20.

**Paso 1. Masa de café cereza:**

\[
M_{cc} = \frac{10.000 \times 1,85}{1000} = 18,5\text{ kg}
\]

**Paso 2. Masa estimada de pergamino seco:**

\[
M_{cps} = 18,5 \times 0,20 = 3,70\text{ kg}
\]

**Resultado:** 10.000 cerezas equivaldrían, bajo esos parámetros, a aproximadamente 18,5 kg de café cereza y 3,70 kg de café pergamino seco.

Este resultado es una estimación matemática basada en referencias bibliográficas; no es una medición directa ni una garantía de producción.

## 6. Recomendaciones de implementación en el sistema

### 6.1. No dejar los parámetros como constantes sin contexto

Se recomienda que el peso por fruto y el factor de conversión sean configurables y estén acompañados de sus metadatos. Como mínimo, registrar:

- Valor numérico y unidad.
- Variedad a la que se refiere la referencia.
- Tipo de muestra o estado de madurez.
- Fuente bibliográfica.
- Fecha de consulta o actualización.
- Estado de validación: bibliográfico, provisional o calibrado en campo.

De esta forma, el sistema puede mostrar de dónde salen los valores y evitar que una estimación parezca una medición exacta.

### 6.2. Parámetros provisionales sugeridos

| Parámetro | Valor inicial sugerido | Fundamento | Estado recomendado |
|---|---:|---|---|
| Peso de fruto para masa sin seleccionar | 1,85 g/fruto | Montilla-Pérez et al. (2008), variedad Colombia, muestra sin seleccionar | Referencia bibliográfica provisional |
| Peso de fruto seleccionado y maduro | 1,99 g/fruto | Montilla-Pérez et al. (2008), variedad Colombia, muestra seleccionada | Referencia bibliográfica provisional |
| Factor cereza → pergamino seco | 0,20 | Montilla-Pérez et al. (2008), seleccionado y sin seleccionar | Referencia bibliográfica provisional |
| Relación cereza : pergamino seco | Aproximadamente 4,9 : 1 | Montilla-Pérez et al. (2008) | Valor de referencia, no universal |
| Corrección por oclusión | Sin valor predeterminado | Requiere comparación contra conteo manual local | Pendiente de calibración |

**Qué valor utilizar:** si el modelo identifica y cuenta principalmente cerezas maduras seleccionadas, 1,99 g es la referencia más alineada con la muestra seleccionada de 2008. Si la estimación representa una masa cosechada sin selección previa, 1,85 g es la referencia más alineada con la muestra sin seleccionar. Si el sistema no clasifica los estados de madurez, debe documentar que aplicar un solo promedio a una mezcla de estados es una simplificación.

### 6.3. Ejemplo de configuración en Python

La siguiente estructura es ilustrativa. Puede adaptarse al diseño actual del proyecto:

```python
AGRONOMIC_PARAMETERS = {
    "cherry_weight_g": {
        "value": 1.85,
        "unit": "g/fruit",
        "cultivar_reference": "Colombia",
        "sample_context": "unselected harvested mass",
        "source": "Montilla-Pérez et al. (2008), Cenicafé 59(2), 120-142",
        "status": "bibliographic_reference_not_locally_calibrated",
    },
    "cherry_to_parchment_factor": {
        "value": 0.20,
        "unit": "kg_dry_parchment/kg_cherry",
        "cultivar_reference": "Colombia",
        "source": "Montilla-Pérez et al. (2008), Cenicafé 59(2), 120-142",
        "status": "bibliographic_reference_not_locally_calibrated",
    },
}


def estimate_yield(cherry_count: int) -> dict:
    """Estimate cherry mass and dry parchment mass from a fruit count."""
    if not isinstance(cherry_count, int) or cherry_count < 0:
        raise ValueError("cherry_count must be a non-negative integer")

    cherry_weight_g = AGRONOMIC_PARAMETERS["cherry_weight_g"]["value"]
    conversion_factor = AGRONOMIC_PARAMETERS[
        "cherry_to_parchment_factor"
    ]["value"]

    cherry_mass_kg = (cherry_count * cherry_weight_g) / 1000
    parchment_mass_kg = cherry_mass_kg * conversion_factor

    return {
        "cherry_count": cherry_count,
        "cherry_mass_kg_estimated": cherry_mass_kg,
        "dry_parchment_mass_kg_estimated": parchment_mass_kg,
        "estimate_type": "indicative",
    }
```

La configuración debería residir en un archivo de parámetros o en una tabla de configuración, no necesariamente dentro del código fuente de la función. En producción, también conviene registrar qué versión de parámetros se utilizó al generar cada resultado.

### 6.4. Separar la conversión agronómica de la corrección por oclusión

El factor de conversión cereza → pergamino y el factor de oclusión cumplen funciones diferentes:

- **Factor agronómico:** transforma masa de café cereza en masa estimada de pergamino seco.
- **Corrección de conteo:** intenta compensar frutos que no fueron detectados debido a oclusiones, errores de detección u otras limitaciones visuales.

No debe presentarse el factor de oclusión como si proviniera de los documentos de Cenicafé citados en este informe. Para estimarlo se necesita una validación local.

Una forma inicial de definir una corrección empírica para una muestra de imágenes es:

\[
C_{occ} = \frac{N_{manual}}{N_{modelo}}
\]

\[
N_{corregido} = N_{modelo} \times C_{occ}
\]

Donde \(N_{manual}\) es el conteo de referencia realizado por una persona y \(N_{modelo}\) es el conteo automático sobre la misma unidad de evaluación. Esta razón corrige el conteo observado, pero por sí sola combina errores de oclusión con otros errores de detección. Para medir específicamente la oclusión sería necesario disponer de una referencia visual más completa, por ejemplo mediante varias vistas o anotación exhaustiva de los frutos.

El factor debe calcularse con varias muestras representativas y evaluarse en un conjunto distinto de imágenes. No es recomendable ajustar un factor usando las mismas imágenes con las que luego se reporta el desempeño final.

## 7. Plan propuesto de calibración local

Los estudios de Cenicafé permiten establecer valores iniciales, pero la finca o zona del proyecto necesita comprobar si esos valores representan sus condiciones. Se propone el siguiente procedimiento como plan de trabajo del proyecto; no debe confundirse con un protocolo normativo oficial.

### 7.1. Calibrar el peso promedio de la cereza

1. Identificar la variedad o variedades realmente presentes en los lotes de evaluación.
2. Recolectar una muestra representativa sin escoger únicamente los frutos más grandes o visualmente uniformes.
3. Separar los frutos por estado de madurez si el sistema los clasifica: verdes, pintones, maduros, sobremaduros y secos.
4. Pesar individualmente una cantidad suficiente de frutos por variedad y estado. Puede tomarse como referencia metodológica que el estudio de Cenicafé de 2008 midió 100 frutos por muestra, pero el tamaño final de la muestra debe definirse según la variabilidad observada y los recursos disponibles.
5. Calcular media, desviación estándar y rango del peso por fruto para cada grupo.
6. Comparar los valores locales con 1,85 g y 1,99 g, sin forzar que coincidan.

### 7.2. Calibrar el factor cereza → pergamino seco

1. Pesar una cantidad conocida de café cereza de la muestra.
2. Registrar variedad, proporción aproximada de estados de madurez y fecha de recolección.
3. Procesar la muestra con el método de beneficio utilizado en la finca.
4. Secar el pergamino hasta una condición de humedad definida y registrada. El estudio de 2008 reporta secado mecánico hasta aproximadamente 11–12 % de humedad; la comparación exige que el criterio de humedad sea consistente.
5. Pesar el café pergamino seco obtenido.
6. Calcular el factor local:

\[
F_{local} = \frac{M_{pergamino\ seco}}{M_{cereza}}
\]

7. Repetir el procedimiento en varias muestras y fechas. Reportar el promedio y la dispersión, en vez de presentar una sola prueba como un factor definitivo.

### 7.3. Validar conteo y oclusión

1. Seleccionar plantas, ramas o escenas representativas de las condiciones reales de captura.
2. Obtener imágenes con el mismo procedimiento que utilizará el sistema.
3. Realizar un conteo manual de referencia con una definición clara de qué cuenta como fruto.
4. Comparar el conteo del modelo y el conteo manual sobre la misma unidad.
5. Calcular métricas como error absoluto, error porcentual y, cuando sea posible, precisión y recall de detección.
6. Analizar los errores por iluminación, distancia, densidad de follaje, variedad y estado de madurez.
7. Reservar imágenes o plantas distintas para la evaluación final, evitando medir el desempeño sobre los mismos casos usados para calibrar.

## 8. Limitaciones que deben declararse

1. **Variedad y localidad:** los valores principales de peso y conversión proceden de estudios con variedad Colombia y condiciones experimentales o productivas concretas. No hay evidencia suficiente en estas fuentes para afirmar que representan automáticamente las variedades y condiciones exactas del lote del proyecto.
2. **Madurez:** el peso y la conversión varían entre frutos verdes, pintones, maduros, sobremaduros y secos. Un único promedio simplifica esa variabilidad.
3. **Beneficio y secado:** la conversión depende del proceso de beneficio, del manejo de la muestra y de la humedad final del pergamino.
4. **Conteo visual:** el número detectado por el modelo puede ser inferior al número real debido a frutos ocultos, iluminación, resolución, solapamientos y errores del detector.
5. **Repetición entre imágenes:** si el sistema procesa varias imágenes o fotogramas de una misma planta, debe evitar contar varias veces el mismo fruto cuando el resultado se interprete como un total único.
6. **Estimación, no pesaje:** las ecuaciones producen una masa calculada a partir de parámetros promedio; no reemplazan el pesaje de la cosecha.

Por estas razones, mientras no se realice calibración local, la interfaz y los reportes deben usar expresiones como **“rendimiento estimado”**, **“masa indicativa”** o **“estimación basada en parámetros bibliográficos”**. No se deben presentar los kilogramos calculados como una medición exacta de producción.

## 9. Conclusiones y decisiones recomendadas

1. **El peso de 1,8 g que estaba en el código tiene una referencia bibliográfica cercana, pero no idéntica.** Cenicafé reportó 1,85 g para una muestra sin seleccionar y 1,99 g para una muestra seleccionada de variedad Colombia (Montilla-Pérez et al., 2008). Se recomienda dejar de usar el valor 1,8 g sin fuente y configurar el parámetro de acuerdo con el tipo de fruto que el sistema cuenta.
2. **El factor 0,20 sí tiene respaldo bibliográfico.** Cenicafé lo reportó tanto para la muestra seleccionada como para la muestra sin seleccionar en el estudio de 2008. Debe registrarse como factor aproximado y provisional, no como una constante universal.
3. **La madurez modifica la conversión.** El estudio de Marín-López et al. (2003) encontró diferencias marcadas entre estados del fruto. Por tanto, si el modelo puede clasificar madurez, sería conveniente mantener parámetros diferenciados o, al menos, advertir que el factor único corresponde a una simplificación.
4. **La composición de la cosecha también importa.** Sanz-Uribe et al. (2018) muestran que una masa con una proporción alta de frutos sobremaduros puede presentar una relación cereza/pergamino diferente. Esto refuerza la necesidad de registrar el estado de madurez de las muestras de calibración.
5. **El factor de oclusión debe calibrarse con datos del proyecto.** No se encontró un valor de Cenicafé que pueda insertarse directamente como corrección universal. Se necesita comparar conteos manuales y automáticos en imágenes representativas.
6. **La implementación puede realizarse desde ahora como estimación indicativa.** No es necesario bloquear el funcionamiento del sistema mientras no exista calibración local, siempre que se muestren las fuentes, se identifiquen los parámetros como provisionales y se declare la limitación en la documentación técnica.
7. **La mejora experimental prioritaria** es realizar una prueba de campo que mida el peso de frutos por variedad y estado de madurez, el factor real de conversión mediante beneficio y secado controlados, y el error de conteo del modelo frente a una referencia manual.

### Decisión práctica sugerida para una primera versión

- Mantener el cálculo de rendimiento habilitado.
- Configurar 1,85 g/fruto y 0,20 como referencias iniciales cuando el sistema represente una masa cosechada sin selección, siempre que la variedad y el contexto sean razonablemente comparables con el estudio de 2008.
- Si el conteo representa exclusivamente frutos maduros seleccionados, evaluar 1,99 g/fruto como referencia alternativa.
- Marcar ambos parámetros como **“referencia bibliográfica, pendiente de calibración local”**.
- No activar un factor de oclusión inventado; mostrar que esa corrección está pendiente de validación.
- Etiquetar la salida en kilogramos como **estimación indicativa**.

## 10. Referencias bibliográficas y enlaces

1. **Montilla-Pérez, J., Arcila-Pulgarín, J., Aristizábal-Loaiza, M., Montoya-Restrepo, E. C., Puerta-Quintero, G. I., Oliveros-Tascón, C. E., & Cadena-Gómez, G. (2008).** Caracterización de algunas propiedades físicas y factores de conversión del café durante el proceso de beneficio húmedo tradicional. *Cenicafé, 59*(2), 120–142. [PDF oficial de Cenicafé](https://www.cenicafe.org/es/publications/arc059%2802%29120-142.pdf) · [Registro en la Biblioteca de Cenicafé](https://biblioteca.cenicafe.org/handle/10778/59).

2. **Marín-López, S. M., Arcila-Pulgarín, J., Montoya-Restrepo, E. C., & Oliveros-Tascón, C. E. (2003).** Relación entre el estado de madurez del fruto del café y las características de beneficio, rendimiento y calidad de la bebida. *Cenicafé, 54*(4), 297–315. [PDF oficial de Cenicafé](https://www.cenicafe.org/es/publications/arc054%2804%29297-315.pdf).

3. **Salazar-Gutiérrez, M. R., Chaves-Córdoba, B., Riaño-Herrera, N. M., Arcila-Pulgarín, J., & Jaramillo-Robledo, A. (1994).** Crecimiento del fruto de café *Coffea arabica* L. var. Colombia. *Cenicafé, 45*(2), 41–50. [PDF en la Biblioteca de Cenicafé](https://biblioteca.cenicafe.org/bitstream/10778/4275/1/arc045%2802%29041-050.pdf) · [Registro bibliográfico](https://biblioteca.cenicafe.org/handle/10778/4275).

4. **Sanz-Uribe, J. R., Oliveros-Tascón, C. E., Duque Orrego, H., Mejía, C. G., Benavides Machado, P., & Medina-Rivera, R. (2018).** Retención de pases: Una opción para mejorar la productividad de la mano de obra en la cosecha de café. *Avances Técnicos Cenicafé, 488*, 1–8. https://doi.org/10.38141/10779/0488. [Página oficial de la publicación](https://publicaciones.cenicafe.org/index.php/avances_tecnicos/article/view/228) · [PDF en la Biblioteca de Cenicafé](https://biblioteca.cenicafe.org/bitstream/10778/4218/1/AVT0488.pdf).

5. **Centro Nacional de Investigaciones de Café (Cenicafé). (2004).** *Cartilla 19: Recolección del café*. Cartillas Cafeteras. [Página oficial de la publicación](https://publicaciones.cenicafe.org/index.php/libros_manuales/article/view/3197) · [PDF](https://www.cenicafe.org/es/publications/cartilla_19_recoleccion_de_cafe.pdf).

---

## Nota final de alcance

Este documento organiza evidencia técnica consultada para sustentar parámetros iniciales del sistema. Las publicaciones citadas no constituyen una calibración específica de la finca o del lote evaluado por el proyecto. Los valores deben presentarse como referencias bibliográficas hasta que sean contrastados con muestras, variedades, procedimientos de beneficio y conteos manuales de la zona de implementación.

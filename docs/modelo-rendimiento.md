# Modelo de rendimiento

> **Estado: estimación indicativa basada en parámetros bibliográficos, no calibrada localmente.** La app y la CLI lo advierten
> en cada resultado. Parámetros: `configs/agronomia.yaml` (versión `2026-11-18`). Respaldo:
> `docs/investigacion/parametros_agronomicos_cenicafe.md` (investigación del equipo) y ADR-010.

## Fórmulas

```
kg_cereza    = cerezas × factor_oclusion × (peso_medio_fruto_g / 1000) × (plantas_totales_lote / plantas_muestra)
kg_pergamino = kg_cereza × factor_cereza_pergamino
```

Se reportan dos cifras:
- **Potencial:** todas las cerezas detectadas, suponiendo que llegan a la cosecha.
- **Listo para cosechar:** solo las cerezas MADURO y SOBREMADURO (criterio: Cenicafé, Cartilla 19, recomienda recolectar frutos maduros).

Ejemplos verificados en `tests/test_rendimiento.py`: 10 000 cerezas × 1,85 g = **18,5 kg** de cereza y × 0,20 = **3,70 kg** de pergamino
seco (el ejemplo de la investigación del equipo).

## Parámetros y su procedencia

| Factor | Tipo | Valor | Fuente | Estado |
|---|---|---|---|---|
| Peso medio de una cereza | literatura | **1,85 g** | Montilla-Pérez et al. (2008), Cenicafé 59(2):120-142, Tabla 17, café **sin seleccionar** (seleccionado: 1,99 g) | referencia bibliográfica, pendiente de calibración local |
| Cereza → pergamino seco | literatura | **0,20** kg/kg | Montilla-Pérez et al. (2008), Tabla 20 (aproximado: 1/4,94–1/4,89 = 0,202–0,205) | referencia bibliográfica, pendiente de calibración local |
| Corrección por cerezas ocultas | calibración | **1,0** (sin corrección) | no existe valor bibliográfico | pendiente de calibración con conteo manual |
| Plantas que cubre la foto | dato | 1 | decisión del equipo (ADR-011): una foto = una planta completa | fijo en la interfaz |
| Plantas del lote | dato | 2500 (ejemplo) | lo ingresa el usuario | — |

Valores verificados contra el PDF original de Cenicafé (Tablas 17, 18 y 20) el 2026-11-18.

**Por qué 1,85 g y no 1,99 g:** las fotos muestran todas las cerezas de la rama, no una cosecha seleccionada; la muestra "sin
seleccionar" es la condición más parecida. **Por qué la oclusión es 1,0:** el valor anterior (1,25) no tenía fuente y la
investigación confirma que no hay valor bibliográfico; corregir con un número inventado sería presentar una suposición como dato.

Reglas (implementadas en `src/domain/models/supuestos.py`):
- **literatura**: solo cuenta con fuente si la tiene en el YAML; si el usuario escribe otro valor se registra como "ingresado por
  el usuario" y se advierte que no tiene fuente documental (el mismo valor documentado conserva la fuente).
- **calibración**: no se "ingresa", se mide; mientras no haya calibración se advierte siempre.
- **dato**: cuenta como fuente cuando lo ingresa el usuario.

## Limitaciones declaradas (se muestran en cada resultado)

1. **Un solo peso y un solo factor para todos los estados de madurez.** La conversión cereza→pergamino cambia mucho con la
   madurez (Marín-López et al., 2003: de 5,04:1 en maduro a más de 11:1 en verde) y un fruto sobremaduro pesa ~12 % menos que uno
   maduro (Sanz-Uribe et al., 2018). No se usan factores por estado porque implicaría combinar estudios con diseños distintos.
2. **Tres contextos distintos:** parámetros de variedad Colombia en Caldas (2005); detector entrenado con fotos de Uganda (Croppie);
   uso previsto en Norte de Santander. Nada está calibrado para la zona.
3. **Oclusión sin corregir:** el conteo solo incluye cerezas visibles.
4. **Unidad de muestreo (ADR-011):** cada foto muestra una planta completa. La foto solo ve una cara de la planta, así que el conteo
   es un mínimo mientras la oclusión siga en 1,0, y se supone que esa planta representa a todo el lote. Con fotos de una rama, el kg
   del lote no es válido: úsense el conteo y la madurez.
5. **Sin validación con cosecha real** (error porcentual y sesgo desconocidos).

## Calibración local propuesta (trabajo futuro)

Ver la sección 7 de la investigación: pesar ≥ 100 frutos por variedad y estado de madurez; beneficiar y secar muestras pesadas
para obtener el factor local; comparar conteo automático con conteo manual en imágenes representativas, calibrando en unas y
evaluando en otras. Al tener los valores, basta editar `configs/agronomia.yaml` (valor, fuente, incertidumbre, estado y
`calibrado_localmente: true`); no hay que tocar código.

## Referencias (formato IEEE)

[1] J. Montilla-Pérez, J. Arcila-Pulgarín, M. Aristizábal-Loaiza, E. C. Montoya-Restrepo, G. I. Puerta-Quintero, C. E. Oliveros-Tascón y G. Cadena-Gómez, "Caracterización de algunas propiedades físicas y factores de conversión del café durante el proceso de beneficio húmedo tradicional," *Cenicafé*, vol. 59, no. 2, pp. 120–142, 2008.

[2] S. M. Marín-López, J. Arcila-Pulgarín, E. C. Montoya-Restrepo y C. E. Oliveros-Tascón, "Relación entre el estado de madurez del fruto del café y las características de beneficio, rendimiento y calidad de la bebida," *Cenicafé*, vol. 54, no. 4, pp. 297–315, 2003.

[3] M. R. Salazar-Gutiérrez, B. Chaves-Córdoba, N. M. Riaño-Herrera, J. Arcila-Pulgarín y A. Jaramillo-Robledo, "Crecimiento del fruto de café *Coffea arabica* L. var. Colombia," *Cenicafé*, vol. 45, no. 2, pp. 41–50, 1994.

[4] J. R. Sanz-Uribe, C. E. Oliveros-Tascón, H. Duque Orrego, C. G. Mejía, P. Benavides Machado y R. Medina-Rivera, "Retención de pases: Una opción para mejorar la productividad de la mano de obra en la cosecha de café," *Avances Técnicos Cenicafé*, no. 488, pp. 1–8, 2018, doi: 10.38141/10779/0488.

[5] Centro Nacional de Investigaciones de Café (Cenicafé), *Cartilla 19: Recolección del café*, Cartillas Cafeteras, 2004.

Las referencias provienen de la investigación del equipo; solo [1] se verificó contra el PDF original durante el desarrollo.

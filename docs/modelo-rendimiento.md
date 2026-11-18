# Modelo de rendimiento (versión mínima)

> Estado: **indicativo, no validado.** Convierte el conteo de una foto en kg de café cereza con factores que, a la fecha,
> **no tienen fuente documental** (`TODO(fuente)` en `configs/agronomia.yaml`). La app y la CLI lo advierten en cada resultado.

## Fórmula

```
kg_cereza    = frutos_detectados × factor_oclusion × (peso_medio_fruto_g / 1000) × (plantas_totales_lote / plantas_muestra)
kg_pergamino = kg_cereza × factor_cereza_pergamino        (solo si ese factor tiene valor y fuente)
```

Ejemplo verificado en las pruebas (`tests/test_rendimiento.py`): 10 cerezas × 1,25 × 0,0018 kg × (2500 / 1) = **56,25 kg** de cereza;
con un factor de pergamino de 0,2 (valor de prueba, no real) serían 11,25 kg.

## Factores y su procedencia

Todos viven en **`configs/agronomia.yaml`**, única fuente de los valores por defecto (la UI y la CLI los leen de ahí).

| Factor | Tipo | Valor actual | Fuente | Qué falta |
|---|---|---|---|---|
| Peso medio de una cereza (g) | literatura | 1,8 | `TODO(fuente)` (heredado del prototipo) | documento Cenicafé/FNC para la(s) variedad(es) de la zona |
| Factor de oclusión (cerezas reales / visibles) | literatura | 1,25 | `TODO(fuente)` | calibrarlo con conteo manual: regresión por el origen de conteo real vs detectado |
| Plantas que cubren las fotos | dato | 1 | lo ingresa el usuario | definir la **unidad de muestreo** (¿rama, lado de planta, planta?) |
| Plantas del lote | dato | 2500 (ejemplo) | lo ingresa el usuario | — |
| Factor cereza → pergamino seco | literatura | sin valor | `TODO(fuente)` | documento Cenicafé/FNC; mientras tanto **no se calcula pergamino** |

Reglas (implementadas en `src/domain/models/supuestos.py`):
- Un factor de **literatura** solo cuenta con fuente si la tiene en el YAML; si el usuario escribe otro valor, se registra como
  "ingresado por el usuario" y **sigue** marcado `TODO(fuente)`.
- Un **dato** del lote (plantas) cuenta como fuente cuando lo ingresa el usuario.
- Mientras `muestreo.definido: false` y `validado_con_cosecha_real: false`, el reporte lo advierte.

## Limitaciones declaradas

1. Supone que la foto cubre completas las `plantas_muestra` plantas: la unidad de muestreo **no está definida**.
2. La oclusión es una constante, no una corrección calibrada; no hay intervalo de confianza.
3. No se distingue "potencial total" de "cosechable ahora" (todas las cerezas detectadas cuentan igual).
4. Sin kg cosechados reales, el resultado **no está validado** (error porcentual y sesgo desconocidos).

## Cómo completarlo (cuando el equipo tenga las fuentes)

Editar `configs/agronomia.yaml`: poner `valor`, `fuente` (cita breve; la referencia completa en el informe) e `incertidumbre`;
cambiar `muestreo.definido` a `true` con la descripción del protocolo. No hay que tocar código.

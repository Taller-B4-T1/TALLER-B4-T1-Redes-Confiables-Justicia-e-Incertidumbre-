# `src/layers.py` — Documentación de la capa customizada

> Taller B4-T1 · *Diseño de Redes Confiables (Justicia e Incertidumbre)*
> Responsabilidad única: definir la capa customizada de Keras (**Pilar 1**). No toca datos, no entrena, no evalúa.

---

## 1. Capa custom: ratio de endeudamiento saturado

Se implementa una capa Keras 3 multi-backend, usando `keras.ops`, que calcula el ratio `AMT_CREDIT / AMT_INCOME_TOTAL` como una feature estructural del modelo.

La capa recibe un canal custom con las variables crudas, solo imputadas y sin escalar. Internamente extrae `AMT_CREDIT` y `AMT_INCOME_TOTAL` por índice y calcula:

```
AMT_CREDIT / AMT_INCOME_TOTAL
```

Este ratio mide el tamaño relativo del crédito concedido respecto a los ingresos del cliente. Dado que presenta una distribución asimétrica y con cola derecha, se aplica una saturación suave mediante:

```
tanh(ratio / k)
```

donde `k` es una constante no entrenable calibrada en el conjunto de entrenamiento, por ejemplo como el percentil 95 del ratio. Con esta parametrización, el valor `ratio = k` se transforma en `tanh(1) ≈ 0.76`, por lo que la mayor parte de la distribución conserva resolución, mientras que la cola alta queda comprimida.

Se usa `tanh` en lugar de `clip` porque es una función suave y diferenciable en todo el dominio. Esto permite limitar el efecto de valores extremos sin anular completamente el gradiente fuera de un rango prefijado.

La capa devuelve una única feature por muestra, con shape `(batch, 1)`, que después se concatena con el resto de variables antes de entrar en las capas densas.

---

## 2. Contrato de entrada (acoplado a `data.py`)

El tensor de entrada es el **canal custom** que produce `data.py`, en crudo (solo imputado, sin escalar), con orden **contractual**:

| Índice | Columna | Papel |
|---|---|---|
| 0 | `AMT_CREDIT` | numerador |
| 1 | `AMT_INCOME_TOTAL` | denominador |

Este orden coincide con `CUSTOM_COLS = ["AMT_CREDIT", "AMT_INCOME_TOTAL"]` en `data.py`. **Si se reordena allí, hay que reordenar aquí.** Es el único punto de acoplamiento entre los dos módulos.

| Entrada | Salida |
|---|---|
| `(batch, 2)` — `[credit, income]` crudo | `(batch, 1)` — ratio saturado en `[0, 1)` |

---

## 3. Interfaz pública

```python
class DebtRatioCustomLayer(keras.layers.Layer):
    def __init__(self, k: float, epsilon: float = 1e-6, **kwargs)
    def call(self, x)                          # (N, 2) -> (N, 1)
    def compute_output_shape(self, input_shape)
    def get_config(self) -> dict               # OBLIGATORIO para serialización
```

| Parámetro | Por defecto | Significado |
|---|---|---|
| `k` | — (requerido) | escala de saturación; constante no entrenable calibrada en train |
| `epsilon` | `1e-6` | término en el denominador para evitar división por cero |

`k` es **obligatorio** (sin valor por defecto) a propósito: fuerza a calibrarlo en el EDA antes de construir el modelo, en lugar de heredar un número arbitrario.

---

## 4. Decisiones de diseño

- **`k` constante, no entrenable.** La capa **no** usa `build`/`add_weight`: no tiene pesos. `k` se calibra una vez en `01_eda` (p. ej. el percentil 95 del ratio en train) y se pasa al constructor. Renunciar a un divisor entrenable es una decisión consciente: añadir ese grado de libertad rompería la interpretación económica del ratio y no se justifica en este contexto. Es el mismo criterio con el que se eligió un ratio interpretable.
- **`tanh` frente a `clip`.** `tanh` es suave y diferenciable en todo el dominio; `clip` anularía el gradiente fuera del rango y bloquearía el aprendizaje en la cola. Con `k = p95`, el `p95` se mapea a `tanh(1) ≈ 0.76` — describir `k` como "el ratio que se mapea a `tanh(1) ≈ 0.76`", no como "el punto de saturación dura", que ocurre más allá.
- **Dimensión falsa `(N, 1)`.** Cada columna se extrae como `x[:, 0:1]` (no `x[:, 0]`) para mantener la forma `(N, 1)` y evitar broadcasting accidental en la división — el mismo fallo clásico documentado para las funciones de coste.
- **`epsilon` en el denominador.** `AMT_INCOME_TOTAL` puede ser 0 o muy pequeño en algún registro; sin `epsilon` el ratio explotaría.
- **`keras.ops` puro.** Sin `tf.*` directo, para que la capa funcione en TensorFlow, PyTorch o JAX.
- **`get_config` completo y registro serializable.** `get_config` devuelve `k` y `epsilon` (todos los argumentos de `__init__`), y la clase se registra con `@keras.saving.register_keras_serializable`, de modo que `keras.models.load_model("...keras")` reconstruye la capa sin pasar `custom_objects` ni argumentos extra. Imprescindible porque los modelos se guardan en formato `.keras`.

---

## 5. Salidas → consumidas por

El tensor de salida `(batch, 1)` (ratio saturado) lo consume `src/model.py` como **rama custom** de la arquitectura funcional, que se concatena con la salida del canal denso antes de las capas densas finales.

---

## 6. Uso

```python
import numpy as np
from src.layers import DebtRatioCustomLayer

# k calibrado en 01_eda sobre el ratio en train, p. ej.:
ratio_train = X_custom_train[:, 0] / X_custom_train[:, 1]   # credit / income
k = float(np.percentile(ratio_train, 95))

layer = DebtRatioCustomLayer(k=k)
# dentro de model.py, sobre la entrada del canal custom:
# custom_feat = layer(input_custom)   # (N, 1)
```

En la construcción del modelo (`src/model.py`), la capa se aplica sobre el `Input` del canal custom y su salida se concatena con la rama densa.

---

## 7. Dependencias

`keras` (Keras 3, multi-backend vía `keras.ops`). Sin dependencia directa de TensorFlow, PyTorch o JAX en el código de la capa.

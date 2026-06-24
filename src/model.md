# `src/model.py` - Documentacion del modelo principal

> Taller B4-T1 - Diseno de Redes Confiables (Justicia e Incertidumbre)  
> Responsabilidad unica: construir y compilar el modelo principal y exponer el `HyperModel` usado por Keras Tuner. No entrena, no evalua y no guarda artefactos.

---

## 1. Vision general

`model.py` define la arquitectura principal de clasificacion binaria para riesgo de impago. El modelo sigue la API funcional de Keras y tiene dos entradas:

- `input_custom`: canal crudo con `[AMT_CREDIT, AMT_INCOME_TOTAL]`, procesado por `DebtRatioCustomLayer`.
- `input_dense`: resto de variables ya preprocesadas por `data.py`.

La salida de la capa custom, de shape `(N, 1)`, se concatena con el canal denso y pasa por una pequena red densa. El modelo se compila con la FAIR loss de `losses.py`, que usa `y_ext[:, 0]` como target real y `y_ext[:, 1]` como variable sensible.

---

## 2. Interfaz publica

### `FairCreditHyperModel`

```python
FairCreditHyperModel(
    input_shape_custom: int,
    input_shape_dense: int,
    k_ratio: float,
    fixed_hparams: dict | None = None,
)
```

| Parametro | Significado |
|---|---|
| `input_shape_custom` | numero de columnas del canal custom; normalmente `2` |
| `input_shape_dense` | numero de columnas del canal denso |
| `k_ratio` | constante calibrada para `DebtRatioCustomLayer` |
| `fixed_hparams` | si es `None`, busca arquitectura; si es `dict`, congela arquitectura y `lr` |

El metodo principal es:

```python
def build(self, hp) -> keras.Model
```

Devuelve un modelo compilado listo para que Keras Tuner o el notebook lo entrene.

---

## 3. Dos modos de busqueda

El diseno del repo exige separar arquitectura y fairness para que el frente de Pareto sea interpretable.

### Busqueda 1: arquitectura base

Se instancia con `fixed_hparams=None`. En este modo:

- varian `num_layers`, `units_i`, `dropout_i` y `lr`;
- `lambda_fair` queda fijo en `0.0`;
- el objetivo esperado del Tuner es `val_auc`.

Esto selecciona la mejor arquitectura predictiva sin mezclar el efecto de la penalizacion de equidad.

### Busqueda 2: fairness

Se instancia con `fixed_hparams` usando los mejores hiperparametros de la busqueda 1:

```python
fixed_hparams = {
    "num_layers": 1,
    "units_0": 32,
    "dropout_0": 0.2,
    "lr": 0.001,
}
```

En este modo:

- la arquitectura queda fija;
- el learning rate queda fijo;
- solo varia `lambda_fair` con valores `[0.0, 2.0, 10.0, 25.0, 50.0]`.

Cada trial genera un punto del compromiso AUC vs dependencia FAIR.

---

## 4. Metricas y loss

El modelo se compila con:

```python
loss = make_fair_loss(lambda_pearson=lambda_val)
```

Metricas:

| Metrica | Funcion |
|---|---|
| `FairAUC(name="auc")` | calcula AUC usando solo `y_ext[:, 0]`; habilita `val_auc` |
| `FairAccuracy(name="accuracy")` | accuracy sobre el target real |
| `fair_pearson_sq` | registra `Pearson(y_pred, s)^2` para analizar fairness |

`FairAUC` y `FairAccuracy` son wrappers necesarios porque Keras recibe `y_true` extendido `(TARGET, s)`, pero las metricas predictivas solo deben mirar la columna `TARGET`.

---

## 5. Entradas y salidas

### Entradas del modelo Keras

| Entrada | Shape | Origen |
|---|---:|---|
| `input_custom` | `(N, input_shape_custom)` | `X_custom_*` de `data.py` |
| `input_dense` | `(N, input_shape_dense)` | `X_dense_*` de `data.py` |

### Target de entrenamiento

El entrenamiento debe usar `y_ext`:

```python
y_ext[:, 0] = TARGET
y_ext[:, 1] = variable sensible
```

### Salida

| Salida | Shape | Significado |
|---|---:|---|
| `clasificacion_final` | `(N, 1)` | probabilidad estimada de impago |

---

## 6. Decisiones importantes

- **Objetivo principal: `val_auc`.** El dataset esta desbalanceado, por lo que accuracy no debe guiar la seleccion del Tuner.
- **Dos busquedas separadas.** Primero se elige arquitectura con `lambda_fair=0`; despues se fija arquitectura y se explora `lambda_fair`.
- **AUC wrapper.** `keras.metrics.AUC` no puede usarse directamente con `y_ext`; por eso `FairAUC` extrae la columna 0.
- **Fairness fuera de la metrica objetivo.** La penalizacion entra en la loss y se registra con `fair_pearson_sq`; la seleccion del compromiso final se hace analizando el frente de Pareto.
- **Sin `lambda_spearman`.** La FAIR loss actual solo implementa Pearson, coherente con `losses.py`.

---

## 7. Uso minimo

```python
import keras_tuner as kt
from src.model import FairCreditHyperModel

hypermodel = FairCreditHyperModel(
    input_shape_custom=X_custom_train.shape[1],
    input_shape_dense=X_dense_train.shape[1],
    k_ratio=k,
)

tuner = kt.RandomSearch(
    hypermodel,
    objective=kt.Objective("val_auc", direction="max"),
    directory="artifacts/tuner_results",
    project_name="architecture",
)
```

Para la segunda busqueda, pasar `fixed_hparams` con la arquitectura ganadora y usar otro proyecto/directorio del Tuner.

---

## 8. Dependencias

`keras`, `keras-tuner`, `src.layers`, `src.losses`.

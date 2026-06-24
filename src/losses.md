# `src/losses.py` - Documentacion de la funcion de coste FAIR

> Taller B4-T1 - Diseno de Redes Confiables (Justicia e Incertidumbre)  
> Responsabilidad unica: definir la funcion de perdida customizada de aprendizaje justo. No toca datos, no construye modelos, no entrena y no evalua.

---

## 1. Vision general

`losses.py` define la funcion de coste FAIR utilizada para entrenar un modelo de clasificacion binaria que prediga el riesgo de impago y reduzca la dependencia lineal entre la prediccion del modelo y la variable sensible `CODE_GENDER`.

La practica combina el error de clasificacion con una penalizacion por dependencia entre la prediccion y la variable sensible. En este proyecto, esa dependencia se mide con la correlacion de Pearson:

```text
Pearson(y_hat, s)
```

donde:

| Simbolo | Significado |
|---|---|
| `y_hat` | prediccion del modelo, probabilidad estimada de impago |
| `s` | variable sensible binaria derivada de `CODE_GENDER` |
| `y` | target real, `TARGET` |
| `lambda_pearson` | peso de la penalizacion Pearson |

La perdida total queda:

```text
Loss = BCE(y, y_hat)
     + lambda_pearson * Pearson(y_hat, s)^2
```

La funcion se implementa como una **factory function**: recibe hiperparametros y devuelve una loss compatible con `model.compile(loss=...)`.

---

## 2. Encaje dentro del proyecto

`losses.py` consume el formato extendido de `y_true` generado por `data.py`.

En el preprocesado, `data.py` separa:

- `y`: variable objetivo real, `TARGET`.
- `s`: variable sensible, derivada de `CODE_GENDER`.

Despues, la funcion `make_extended_y(y, s)` empaqueta ambas columnas en un unico array:

```text
y_ext[:, 0] = TARGET
y_ext[:, 1] = variable sensible s
```

Durante el entrenamiento, la loss recibe:

| Tensor | Shape | Contenido |
|---|---:|---|
| `y_true` | `(batch, 2)` | columna 0 = target real, columna 1 = sensible |
| `y_pred` | `(batch, 1)` | probabilidad predicha por el modelo |

Esta decision permite que la variable sensible viaje dentro de `y_true`, sin ser necesariamente una feature de entrada del modelo.

---

## 3. Interfaz publica

### `make_fair_loss(...)`

```python
def make_fair_loss(
    lambda_pearson: float = 1.0,
    epsilon: float = 1e-7,
) -> Callable:
```

Devuelve una funcion `fair_loss(y_true, y_pred)` compatible con Keras.

#### Parametros

| Parametro | Tipo | Por defecto | Significado |
|---|---:|---:|---|
| `lambda_pearson` | `float` | `1.0` | peso del penalty de correlacion lineal |
| `epsilon` | `float` | `1e-7` | constante de estabilidad numerica |

#### Uso tipico

```python
from src.losses import make_fair_loss

model.compile(
    optimizer=optimizer,
    loss=make_fair_loss(lambda_pearson=1.0),
    metrics=[
        keras.metrics.AUC(name="auc"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
    ],
)
```

---

## 4. Componentes de la loss

### 4.1 Binary Crossentropy

La primera parte de la loss es la perdida estandar de clasificacion binaria:

```text
BCE(y, y_hat)
```

Esta parte mide si el modelo acierta en la prediccion del target `TARGET`.

En el problema de Home Credit:

| `TARGET` | Interpretacion |
|---:|---|
| `0` | cliente pago a tiempo |
| `1` | cliente tuvo dificultades de pago |

### 4.2 Penalizacion Pearson

La segunda parte de la loss penaliza la correlacion lineal entre la prediccion y la variable sensible:

```text
Pearson(y_hat, s)^2
```

La correlacion de Pearson detecta dependencia lineal. Si la prediccion media tiende a ser sistematicamente mayor para un grupo sensible que para otro, Pearson lo captura.

La correlacion se eleva al cuadrado por dos motivos:

1. Queremos empujar la dependencia hacia `0`.
2. No queremos que el modelo sustituya una correlacion positiva por una negativa.

Es decir, tanto `Pearson(y_hat, s) = 0.8` como `Pearson(y_hat, s) = -0.8` representan dependencia fuerte entre la prediccion y la variable sensible.

---

## 5. Funciones internas

### `_as_column(x)`

Convierte un tensor a shape `(batch, 1)`.

Esto evita errores de broadcasting entre tensores de shape `(batch,)` y `(batch, 1)`.

### `pearson_corr(x, y, epsilon=1e-7)`

Calcula la correlacion de Pearson entre dos tensores.

Pasos:

1. Convierte ambos tensores a columnas.
2. Centra cada tensor restando su media.
3. Calcula el producto cruzado centrado.
4. Divide por el producto de las normas.
5. Anade `epsilon` para evitar divisiones por cero.

Formula:

```text
rho(x, y) = cov(x, y) / (std(x) * std(y))
```

### `fairness_metrics(...)`

Calcula metricas auxiliares para analisis fuera de la loss.

Devuelve un diccionario con:

```python
{
    "pearson": rho_p,
    "pearson_abs": abs(rho_p),
}
```

Es util para notebooks o scripts que comparen modelo base vs FAIR y quieran medir dependencia lineal residual.

---

## 6. Decisiones de diseno

### 6.1 La variable sensible no entra necesariamente como input

En la configuracion base del proyecto, `CODE_GENDER` no se usa como feature de entrada del modelo. Viaja como `s` dentro de `y_true` extendido.

Esto permite imponer la penalizacion:

```text
Pearson(y_hat, s)^2
```

sin que el modelo reciba directamente el genero como input.

### 6.2 AUC no entra en la loss

AUC es la metrica principal para seleccionar modelos porque el dataset esta desbalanceado. Sin embargo, AUC no se usa dentro de la loss porque no es una funcion diferenciable sencilla para entrenamiento estandar por gradiente.

Por eso:

- la loss usa `BCE`;
- el Tuner optimiza `val_auc`;
- la comparacion final reporta AUC en test.

### 6.3 El desbalanceo de clases se trata fuera de la FAIR loss

La FAIR loss no incluye internamente pesos de clase.

El desbalanceo debe tratarse con:

```python
class_weight=...
```

en `model.fit(...)`, o mediante una BCE ponderada si se decide extender el modulo.

### 6.4 Penalizacion al cuadrado

La correlacion se eleva al cuadrado para reducir su magnitud sin favorecer un cambio de signo.

### 6.5 Implementacion con `keras.ops`

El modulo usa `keras.ops` en lugar de llamadas directas a `tf.*`.

Esto mantiene la coherencia con Keras 3 multi-backend.

---

## 7. Coste computacional

La penalizacion Pearson es barata:

```text
O(batch_size)
```

No se construyen matrices pareadas ni rankings. Por tanto, la loss escala linealmente con el tamano del batch.

---

## 8. Uso con Keras Tuner

En la busqueda de fairness, `lambda_pearson` puede ser hiperparametro del Tuner.

Ejemplo:

```python
lambda_pearson = hp.Float(
    "lambda_pearson",
    min_value=0.0,
    max_value=5.0,
    step=0.25,
)
```

Luego se compila el modelo con:

```python
model.compile(
    optimizer=optimizer,
    loss=make_fair_loss(lambda_pearson=lambda_pearson),
    metrics=[
        keras.metrics.AUC(name="auc"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
    ],
)
```

Cada trial del Tuner produce un punto del compromiso:

```text
rendimiento predictivo vs dependencia FAIR
```

---

## 9. Uso en entrenamiento base y FAIR

### Modelo base

Para entrenar el modelo base sin penalizacion FAIR:

```python
base_loss = make_fair_loss(lambda_pearson=0.0)
```

En este caso, la loss queda reducida a:

```text
Loss = BCE(y, y_hat)
```

### Modelo FAIR

Para entrenar el modelo FAIR:

```python
fair_loss = make_fair_loss(lambda_pearson=1.0)
```

En este caso, el modelo minimiza simultaneamente:

1. error de clasificacion;
2. dependencia lineal con la variable sensible.

---

## 10. Metricas recomendadas para reportar

En la tabla final base vs FAIR conviene reportar:

| Metrica | Motivo |
|---|---|
| Accuracy | referencia general, aunque no debe ser la metrica principal |
| AUC | metrica principal por desbalanceo de clases |
| BCE | error de clasificacion de la loss |
| `abs(Pearson(y_hat, s))` | dependencia lineal residual |

Para la curva de Pareto, una opcion simple es usar:

```text
Eje X = abs(Pearson(y_hat, s))
Eje Y = AUC
```

---

## 11. Limitaciones

### 11.1 Dependencia por batch

La correlacion se calcula dentro de cada batch, no sobre todo el dataset completo.

Eso implica que la estimacion de dependencia puede variar algo segun:

- tamano de batch;
- composicion de cada batch;
- proporcion de la variable sensible dentro del batch.

Por eso es recomendable usar batches suficientemente grandes y, al evaluar el modelo, calcular las metricas de dependencia sobre validacion o test completos.

### 11.2 Fairness no equivale a ausencia total de sesgo

Penalizar correlacion con `CODE_GENDER` reduce una forma concreta de dependencia estadistica, pero no garantiza por si solo ausencia total de discriminacion.

Especialmente si existen proxies del genero en otras variables, puede quedar dependencia residual no capturada por Pearson.

---

## 12. Dependencias

`keras`

El modulo usa:

```python
import keras
from keras import ops
```

No depende directamente de TensorFlow, PyTorch ni JAX.

---

## 13. Resumen ejecutivo

`losses.py` implementa aprendizaje justo mediante una funcion de coste customizada.

La idea central es entrenar el modelo para que sea predictivo, pero penalizando que sus predicciones dependan linealmente de la variable sensible:

```text
Loss = BCE
     + lambda_pearson * dependencia_lineal^2
```

Pearson aporta una penalizacion simple, estable y barata computacionalmente. La penalizacion se eleva al cuadrado para empujar la dependencia hacia cero sin favorecer correlaciones negativas.

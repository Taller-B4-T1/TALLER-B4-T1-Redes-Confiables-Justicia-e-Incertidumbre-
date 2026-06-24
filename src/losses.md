# `src/losses.py` — Documentación de la función de coste FAIR

> Taller B4-T1 · *Diseño de Redes Confiables (Justicia e Incertidumbre)*  
> Responsabilidad única: definir la función de pérdida customizada de aprendizaje justo (**Pilar 2**). No toca datos, no construye modelos, no entrena y no evalúa.

---

## 1. Visión general

`losses.py` define la función de coste FAIR utilizada para entrenar un modelo de clasificación binaria que no solo prediga bien el riesgo de impago, sino que además reduzca la dependencia estadística entre la predicción del modelo y la variable sensible `CODE_GENDER`.

La práctica exige combinar el error de clasificación con una penalización por dependencia entre la predicción y la variable sensible. En este proyecto, esa dependencia se mide de dos formas complementarias:

```text
Pearson(ŷ, s)
Spearman_soft(ŷ, s)
```

donde:

| Símbolo | Significado |
|---|---|
| `ŷ` | predicción del modelo, probabilidad estimada de impago |
| `s` | variable sensible binaria derivada de `CODE_GENDER` |
| `y` | target real, `TARGET` |
| `λ_pearson` | peso de la penalización Pearson |
| `λ_spearman` | peso de la penalización Spearman |

La pérdida total queda:

```text
Loss = BCE(y, ŷ)
     + λ_pearson · Pearson(ŷ, s)^2
     + λ_spearman · Spearman_soft(ŷ, s)^2
```

La función se implementa como una **factory function**, es decir, una función que recibe hiperparámetros y devuelve una loss compatible con `model.compile(loss=...)`.

---

## 2. Encaje dentro del proyecto

`losses.py` consume el formato extendido de `y_true` generado por `data.py`.

En el preprocesado, `data.py` separa:

- `y`: variable objetivo real, `TARGET`.
- `s`: variable sensible, derivada de `CODE_GENDER`.

Después, la función `make_extended_y(y, s)` empaqueta ambas columnas en un único array:

```text
y_ext[:, 0] = TARGET
y_ext[:, 1] = variable sensible s
```

Por tanto, durante el entrenamiento, la loss recibe:

| Tensor | Shape | Contenido |
|---|---:|---|
| `y_true` | `(batch, 2)` | columna 0 = target real, columna 1 = sensible |
| `y_pred` | `(batch, 1)` | probabilidad predicha por el modelo |

Esta decisión permite que la variable sensible viaje dentro de `y_true`, sin ser necesariamente una feature de entrada del modelo. En la configuración base del proyecto, `CODE_GENDER` no entra en `X`; solo se usa en la loss para penalizar dependencia entre predicción y género.

---

## 3. Interfaz pública

### `make_fair_loss(...)`

```python
def make_fair_loss(
    lambda_pearson: float = 1.0,
    lambda_spearman: float = 1.0,
    spearman_temperature: float = 0.05,
    epsilon: float = 1e-7,
) -> Callable:
```

Devuelve una función `fair_loss(y_true, y_pred)` compatible con Keras.

#### Parámetros

| Parámetro | Tipo | Por defecto | Significado |
|---|---:|---:|---|
| `lambda_pearson` | `float` | `1.0` | peso del penalty de correlación lineal |
| `lambda_spearman` | `float` | `1.0` | peso del penalty de correlación monótona |
| `spearman_temperature` | `float` | `0.05` | suavizado usado para aproximar los rankings |
| `epsilon` | `float` | `1e-7` | constante de estabilidad numérica |

#### Uso típico

```python
from src.losses import make_fair_loss

model.compile(
    optimizer=optimizer,
    loss=make_fair_loss(
        lambda_pearson=1.0,
        lambda_spearman=0.5,
        spearman_temperature=0.05,
    ),
    metrics=[
        keras.metrics.AUC(name="auc"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
    ],
)
```

---

## 4. Componentes de la loss

### 4.1 Binary Crossentropy

La primera parte de la loss es la pérdida estándar de clasificación binaria:

```text
BCE(y, ŷ)
```

Esta parte mide si el modelo acierta en la predicción del target `TARGET`.

En el problema de Home Credit:

| `TARGET` | Interpretación |
|---:|---|
| `0` | cliente pagó a tiempo |
| `1` | cliente tuvo dificultades de pago |

La BCE fuerza al modelo a aprender el problema predictivo principal: estimar la probabilidad de impago.

---

### 4.2 Penalización Pearson

La segunda parte de la loss penaliza la correlación lineal entre la predicción y la variable sensible:

```text
Pearson(ŷ, s)^2
```

La correlación de Pearson detecta dependencia lineal. Si la predicción media tiende a ser sistemáticamente mayor para un grupo sensible que para otro, Pearson lo captura.

La correlación se eleva al cuadrado:

```text
Pearson(ŷ, s)^2
```

Esto se hace por dos motivos:

1. Queremos empujar la dependencia hacia `0`.
2. No queremos que el modelo sustituya una correlación positiva por una negativa.

Es decir, tan indeseable sería:

```text
Pearson(ŷ, s) = 0.8
```

como:

```text
Pearson(ŷ, s) = -0.8
```

En ambos casos hay dependencia fuerte entre la predicción y la variable sensible.

---

### 4.3 Penalización Spearman suave

La tercera parte de la loss penaliza la dependencia monótona entre la predicción y la variable sensible:

```text
Spearman_soft(ŷ, s)^2
```

Spearman es equivalente a aplicar Pearson sobre los rankings:

```text
Spearman(x, y) = Pearson(rank(x), rank(y))
```

La diferencia frente a Pearson es que Spearman no solo captura relaciones lineales, sino también relaciones monótonas. Por ejemplo, si al aumentar la variable sensible también tiende a aumentar la predicción, aunque la relación no sea perfectamente lineal, Spearman puede detectarlo.

El problema es que `rank()` no es diferenciable. Una red neuronal necesita gradientes para entrenar, por lo que no podemos usar el ranking duro directamente dentro de la loss.

Por eso `losses.py` implementa una aproximación diferenciable llamada `soft_rank`.

---

## 5. Soft rank

La función `soft_rank(x)` aproxima el ranking de cada elemento del batch usando comparaciones pareadas suaves.

Para cada elemento `x_i`, se calcula aproximadamente cuántos elementos del batch son menores que él:

```text
rank_i ≈ 1 + Σ_j sigmoid((x_i - x_j) / temperature)
```

Si `x_i` es mucho mayor que `x_j`, entonces:

```text
sigmoid((x_i - x_j) / temperature) ≈ 1
```

Si `x_i` es mucho menor que `x_j`, entonces:

```text
sigmoid((x_i - x_j) / temperature) ≈ 0
```

De esta forma, la suma se comporta como un ranking, pero sigue siendo diferenciable.

---

## 6. Papel de `spearman_temperature`

El parámetro `spearman_temperature` controla lo parecido que es el ranking suave al ranking real.

| Temperatura | Efecto |
|---:|---|
| Muy baja | ranking más parecido al ranking duro, pero gradientes más inestables |
| Más alta | ranking más suave, pero menos parecido al ranking real |

Valores razonables para empezar:

```python
spearman_temperature = 0.05
```

o:

```python
spearman_temperature = 0.1
```

Para este proyecto, una configuración inicial razonable es:

```python
lambda_pearson = 1.0
lambda_spearman = 0.5
spearman_temperature = 0.05
```

---

## 7. Por qué usar Pearson y Spearman

Pearson y Spearman no miden exactamente lo mismo.

| Métrica | Captura | Ventaja |
|---|---|---|
| Pearson | dependencia lineal | simple, estable, barata computacionalmente |
| Spearman | dependencia monótona | detecta relaciones ordenadas no necesariamente lineales |

Usar ambas permite penalizar dos formas distintas de dependencia entre `ŷ` y `s`.

### Ejemplo conceptual

Puede ocurrir que la relación entre predicción y género no sea perfectamente lineal, pero sí ordenada. Por ejemplo, que un grupo tienda a recibir predicciones más altas en los percentiles superiores de riesgo. En ese caso, Spearman puede aportar información adicional frente a Pearson.

Por tanto, la loss queda más robusta como penalización de dependencia:

```text
BCE
+ penalty lineal
+ penalty monótono
```

---

## 8. Funciones internas

### `_as_column(x)`

Convierte un tensor a shape `(batch, 1)`.

Esto evita errores de broadcasting entre tensores de shape `(batch,)` y `(batch, 1)`.

```python
x = _as_column(x)
```

Esta decisión es importante porque tanto `y_pred` como `s` deben operar con dimensiones compatibles.

---

### `pearson_corr(x, y, epsilon=1e-7)`

Calcula la correlación de Pearson entre dos tensores.

Pasos:

1. Convierte ambos tensores a columnas.
2. Centra cada tensor restando su media.
3. Calcula el producto cruzado centrado.
4. Divide por el producto de las normas.
5. Añade `epsilon` para evitar divisiones por cero.

Fórmula:

```text
ρ(x, y) = cov(x, y) / (std(x) · std(y))
```

Implementación conceptual:

```text
x_centered = x - mean(x)
y_centered = y - mean(y)

rho = sum(x_centered · y_centered)
      / sqrt(sum(x_centered²) · sum(y_centered²))
```

---

### `soft_rank(x, temperature=0.05)`

Calcula una aproximación diferenciable al ranking.

Pasos:

1. Convierte `x` a columna.
2. Construye una matriz de diferencias pareadas:

```text
x_i - x_j
```

3. Aplica una sigmoide suavizada:

```text
sigmoid((x_i - x_j) / temperature)
```

4. Suma las comparaciones para obtener un ranking aproximado.

Su coste computacional es aproximadamente:

```text
O(batch_size²)
```

porque compara cada muestra del batch con todas las demás.

---

### `spearman_corr_soft(x, y, temperature=0.05, epsilon=1e-7)`

Calcula Spearman aproximado como Pearson sobre soft-ranks:

```text
Spearman_soft(x, y) = Pearson(soft_rank(x), soft_rank(y))
```

Es diferenciable y, por tanto, puede entrar en la loss.

---

### `fairness_metrics(...)`

Calcula métricas auxiliares para análisis, sin usarlas necesariamente como loss.

Devuelve un diccionario con:

```python
{
    "pearson": rho_p,
    "spearman_soft": rho_s,
    "pearson_abs": abs(rho_p),
    "spearman_soft_abs": abs(rho_s),
}
```

Es útil para los notebooks, especialmente para construir:

- tabla comparativa base vs FAIR;
- curva de Pareto;
- análisis de dependencia residual por grupo.

---

## 9. Decisiones de diseño

### 9.1 La variable sensible no entra necesariamente como input

En la configuración base del proyecto, `CODE_GENDER` no se usa como feature de entrada del modelo. Viaja como `s` dentro de `y_true` extendido.

Esto permite imponer la penalización:

```text
ρ(ŷ, s)^2
```

sin que el modelo reciba directamente el género como input.

La opción `include_gender_in_X=True` sigue disponible en `data.py` para experimentar con la alternativa de incluir el género como feature y forzar independencia en la salida.

---

### 9.2 AUC no entra en la loss

AUC es la métrica principal para seleccionar modelos porque el dataset está desbalanceado. Sin embargo, AUC no se usa dentro de la loss porque no es una función diferenciable sencilla para entrenamiento estándar por gradiente.

Por eso:

- la loss usa `BCE`;
- el Tuner optimiza `val_auc`;
- la comparación final reporta AUC en test.

---

### 9.3 El desbalanceo de clases se trata fuera de la FAIR loss

La FAIR loss no incluye internamente pesos de clase.

El desbalanceo debe tratarse con:

```python
class_weight=...
```

en `model.fit(...)`, o mediante una BCE ponderada si se decide extender el módulo.

La razón es mantener separadas dos responsabilidades:

| Problema | Dónde tratarlo |
|---|---|
| Clasificación binaria | BCE |
| Desbalanceo de clases | `class_weight` o BCE ponderada |
| Dependencia con variable sensible | penalty Pearson/Spearman |

---

### 9.4 Penalización al cuadrado

Las correlaciones se elevan al cuadrado:

```text
Pearson(ŷ, s)^2
Spearman_soft(ŷ, s)^2
```

Esto evita que el modelo intente maximizar una correlación negativa. El objetivo no es cambiar el signo de la dependencia, sino reducir su magnitud.

---

### 9.5 Implementación con `keras.ops`

El módulo usa `keras.ops` en lugar de llamadas directas a `tf.*`.

Esto mantiene la coherencia con `layers.py`, donde la capa customizada también se implementa con `keras.ops` para ser compatible con Keras 3 multi-backend.

---

## 10. Coste computacional

La parte Pearson es barata:

```text
O(batch_size)
```

La parte Spearman suave es más cara:

```text
O(batch_size²)
```

porque `soft_rank` construye una matriz de diferencias entre todas las muestras del batch.

Por ejemplo:

| Batch size | Comparaciones aproximadas |
|---:|---:|
| 128 | 16.384 |
| 256 | 65.536 |
| 512 | 262.144 |
| 1024 | 1.048.576 |

Por eso conviene evitar batches excesivamente grandes si `lambda_spearman > 0`.

Una configuración práctica inicial:

```python
batch_size = 512
spearman_temperature = 0.05
```

---

## 11. Uso con Keras Tuner

En la búsqueda de fairness, `lambda_pearson` y `lambda_spearman` pueden ser hiperparámetros del Tuner.

Ejemplo:

```python
lambda_pearson = hp.Float(
    "lambda_pearson",
    min_value=0.0,
    max_value=5.0,
    step=0.25,
)

lambda_spearman = hp.Float(
    "lambda_spearman",
    min_value=0.0,
    max_value=5.0,
    step=0.25,
)
```

Luego se compila el modelo con:

```python
model.compile(
    optimizer=optimizer,
    loss=make_fair_loss(
        lambda_pearson=lambda_pearson,
        lambda_spearman=lambda_spearman,
        spearman_temperature=0.05,
    ),
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

Este compromiso se reporta después mediante una curva de Pareto.

---

## 12. Uso en entrenamiento base y FAIR

### Modelo base

Para entrenar el modelo base sin penalización FAIR:

```python
base_loss = make_fair_loss(
    lambda_pearson=0.0,
    lambda_spearman=0.0,
)
```

En este caso, la loss queda reducida a:

```text
Loss = BCE(y, ŷ)
```

---

### Modelo FAIR

Para entrenar el modelo FAIR:

```python
fair_loss = make_fair_loss(
    lambda_pearson=1.0,
    lambda_spearman=0.5,
    spearman_temperature=0.05,
)
```

En este caso, el modelo minimiza simultáneamente:

1. error de clasificación;
2. dependencia lineal con la variable sensible;
3. dependencia monótona con la variable sensible.

---

## 13. Métricas recomendadas para reportar

En la tabla final base vs FAIR conviene reportar:

| Métrica | Motivo |
|---|---|
| Accuracy | referencia general, aunque no debe ser la métrica principal |
| AUC | métrica principal por desbalanceo de clases |
| BCE | error de clasificación de la loss |
| `abs(Pearson(ŷ, s))` | dependencia lineal residual |
| `abs(Spearman_soft(ŷ, s))` | dependencia monótona residual |

Para la curva de Pareto, una opción simple es usar:

```text
Eje X = medida de dependencia FAIR
Eje Y = AUC
```

donde la dependencia FAIR puede definirse como:

```text
fairness_metric = abs(Pearson(ŷ, s)) + abs(Spearman_soft(ŷ, s))
```

o como:

```text
fairness_metric = sqrt(Pearson(ŷ, s)^2 + Spearman_soft(ŷ, s)^2)
```

---

## 14. Limitaciones

### 14.1 Spearman es aproximado

La métrica Spearman usada en la loss no es el Spearman exacto, sino una aproximación diferenciable basada en soft-ranks.

Esto es necesario para poder entrenar por descenso de gradiente.

---

### 14.2 Dependencia por batch

Las correlaciones se calculan dentro de cada batch, no sobre todo el dataset completo.

Eso implica que la estimación de dependencia puede variar algo según:

- tamaño de batch;
- composición de cada batch;
- proporción de la variable sensible dentro del batch.

Por eso es recomendable usar batches suficientemente grandes y, al evaluar el modelo, calcular las métricas de dependencia sobre validación o test completos.

---

### 14.3 Fairness no equivale a ausencia total de sesgo

Penalizar correlación con `CODE_GENDER` reduce una forma concreta de dependencia estadística, pero no garantiza por sí solo ausencia total de discriminación.

Especialmente si existen proxies del género en otras variables, puede quedar dependencia residual no capturada por Pearson o Spearman.

Por eso el resultado debe interpretarse como una auditoría parcial de equidad, no como una certificación absoluta de justicia.

---

## 15. Dependencias

`keras`

El módulo usa:

```python
import keras
from keras import ops
```

No depende directamente de TensorFlow, PyTorch ni JAX.

---

## 16. Resumen ejecutivo

`losses.py` implementa el Pilar 2 del taller: aprendizaje justo mediante una función de coste customizada.

La idea central es entrenar el modelo para que sea predictivo, pero penalizando que sus predicciones dependan de la variable sensible:

```text
Loss = BCE
     + λ_pearson · dependencia_lineal²
     + λ_spearman · dependencia_monótona²
```

Pearson aporta una penalización simple y estable de dependencia lineal. Spearman suave añade sensibilidad frente a relaciones monótonas no necesariamente lineales. Ambas penalizaciones se elevan al cuadrado para empujar la dependencia hacia cero.

El resultado es una loss diferenciable, compatible con Keras, integrable con Keras Tuner y alineada con los entregables del taller: curva de Pareto, comparación Base vs FAIR y explicación de la métrica de dependencia seleccionada.

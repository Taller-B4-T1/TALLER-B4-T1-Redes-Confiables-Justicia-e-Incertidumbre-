# `src/uncertainty.py` - Documentacion del modelo de incertidumbre

> Taller B4-T1 - Diseno de Redes Confiables (Justicia e Incertidumbre)  
> Responsabilidad unica: construir y usar el modelo secundario de incertidumbre predictiva. No carga datos, no hace splits y no decide que modelo principal usar.

---

## 1. Vision general

`uncertainty.py` implementa el Pilar 4 del proyecto: estimar donde el modelo principal es mas propenso a equivocarse. Para ello se entrena un segundo modelo supervisado con el error observado del clasificador principal.

El target de este modelo secundario es:

```text
|y_hat - y|
```

donde `y_hat` es la probabilidad predicha por el modelo principal e `y` es el target real. Se usa error absoluto porque es interpretable en clasificacion binaria probabilistica: mide distancia directa entre probabilidad estimada y etiqueta observada.

---

## 2. Interfaz publica

### `compute_errors(model, X, y) -> np.ndarray`

Calcula el error absoluto observado del modelo principal.

| Parametro | Significado |
|---|---|
| `model` | modelo principal ya entrenado, con metodo `predict` |
| `X` | entrada que espera el modelo principal; puede ser lista `[X_custom, X_dense]` |
| `y` | target simple `(N,)` o target extendido `(N, 2)` |

Salida:

```python
errors  # shape (N,)
```

Si `y` llega como `y_ext`, se usa solo `y_ext[:, 0]`. La columna sensible no participa en el error de incertidumbre.

### `build_uncertainty_model(input_dim) -> keras.Model`

Construye y compila una red densa pequena para predecir el error absoluto.

| Parametro | Significado |
|---|---|
| `input_dim` | numero de features del input concatenado |

Salida del modelo:

| Capa | Shape | Significado |
|---|---:|---|
| `estimacion_incertidumbre` | `(N, 1)` | error absoluto esperado |

La salida usa `softplus`, por lo que siempre es no negativa.

### `predict_uncertainty(uncertainty_model, X_concat) -> np.ndarray`

Devuelve la incertidumbre estimada para cada muestra, con shape `(N,)`.

---

## 3. Entradas y salidas esperadas

El modelo principal usa dos entradas (`X_custom`, `X_dense`), pero el modelo de incertidumbre recibe una matriz ya concatenada:

```python
X_concat = np.concatenate([X_custom, X_dense], axis=1)
```

| Funcion | Entrada principal | Salida |
|---|---|---|
| `compute_errors` | modelo principal + `X` + `y` | error absoluto observado `(N,)` |
| `build_uncertainty_model` | `input_dim` | modelo Keras compilado |
| `predict_uncertainty` | modelo de incertidumbre + `X_concat` | incertidumbre estimada `(N,)` |

---

## 4. Decisiones importantes

- **Error absoluto, no error cuadratico.** El repo define la supervision como `|y_hat - y|`; penalizar al cuadrado cambiaria la escala y daria mucho mas peso a errores grandes.
- **`y` robusto.** `compute_errors` acepta tanto `y` simple como `y_ext`, para evitar fallos si el notebook reutiliza los targets de entrenamiento FAIR.
- **Prediccion dentro de `compute_errors`.** La funcion recibe el modelo principal y llama a `predict`, de modo que el calculo del error queda centralizado.
- **Salida `softplus`.** La incertidumbre estimada no puede ser negativa, pero se evita el corte brusco de `ReLU`.
- **MSE como loss del modelo secundario.** Aunque el target sea error absoluto, el estimador se entrena por regresion con MSE.
- **Flags de imputacion como senal util.** Al concatenar `X_custom` y `X_dense`, el modelo de incertidumbre puede explotar los flags `is_imputed_EXT_SOURCE_*`, que suelen indicar menor fiabilidad del perfil.

---

## 5. Uso minimo

```python
import numpy as np
from src.uncertainty import (
    compute_errors,
    build_uncertainty_model,
    predict_uncertainty,
)

# Error observado del modelo principal
errors_train = compute_errors(
    model=main_model,
    X=[X_custom_train, X_dense_train],
    y=y_train_ext,
)

# Features para el modelo secundario
X_unc_train = np.concatenate([X_custom_train, X_dense_train], axis=1)
X_unc_test = np.concatenate([X_custom_test, X_dense_test], axis=1)

unc_model = build_uncertainty_model(input_dim=X_unc_train.shape[1])
unc_model.fit(X_unc_train, errors_train)

uncertainty_test = predict_uncertainty(unc_model, X_unc_test)
```

---

## 6. Dependencias

`numpy`, `keras`.

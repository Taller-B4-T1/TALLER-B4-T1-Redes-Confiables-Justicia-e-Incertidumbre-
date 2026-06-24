# `src/data.py` — Documentación del módulo de datos

> Taller B4-T1 · *Diseño de Redes Confiables (Justicia e Incertidumbre)*
> Responsabilidad única: convertir el CSV crudo en arrays listos para entrenar, aplicando el routing en dos canales, el escalado y la persistencia de artefactos. **Ningún notebook ni otro módulo lee ficheros ni hace splits.**

---

## 1. Visión general

`data.py` es el punto único de entrada a los datos del proyecto. Lee `application_train.csv` (el dataset Home Credit completo, ~122 columnas), selecciona por nombre únicamente las columnas que el modelo necesita, las limpia y transforma, y entrega los arrays separados en los **dos canales** de la arquitectura funcional de Jesús. Todo lo que se ajusta a partir de los datos (medianas de imputación, escaladores) se calcula **solo sobre el conjunto de entrenamiento** para no introducir fuga de información.

El módulo expone dos funciones de uso en notebooks (`load_and_split`, `make_extended_y`), dos funciones de persistencia (`save_preprocessing`, `save_processed`) y una interfaz de línea de comandos para ejecutar todo el preprocesado de una vez.

---

## 2. Datos de entrada

| Aspecto | Detalle |
|---|---|
| Fichero | `application_train.csv` |
| Forma | ~307.000 filas × ~122 columnas |
| Columnas usadas | `TARGET`, `CODE_GENDER`, `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_BIRTH`, `EXT_SOURCE_1/2/3` |
| Columnas ignoradas | El resto (~110). Se seleccionan por nombre; las demás no se cargan al pipeline |

La variable objetivo `TARGET` está fuertemente desbalanceada (~11:1, mayoría de buenos pagadores), lo que justifica usar AUC en lugar de *accuracy* como objetivo del Tuner aguas abajo. La variable sensible es `CODE_GENDER`.

---

## 3. Routing en dos canales

El modelo tiene **dos entradas**. `data.py` produce un array por canal:

### Canal custom — `X_custom` (shape `(N, 2)`)

Alimenta la `DebtRatioCustomLayer`. Contiene exclusivamente las dos columnas que forman el ratio de endeudamiento, **en crudo** (solo imputadas, sin escalar):

| Índice | Columna | Papel en el ratio |
|---|---|---|
| 0 | `AMT_CREDIT` | numerador |
| 1 | `AMT_INCOME_TOTAL` | denominador |

> **El orden es contractual.** `layers.py` extrae estas columnas por índice para calcular `AMT_CREDIT / AMT_INCOME_TOTAL`. No reordenar sin coordinar con `layers.py`.

**Por qué crudo y no escalado:** escalar credit o income antes de la capa rompería la interpretación del ratio (`credit / log1p(income)` no es un ratio de endeudamiento). El acotado del ratio lo realiza la saturación `tanh(ratio / k)` dentro de la capa, con `k` calibrado sobre la escala real del ratio. El ratio `AMT_CREDIT / AMT_INCOME_TOTAL` mide cuántas veces el crédito concedido supera la renta anual: vive del orden de unidades (típicamente ~1–10 en perfiles normales, con cola derecha larga). El valor exacto de `k` (p. ej. el percentil 95) se calibra en `01_eda`.

### Canal denso — `X_dense` (shape `(N, 8)`, o `(N, 10)` con género)

Recibe el resto de features ya escaladas, más los flags de imputación. Orden exacto de columnas:

| # | Columna | Tratamiento |
|---|---|---|
| 0 | `AMT_ANNUITY` | `log1p` + `RobustScaler` |
| 1 | `AGE` | `StandardScaler` (derivada de `DAYS_BIRTH`) |
| 2 | `EXT_SOURCE_1` | solo imputación (ya en ~[0,1]) |
| 3 | `EXT_SOURCE_2` | solo imputación |
| 4 | `EXT_SOURCE_3` | solo imputación |
| 5 | `is_imputed_EXT_SOURCE_1` | flag binario 0/1, sin escalar |
| 6 | `is_imputed_EXT_SOURCE_2` | flag binario 0/1, sin escalar |
| 7 | `is_imputed_EXT_SOURCE_3` | flag binario 0/1, sin escalar |
| (8, 9) | `CODE_GENDER_F`, `CODE_GENDER_M` | one-hot, **solo si** `include_gender_in_X=True` |

> **Credit e income NO se repiten en el canal denso.** Solo viven en el canal custom. `AMT_ANNUITY` es la única monetaria del canal denso.

---

## 4. Preprocesado por feature

- **`AMT_CREDIT`, `AMT_INCOME_TOTAL`** → imputación de NaN con la mediana de train; van crudas al canal custom.
- **`AMT_ANNUITY`** → `log1p` (comprime la cola alta sesgada) seguido de `RobustScaler` (centra por mediana/IQR, robusto a outliers). No se aplica clip explícito: la combinación `log1p` + `RobustScaler` ya controla los extremos. Es la única monetaria del canal denso.
- **`AGE`** → se deriva de `DAYS_BIRTH` como `años = -DAYS_BIRTH / 365.25` y se escala con `StandardScaler`.
- **`EXT_SOURCE_1/2/3`** → imputación de NaN con la mediana de train. No se escalan porque ya son puntuaciones normalizadas en ~[0,1].
- **Flags de missingness** → para cada `EXT_SOURCE` se genera `is_imputed_EXT_SOURCE_x` = 1 si el valor era NaN antes de imputar, 0 si era real. Se calculan **antes** de imputar.
- **`CODE_GENDER`** → codificada a binario para la variable sensible `s` (`M→1, F→0, XNA→0`). Por defecto **no entra como feature**; opcionalmente puede añadirse como one-hot al canal denso.

### Disciplina anti-fuga

Las medianas de imputación, el `RobustScaler` y el `StandardScaler` se **ajustan únicamente sobre las filas de entrenamiento** y luego se aplican a validación y test. Los flags de imputación se calculan sobre todo el `DataFrame` antes del split, lo cual **no es fuga**: un flag es un valor determinista por fila (`isna()`), no un estadístico agregado, y da el mismo resultado se calcule antes o después de partir.

---

## 5. Interfaz pública

### `load_and_split(path, seed=42, val_size=0.15, test_size=0.15, include_gender_in_X=False)`

Lee el CSV, imputa, escala y devuelve una tupla de **14 elementos** en este orden:

```
X_custom_train, X_custom_val, X_custom_test,   # (N, 2)  crudo: [credit, income]
X_dense_train,  X_dense_val,  X_dense_test,    # (N, 8)  escalado + flags (+ género opcional)
y_train, y_val, y_test,                        # (N,)    target
s_train, s_val, s_test,                        # (N,)    variable sensible (NO entra al modelo)
scalers,                                       # dict    objetos ajustados en train
feature_names                                  # dict    {'custom': [...], 'dense': [...]}
```

| Parámetro | Por defecto | Significado |
|---|---|---|
| `path` | — | ruta a `application_train.csv` |
| `seed` | `42` | semilla de los splits |
| `val_size` | `0.15` | fracción de validación |
| `test_size` | `0.15` | fracción de test (train queda en 0.70) |
| `include_gender_in_X` | `False` | si `True`, añade `CODE_GENDER` one-hot al canal denso |

El split es **estratificado sobre `TARGET`** (preserva la proporción de clases en los tres conjuntos) y se hace en dos etapas: primero train vs. resto, luego el resto en validación y test.

### `make_extended_y(y, s) → np.ndarray (N, 2)`

Construye el `y_true` que exige la FAIR loss: columna 0 = target real, columna 1 = variable sensible `s`. Así la variable sensible "viaja" empaquetada en el target sin ser una entrada del modelo. Se llama una vez por cada split antes de entrenar.

### `save_preprocessing(scalers, feature_names, out_dir="artifacts") → ruta`

Serializa con `joblib` un diccionario `{'scalers', 'feature_names'}` en `artifacts/preprocessing.joblib`, para reaplicar exactamente las mismas transformaciones en inferencia o en otros notebooks sin reajustar nada.

### `save_processed(splits, out_dir="artifacts") → ruta`

Guarda todos los arrays procesados (los 12 primeros elementos de la tupla de `load_and_split`) en un `.npz` comprimido en `artifacts/processed_data.npz`.

---

## 6. Contenido del dict `scalers`

| Clave | Tipo | Para qué |
|---|---|---|
| `medians` | `dict[str, float]` | medianas de imputación por columna (ajustadas en train) |
| `age_median` | `float` | mediana de `AGE` para imputar (raro, por robustez) |
| `monetary_robust` | `RobustScaler` | escalador de `AMT_ANNUITY` (ajustado sobre `log1p`) |
| `monetary_cols` | `list[str]` | columnas que pasan por el `RobustScaler` (`['AMT_ANNUITY']`) |
| `age_scaler` | `StandardScaler` | escalador de `AGE` |
| `ext_source_cols` | `list[str]` | columnas EXT_SOURCE (solo imputadas) |
| `include_gender_in_X` | `bool` | si el género se incluyó como feature |
| `gender_feature_names` | `list[str]` | nombres de las columnas one-hot de género (vacío si no se incluye) |
| `seed` | `int` | semilla usada |

---

## 7. Salidas del script

### Valores devueltos (en memoria)

La tupla de 14 elementos de `load_and_split` y el array de `make_extended_y`. Consumidos por:

- `X_custom_*`, `X_dense_*` → entradas del modelo en `02_model` y `03_uncertainty`.
- `y_*` → targets y construcción de `y_ext`.
- `s_*` → segunda columna de `y_ext` (FAIR loss) y análisis de incertidumbre por grupo.
- `scalers` → reproducir o invertir transformaciones en gráficas.
- `feature_names` → routing en `model.py` (qué columna alimenta a qué rama).

### Artefactos persistidos (en disco)

| Fichero | Generado por | Contenido |
|---|---|---|
| `artifacts/preprocessing.joblib` | `save_preprocessing` | scalers ajustados + `feature_names` |
| `artifacts/processed_data.npz` | `save_processed` | los 12 arrays de los splits |

---

## 8. Uso

### Desde un notebook

```python
from src.data import load_and_split, make_extended_y

(X_custom_train, X_custom_val, X_custom_test,
 X_dense_train,  X_dense_val,  X_dense_test,
 y_train, y_val, y_test,
 s_train, s_val, s_test,
 scalers, feature_names) = load_and_split("application_train.csv")

# y_true (N, 2) para la FAIR loss
y_train_ext = make_extended_y(y_train, s_train)
y_val_ext   = make_extended_y(y_val,   s_val)
```

### Desde la línea de comandos

```bash
# preprocesado por defecto (género fuera de X) + guardado de artefactos
python src/data.py --path application_train.csv --out artifacts

# variante: género one-hot como input del modelo
python src/data.py --path application_train.csv --out artifacts --gender-in-x
```

Imprime un resumen con las formas de cada array, la tasa de impago en train, la proporción de la variable sensible y las rutas de los artefactos guardados.

---

## 9. Decisiones de diseño relevantes

- **Ratio de endeudamiento = `AMT_CREDIT / AMT_INCOME_TOTAL`.** El canal custom enruta credit (numerador) e income (denominador); `layers.py` calcula este cociente. Mide el apalancamiento del cliente (cuántas veces su renta anual es el crédito concedido). Es coherente y único entre `data.py` y `layers.py`.
- **Variable sensible fuera de las features (por defecto).** `CODE_GENDER` no entra a ninguna `X`; solo viaja en `s` / `y_ext` para el penalty `λ·ρ(ŷ,s)²`. Es una divergencia consciente del material del curso (que recomienda darle el género al modelo y forzar independencia), priorizando la garantía dura de que el modelo nunca accede al género. El interruptor `include_gender_in_X` permite la opción contraria si se quiere reabrir esa decisión.
- **Canal custom en crudo.** Las columnas del ratio entran sin escalar para preservar la interpretación del ratio de endeudamiento; la `tanh` se encarga del acotado.
- **Flags de imputación como señal de incertidumbre.** Distinguir un `EXT_SOURCE` real de uno imputado es información clave para el modelo de incertidumbre (Pilar 4); por eso los flags se conservan como features del canal denso.
- **Escalado heterogéneo a propósito.** `RobustScaler` para la monetaria sesgada (`AMT_ANNUITY`), `StandardScaler` para la edad, nada para las puntuaciones ya normalizadas.
- **Reproducibilidad.** `SEED = 42` fijo; split estratificado; todos los ajustes solo en train.

---

## 10. Dependencias

`numpy`, `pandas`, `scikit-learn` (`RobustScaler`, `StandardScaler`, `train_test_split`), `joblib`.
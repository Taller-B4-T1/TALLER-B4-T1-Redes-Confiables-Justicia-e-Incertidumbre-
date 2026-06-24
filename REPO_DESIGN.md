# Diseño del Repositorio — Taller B4-T1
> Guía de referencia para el equipo. Documento vivo: cualquier cambio de decisión debe reflejarse aquí antes de tocar código.

---

## Estructura de directorios

```
repo/
├── src/
│   ├── data.py
│   ├── layers.py
│   ├── losses.py
│   ├── model.py
│   └── uncertainty.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model.ipynb
│   └── 03_uncertainty.ipynb
│
├── artifacts/
│   ├── preprocessing.joblib
│   ├── processed_data.npz
│   ├── base_model.keras
│   ├── fair_model.keras
│   ├── best_model.keras
│   ├── uncertainty_model.keras
│   ├── results_table.csv
│   ├── pareto_data.csv
│   └── tuner_results/
│
├── TALLER_B4_T1.ipynb
└── README.md
```

**Regla fundamental:** ningún notebook define funciones. Todo el código reutilizable vive en `src/`. Los notebooks importan, llaman, entrenan cuando corresponda y visualizan.

---

## Routing de features: dos canales de entrada

Decisión de arquitectura que condiciona a `data.py`, `layers.py` y `model.py`. El modelo tiene **dos entradas** usando la API funcional de Keras:

- **Canal custom (`X_custom`)**: contiene únicamente las dos variables financieras que forman el ratio de endeudamiento, en este orden contractual: `AMT_CREDIT`, `AMT_INCOME_TOTAL`. Entran **en crudo**, solo imputadas y sin escalar, para que la capa custom calcule un ratio económicamente interpretable: `AMT_CREDIT / AMT_INCOME_TOTAL`.
- **Canal denso (`X_dense`)**: contiene el resto de variables ya preprocesadas: `AMT_ANNUITY`, `AGE`, `EXT_SOURCE_1/2/3` y los flags de imputación de `EXT_SOURCE_1/2/3`. Opcionalmente puede incluir `CODE_GENDER` one-hot si se llama a `load_and_split(..., include_gender_in_X=True)`.

La **variable sensible (`CODE_GENDER`) no entra por defecto en ninguna `X`**. Viaja como `s` y se empaqueta en `y_ext` para que la FAIR loss pueda calcular la penalización. El interruptor `include_gender_in_X=True` permite probar la variante recomendada por el material del curso, pero la decisión base del equipo es dejarla fuera de las features. Ver disclaimer al final.

> El escalado es **heterogéneo a propósito**: las variables del ratio entran crudas al canal custom; `AMT_ANNUITY` usa `log1p` + `RobustScaler`; `AGE` usa `StandardScaler`; `EXT_SOURCE_1/2/3` no se escalan porque ya son puntuaciones normalizadas; los flags binarios no se escalan.

---

## Datos de entrada

| Aspecto | Detalle |
|---|---|
| Fichero | `application_train.csv` |
| Dataset | Home Credit, aproximadamente 307.000 filas × 122 columnas |
| Columnas usadas | `TARGET`, `CODE_GENDER`, `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_BIRTH`, `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| Columnas ignoradas | El resto. Se seleccionan explícitamente por nombre para evitar dependencias accidentales |

La variable objetivo `TARGET` está fuertemente desbalanceada, aproximadamente 11:1 a favor de buenos pagadores. Esto justifica que el objetivo principal del Tuner sea **AUC** y no `accuracy`.

---

## Contrato entre `data.py` y `layers.py`

El punto de acoplamiento más importante del repositorio es el canal custom.

| Índice en `X_custom` | Columna | Papel en la capa custom |
|---|---|---|
| 0 | `AMT_CREDIT` | numerador |
| 1 | `AMT_INCOME_TOTAL` | denominador |

La capa `DebtRatioCustomLayer` asume exactamente ese orden y calcula:

```text
ratio = AMT_CREDIT / AMT_INCOME_TOTAL
ratio_saturado = tanh(ratio / k)
```

**No reordenar `CUSTOM_COLS` en `data.py` sin cambiar `layers.py`.**

---

## Módulos de `src/`

### `src/data.py`

**Responsabilidad única:** convertir el CSV crudo en arrays listos para entrenar, aplicando selección de columnas, limpieza, imputación, split, escalado, routing en dos canales y persistencia de artefactos. Nadie más lee ficheros ni hace splits.

**Interfaz pública:**

```python
load_and_split(
    path: str,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
    include_gender_in_X: bool = False,
)
    -> X_custom_train, X_custom_val, X_custom_test,   # (N, 2): [AMT_CREDIT, AMT_INCOME_TOTAL], crudo imputado
       X_dense_train,  X_dense_val,  X_dense_test,    # (N, 8): variables densas preprocesadas (+ género opcional)
       y_train, y_val, y_test,                        # (N,): target
       s_train, s_val, s_test,                        # (N,): variable sensible
       scalers,                                       # dict con medianas, escaladores y metadatos
       feature_names                                  # dict {'custom': [...], 'dense': [...]}

make_extended_y(y: np.ndarray, s: np.ndarray) -> np.ndarray  # (N, 2)

save_preprocessing(scalers, feature_names, out_dir="artifacts") -> str

save_processed(splits, out_dir="artifacts") -> str
```

#### Salidas principales

- `X_custom_*` → entrada custom del modelo; contiene `[AMT_CREDIT, AMT_INCOME_TOTAL]` en crudo imputado.
- `X_dense_*` → entrada densa del modelo; contiene variables escaladas o imputadas según su tipo.
- `y_*` → target real.
- `s_*` → variable sensible; se usa para construir `y_ext` y para análisis por grupo.
- `make_extended_y(y, s)` → construye el `y_true` de la FAIR loss: columna 0 = target, columna 1 = sensible.
- `scalers` y `feature_names` → permiten reproducir transformaciones, depurar routing y guardar el preprocesado.

#### Decisiones fijadas de `data.py`

- `SEED = 42` en todos los splits y procesos aleatorios.
- Split estratificado sobre `TARGET` en dos etapas: train vs. resto, y luego validación vs. test.
- Todos los estadísticos agregados se ajustan **solo sobre train**: medianas de imputación, `RobustScaler` y `StandardScaler`.
- Los flags de imputación de `EXT_SOURCE` se calculan antes de imputar. No son fuga porque son información determinista fila a fila (`isna()`), no estadísticos agregados.
- `AMT_CREDIT` y `AMT_INCOME_TOTAL` van al canal custom **sin escalar** para preservar el significado del ratio.
- `AMT_ANNUITY` es la única monetaria del canal denso: `log1p` + `RobustScaler`, sin clip explícito.
- `AGE` se deriva de `DAYS_BIRTH` como `-DAYS_BIRTH / 365.25` y se escala con `StandardScaler`.
- `EXT_SOURCE_1/2/3` se imputan con mediana de train y no se escalan.
- `CODE_GENDER` se codifica como sensible binaria (`M -> 1`, `F -> 0`, `XNA -> 0`). Por defecto no entra como feature.
- `include_gender_in_X=True` añade `CODE_GENDER_F` y `CODE_GENDER_M` al canal denso para experimentar con la alternativa del curso.

#### Orden del canal denso

| # | Columna | Tratamiento |
|---|---|---|
| 0 | `AMT_ANNUITY` | `log1p` + `RobustScaler` |
| 1 | `AGE` | `StandardScaler` |
| 2 | `EXT_SOURCE_1` | imputación mediana train, sin escalar |
| 3 | `EXT_SOURCE_2` | imputación mediana train, sin escalar |
| 4 | `EXT_SOURCE_3` | imputación mediana train, sin escalar |
| 5 | `is_imputed_EXT_SOURCE_1` | flag binario, sin escalar |
| 6 | `is_imputed_EXT_SOURCE_2` | flag binario, sin escalar |
| 7 | `is_imputed_EXT_SOURCE_3` | flag binario, sin escalar |
| 8-9 | `CODE_GENDER_F`, `CODE_GENDER_M` | solo si `include_gender_in_X=True` |

#### Contenido de `scalers`

| Clave | Tipo | Para qué |
|---|---|---|
| `medians` | `dict[str, float]` | medianas de imputación ajustadas en train |
| `age_median` | `float` | mediana de `AGE` para imputar por robustez |
| `monetary_robust` | `RobustScaler` | escalador de `AMT_ANNUITY` tras `log1p` |
| `monetary_cols` | `list[str]` | columnas monetarias del canal denso: `['AMT_ANNUITY']` |
| `age_scaler` | `StandardScaler` | escalador de `AGE` |
| `ext_source_cols` | `list[str]` | columnas `EXT_SOURCE` imputadas |
| `include_gender_in_X` | `bool` | indica si el género entró como feature |
| `gender_feature_names` | `list[str]` | columnas one-hot de género, o lista vacía |
| `seed` | `int` | semilla usada |

#### Artefactos generados por `data.py`

| Fichero | Generado por | Contenido |
|---|---|---|
| `artifacts/preprocessing.joblib` | `save_preprocessing` | `scalers` ajustados + `feature_names` |
| `artifacts/processed_data.npz` | `save_processed` | los 12 arrays principales de los splits |

#### Uso desde notebook

```python
from src.data import load_and_split, make_extended_y

(X_custom_train, X_custom_val, X_custom_test,
 X_dense_train,  X_dense_val,  X_dense_test,
 y_train, y_val, y_test,
 s_train, s_val, s_test,
 scalers, feature_names) = load_and_split("application_train.csv")

y_train_ext = make_extended_y(y_train, s_train)
y_val_ext   = make_extended_y(y_val,   s_val)
y_test_ext  = make_extended_y(y_test,  s_test)
```

#### Uso desde línea de comandos

```bash
# Preprocesado por defecto: género fuera de X y guardado de artefactos
python src/data.py --path application_train.csv --out artifacts

# Variante experimental: género one-hot dentro del canal denso
python src/data.py --path application_train.csv --out artifacts --gender-in-x
```

**Consumido por:** todos los notebooks, `src/model.py` y `src/uncertainty.py`.

---

### `src/layers.py`

**Responsabilidad única:** definir la capa customizada de Keras 3, correspondiente al Pilar 1. No toca datos, no entrena y no evalúa.

**Interfaz pública:**

```python
class DebtRatioCustomLayer(keras.layers.Layer):
    def __init__(self, k: float, epsilon: float = 1e-6, **kwargs)
    def call(self, x)                          # (N, 2) -> (N, 1)
    def compute_output_shape(self, input_shape)
    def get_config(self) -> dict               # obligatorio para serialización
```

| Parámetro | Por defecto | Significado |
|---|---|---|
| `k` | requerido | escala de saturación, constante no entrenable calibrada en train |
| `epsilon` | `1e-6` | término del denominador para evitar división por cero |

La capa recibe el canal custom `(batch, 2)` con `[AMT_CREDIT, AMT_INCOME_TOTAL]` crudo imputado. Extrae cada columna manteniendo dimensión `(N, 1)` para evitar broadcasting accidental:

```text
credit = x[:, 0:1]
income = x[:, 1:2]
ratio = credit / (income + epsilon)
out = tanh(ratio / k)
```

Devuelve una única feature por muestra, con shape `(batch, 1)`, que se concatena en `model.py` con la rama densa antes de las capas finales.

#### Decisiones fijadas de `layers.py`

- **Ratio único y cerrado:** `AMT_CREDIT / AMT_INCOME_TOTAL`. Queda descartada la ambigüedad anterior `annuity/income` vs. `credit/income`.
- **Canal custom en crudo:** escalar `credit` o `income` antes de dividir rompería la interpretación económica del ratio.
- **`k` obligatorio:** no tiene valor por defecto para forzar su calibración en `01_eda`, normalmente como percentil 95 del ratio en train.
- **Interpretación correcta de `k`:** si `k = p95`, entonces `ratio = k` se transforma en `tanh(1) ≈ 0.76`. No es un punto de saturación dura; la saturación real ocurre más allá.
- **`k` constante, no entrenable:** la capa no usa `build` ni `add_weight`; añadir ese grado de libertad debilitaría la interpretación económica.
- **`tanh` frente a `clip`:** `tanh` es suave y diferenciable; `clip` anularía el gradiente fuera del rango y bloquearía aprendizaje en la cola.
- **`epsilon` en denominador:** protege frente a ingresos cero o muy pequeños.
- **`keras.ops` puro:** sin `tf.*` directo para mantener portabilidad multi-backend.
- **Serialización completa:** `get_config` devuelve `k` y `epsilon`, y la clase debe registrarse con `@keras.saving.register_keras_serializable` para que `keras.models.load_model("...keras")` funcione sin `custom_objects`.

#### Uso

```python
import numpy as np
from src.layers import DebtRatioCustomLayer

ratio_train = X_custom_train[:, 0] / X_custom_train[:, 1]
k = float(np.percentile(ratio_train, 95))

custom_feat = DebtRatioCustomLayer(k=k)(input_custom)
```

**Consumido por:** `src/model.py`.

---

### `src/losses.py`

**Responsabilidad única:** definir la función de pérdida FAIR, correspondiente al Pilar 2.

**Interfaz pública:**

```python
def make_fair_loss(lambda_: float) -> Callable[[y_true, y_pred], tensor]
```

Devuelve una función con signatura `(y_true, y_pred)` compatible con `model.compile(loss=...)`.

**Decisiones fijadas:**

- `y_true` llega con shape `(N, 2)`: columna 0 = target real, columna 1 = variable sensible `s`.
- La pérdida combina: `BCE(y_real, ŷ) + λ · ρ(ŷ, s)²`.
- El desbalanceo de clases se trata con `class_weight` en entrenamiento o BCE ponderada, no dentro del término de equidad.
- AUC no entra en la loss porque no es diferenciable. AUC es objetivo y métrica del Tuner.
- La correlación `ρ` se implementa con `keras.ops`, con `epsilon` en el denominador.
- La correlación se eleva al cuadrado para empujar la dependencia hacia 0, no hacia -1.
- `lambda_` es argumento de la función fábrica, no global; así Keras Tuner puede variarlo dentro de `build(hp)`.

**Consumido por:** `src/model.py`.

---

### `src/model.py`

**Responsabilidad única:** construir y compilar el modelo. No entrena, no evalúa y no guarda.

**Interfaz pública:**

```python
class FairCreditHyperModel(kt.HyperModel):
    def build(self, hp) -> keras.Model
```

**Decisiones fijadas:**

- Sigue el patrón `HyperModel` de Keras Tuner: `build(hp)` es el único punto de entrada.
- Arquitectura de dos entradas: canal custom + canal denso.
- La capa `DebtRatioCustomLayer(k=...)` se aplica al canal custom y su salida `(N, 1)` se concatena con la rama densa.
- La arquitectura de Jesús se usa como plantilla / espacio de búsqueda del Tuner, no como red única congelada.
- Objetivo y métrica del Tuner: `val_auc`, no `val_accuracy`.
- `build(hp)` define hiperparámetros de arquitectura y también `lambda_` (`hp.Float`) en la búsqueda de fairness.
- El modelo base se construye con `lambda_ = 0.0`.

**Estructura de las dos búsquedas, orquestadas en `02_model`:**

1. **Búsqueda de arquitectura:** Tuner explora la arquitectura con `λ = 0`, objetivo `val_auc`, y selecciona la mejor arquitectura.
2. **Búsqueda de fairness:** arquitectura fija a la ganadora; `λ` entra como `hp.Float` dentro de `build(hp)`. Cada trial produce un punto `(AUC, dependencia)`. El frente de Pareto lo genera el Tuner, no un bucle externo sobre `λ`.

**Consumido por:** `notebooks/02_model.ipynb`.

---

### `src/uncertainty.py`

**Responsabilidad única:** todo lo relativo al modelo secundario de incertidumbre, correspondiente al Pilar 4.

**Interfaz pública:**

```python
def compute_errors(model: keras.Model, X, y) -> np.ndarray
    # Devuelve |ŷ - y|, shape (N,)

def build_uncertainty_model(input_dim: int) -> keras.Model
    # Modelo con salida softplus, no negativa, y pérdida MSE

def predict_uncertainty(uncertainty_model: keras.Model, X) -> np.ndarray
    # Devuelve incertidumbre estimada, shape (N,)
```

**Decisiones fijadas:**

- La salida usa `softplus`: diferenciable y no negativa, sin corte brusco como ReLU.
- La pérdida del modelo de incertidumbre es MSE.
- El modelo de incertidumbre recibe como entrada el mismo `X` que el modelo principal, no solo `ŷ`.
- El error supervisado es `|ŷ - y|`.
- Los flags de imputación de `EXT_SOURCE` son especialmente útiles para este módulo porque permiten distinguir valores reales de valores imputados.

**Consumido por:** `notebooks/03_uncertainty.ipynb`.

---

## Notebooks

> Los notebooks son orquestadores. Importan de `src/`, ejecutan experimentos y producen entregables. Cada output marcado **[ENTREGABLE]** debe aparecer en el repo o en la presentación.

### `notebooks/01_eda.ipynb`

**Propósito:** entender los datos antes de modelar y cerrar decisiones de preprocesado.

**Inputs:** `application_train.csv`, `src/data.py` para consistencia con el pipeline.

**Estructura / contenido:**

- Distribuciones marginales y correlaciones.
- Desbalanceo de clases (~11:1), para justificar AUC sobre accuracy.
- Distribución de `AMT_CREDIT / AMT_INCOME_TOTAL` y calibración de `k` para la capa custom, por ejemplo percentil 95 en train.
- Explicación de que `ratio = k` se mapea a `tanh(1) ≈ 0.76`.
- Distribución de `AMT_ANNUITY`, justificando `log1p` + `RobustScaler`.
- Análisis de missingness en `EXT_SOURCE_1/2/3` y motivación de los flags de imputación.
- Revisión de la variable sensible `CODE_GENDER` y de la decisión de incluirla o no como input.

**Outputs:** plots exploratorios y valor elegido de `k`. No produce artefactos consumidos obligatoriamente por otros ficheros, salvo que el equipo decida guardar `k` en configuración o en `preprocessing.joblib`.

---

### `notebooks/02_model.ipynb`

**Propósito:** arquitectura óptima, aprendizaje justo, curva de Pareto y tabla comparativa. Es el camino crítico del proyecto.

**Inputs:** `src/data.py`, `src/model.py`, `src/losses.py`, `src/layers.py`.

**Estructura / pasos:**

1. Carga de datos con `load_and_split` y construcción de `y_ext` con `make_extended_y`.
2. Calibración o lectura de `k` para `DebtRatioCustomLayer`.
3. **Búsqueda de arquitectura:** Tuner explora la arquitectura con `λ = 0`, objetivo `val_auc`.
4. **Búsqueda de fairness:** arquitectura fija; `λ` como `hp.Float` dentro de `build(hp)`. Cada trial es un punto `(AUC, dependencia)`. Se vuelca a `pareto_data.csv`.
5. **Entrenamientos finales:** con la misma arquitectura, se entrena el modelo base (`λ = 0`) y el modelo FAIR (`λ*`) elegido del frente.
6. Registro de curvas de loss de ambos entrenamientos finales.
7. Tabla comparativa en test: accuracy, AUC, BCE y `ρ` Pearson con la variable sensible.

**Outputs / artefactos:**

- `artifacts/preprocessing.joblib` — scalers y nombres de features.
- `artifacts/processed_data.npz` — arrays procesados, si se decide persistirlos desde este notebook o desde CLI.
- `artifacts/best_model.keras` — arquitectura ganadora del Tuner.
- `artifacts/pareto_data.csv` — `(AUC, fairness_metric, lambda)` por trial. **[ENTREGABLE: curva de Pareto]**
- `artifacts/base_model.keras` — base (`λ = 0`), arquitectura fija.
- `artifacts/fair_model.keras` — FAIR (`λ*`), misma arquitectura.
- `artifacts/results_table.csv` — métricas base vs. FAIR en test. **[ENTREGABLE: tabla comparativa]**
- Curvas de loss de los dos entrenamientos finales. **[ENTREGABLE: curvas de convergencia]**
- `artifacts/tuner_results/` — trials del Tuner, normalmente gitignored.

---

### `notebooks/03_uncertainty.ipynb`

**Propósito:** estimar y analizar la incertidumbre del modelo final.

**Inputs:** `src/data.py`, `src/uncertainty.py`, `artifacts/best_model.keras` o el modelo final elegido (`fair_model.keras` si esa es la decisión de operación).

**Estructura / pasos:**

1. Carga de datos y modelo final.
2. `compute_errors` para generar el error supervisado `|ŷ - y|`.
3. `build_uncertainty_model` + entrenamiento con salida `softplus` y pérdida MSE.
4. `predict_uncertainty` sobre test.
5. Plot de distribución de incertidumbre por clase y, si procede, por grupo sensible.

**Outputs / artefactos:**

- `artifacts/uncertainty_model.keras`.
- Plot de distribución de incertidumbre por clase. **[ENTREGABLE: distribución de incertidumbre]**

---

## `TALLER_B4_T1.ipynb`

Notebook de presentación final. No define funciones ni reentrena modelos pesados. Carga artefactos ya generados y construye la narrativa completa del entregable.

**Consume:** `src/`, `artifacts/preprocessing.joblib`, modelos `.keras`, `pareto_data.csv`, `results_table.csv` y outputs de incertidumbre.

**Orden narrativo:**

1. Contexto del problema y variables.
2. Routing de features y disciplina anti-fuga.
3. Capa customizada: ratio `AMT_CREDIT / AMT_INCOME_TOTAL`, saturación `tanh(ratio / k)` y calibración de `k`.
4. FAIR loss: formulación y compromiso AUC/equidad.
5. Curva de Pareto desde `pareto_data.csv`. **[ENTREGABLE]**
6. Tabla comparativa base vs. mejor FAIR desde `results_table.csv`. **[ENTREGABLE]**
7. Curvas de convergencia de los entrenamientos finales. **[ENTREGABLE]**
8. Análisis de incertidumbre y plot de distribución por clase. **[ENTREGABLE]**
9. Disclaimer sobre la variable sensible y comparación con la recomendación del curso.

---

## `artifacts/`

Todos los artefactos se guardan aquí. El directorio se incluye en el repositorio, excepto subdirectorios pesados como `tuner_results/`, que deben ir a `.gitignore`.

| Fichero | Generado por | Consumido por |
|---|---|---|
| `preprocessing.joblib` | `data.py` / `02_model` | `02_model`, `03_uncertainty`, `TALLER_B4_T1`, inferencia |
| `processed_data.npz` | `data.py` / `02_model` | `02_model`, `03_uncertainty` |
| `best_model.keras` | `02_model` | `03_uncertainty`, `TALLER_B4_T1` |
| `base_model.keras` | `02_model` | `TALLER_B4_T1` |
| `fair_model.keras` | `02_model` | `TALLER_B4_T1`, opcionalmente `03_uncertainty` |
| `pareto_data.csv` | `02_model` | `TALLER_B4_T1` |
| `results_table.csv` | `02_model` | `TALLER_B4_T1` |
| `uncertainty_model.keras` | `03_uncertainty` | `TALLER_B4_T1` |
| `tuner_results/` | `02_model` | nadie directamente; gitignored |

---

## Diagrama de dependencias

```
src/data.py ────────────────────────────────────────────────────────────────┐
     │                                                                       │
     │   src/layers.py ──┐                                                   │
     │   src/losses.py ──┤──→ src/model.py ──┐                               │
     │                   │                   │                               │
     ├───────────────────┴───→ 02_model ─────┼─→ preprocessing.joblib        │
     │                                       ├─→ processed_data.npz          │
     │                                       ├─→ best_model.keras            │
     │                                       ├─→ pareto_data.csv             │
     │                                       ├─→ base_model.keras            │
     │                                       ├─→ fair_model.keras            │
     │                                       └─→ results_table.csv           │
     │                                                                       │
     │   src/uncertainty.py ──┐                                              │
     └────────────────────────┴──→ 03_uncertainty ←── modelo final .keras    │
                                        └────────→ uncertainty_model.keras    │
                                                                             │
     TALLER_B4_T1.ipynb ←──── todos los artifacts ←──────────────────────────┘
```

---

## Convenciones de coordinación

**Antes de empezar a entrenar:** `src/data.py`, `src/layers.py` y `src/model.py` deben estar cerrados y acordados por el equipo. Cualquier cambio posterior en routing, split, escalado, orden de columnas, definición del ratio o arquitectura base invalida los resultados comparativos.

**Contrato del ratio:** `X_custom[:, 0] = AMT_CREDIT` y `X_custom[:, 1] = AMT_INCOME_TOTAL`. La capa calcula `credit / income`. Esta decisión ya está cerrada.

**Seeds:** `SEED = 42` en todos los sitios donde aparezca aleatoriedad. Nunca dejar seeds sin fijar.

**Disciplina anti-fuga:** todo ajuste aprendido de datos se calcula solo en train. Esto incluye imputaciones agregadas, escaladores y calibración de cualquier hiperparámetro derivado de distribución, como `k`, salvo que se justifique explícitamente otra cosa.

**Serialización:** todos los modelos se guardan con `.keras` (formato nativo Keras 3), no con `h5`. `DebtRatioCustomLayer` debe tener `get_config` completo y registro serializable.

**Pareto generado por el Tuner:** la segunda búsqueda es una búsqueda del Tuner con `λ` como `hp.Float` dentro de `build(hp)`, con la arquitectura ya fija. No se usa un bucle externo en Python barriendo `λ`.

**Notebooks limpios:** si una celda define una función reutilizable, esa función debe moverse a `src/`.

---

## Disclaimer: tratamiento de la variable sensible

El equipo decide **excluir `CODE_GENDER` de las features de entrada (`X`) por defecto**. La variable sensible solo viaja como `s` y en `y_ext` (columna 1), y se usa exclusivamente en el término de penalización de la FAIR loss `λ · ρ(ŷ, s)²`. En la configuración base, el modelo nunca accede directamente al género como input.

Esta decisión **diverge conscientemente de la recomendación del material del curso**, que defiende incluir la variable sensible como input y forzar independencia en la salida, con el argumento de que excluirla no elimina su información porque los proxies pueden recodificarla en otras variables, y dársela explícitamente al modelo puede permitir neutralizar el sesgo con menor coste en rendimiento.

El mecanismo de la penalización `ρ(ŷ, s)²` funciona en ambos casos, porque opera sobre la correlación entre predicción y variable sensible, exista o no `s` como input. Lo que cambia es la eficiencia del frente de Pareto: dando el género al modelo, a menudo se puede alcanzar el mismo nivel de equidad con menos pérdida de AUC. El equipo prioriza la **garantía dura** de que el modelo base no usa directamente el género, asumiendo ese posible coste. La opción `include_gender_in_X=True` queda disponible para un experimento comparativo si se quiere discutir esta decisión en la presentación.

---

## Dependencias principales

- `numpy`
- `pandas`
- `scikit-learn` (`train_test_split`, `RobustScaler`, `StandardScaler`)
- `joblib`
- `keras` / Keras 3
- `keras-tuner`


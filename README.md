# Taller B4-T1 - Redes confiables: justicia e incertidumbre

Proyecto de aprendizaje automatico sobre el dataset **Home Credit Default Risk**. El objetivo es entrenar y auditar una red neuronal para estimar riesgo de impago incorporando dos requisitos de confiabilidad:

- **Justicia algorítmica**: penalizar la dependencia estadística entre la predicción y la variable sensible `CODE_GENDER`.
- **Incertidumbre predictiva**: entrenar un modelo auxiliar que estime dónde el clasificador principal es más propenso a equivocarse.

El código reutilizable vive en `src/` y los notebooks funcionan como orquestadores de análisis, entrenamiento y auditoría.

## Tabla de contenidos

- [Estructura del repositorio](#estructura-del-repositorio)
- [Dataset](#dataset)
- [Instalación](#instalación)
- [Ejecución recomendada](#ejecución-recomendada)
- [Diseño técnico](#diseño-técnico)
- [Artefactos generados](#artefactos-generados)
- [Resultados actuales](#resultados-actuales)
- [Carga de modelos](#carga-de-modelos)
- [Documentación por módulo](#documentación-por-módulo)
- [Notas de reproducibilidad](#notas-de-reproducibilidad)
- [Licencia](#licencia)

## Estructura del repositorio

```text
.
├── artifacts/
│   ├── base_model.keras
│   ├── best_model.keras
│   ├── fair_model.keras
│   ├── flowchart_modelo.png
│   ├── loss_curves.png
│   ├── pareto_data.csv
│   ├── pareto_frontier.png
│   ├── preprocessing.joblib
│   ├── processed_data.npz
│   ├── results_table.csv
│   └── tuner_results/
├── csvs/
│   └── application_train.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model.ipynb
│   └── 03_uncertainty.ipynb
├── slides/
│   └── document_pdf_merged.pdf
├── src/
│   ├── data.py
│   ├── layers.py
│   ├── losses.py
│   ├── model.py
│   └── uncertainty.py
├── requirements.txt
└── README.md
```

## Dataset

El proyecto usa `application_train.csv` del dataset **Home Credit Default Risk**.

Ruta esperada en este checkout:

```text
csvs/application_train.csv
```

Columnas principales usadas por el pipeline:

| Columna | Uso |
|---|---|
| `TARGET` | Variable objetivo: impago/dificultad de pago |
| `CODE_GENDER` | Variable sensible para auditoría FAIR |
| `AMT_CREDIT` | Numerador del ratio de endeudamiento |
| `AMT_INCOME_TOTAL` | Denominador del ratio de endeudamiento |
| `AMT_ANNUITY` | Feature monetaria del canal denso |
| `DAYS_BIRTH` | Fuente para calcular `AGE` |
| `EXT_SOURCE_1/2/3` | Scores externos predictivos |

El CSV completo es grande y puede no estar versionado en otros entornos. Si no existe, descárgalo desde Kaggle y colócalo en `csvs/application_train.csv`.

## Instalación

Requisitos principales:

- Python 3.10 o superior recomendado.
- TensorFlow/Keras para entrenar modelos.
- Jupyter o VS Code para ejecutar notebooks.

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
```

En macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
```

Dependencias declaradas en `requirements.txt`:

```text
tensorflow
keras
keras-tuner
scikit-learn
pandas
numpy
matplotlib
seaborn
scipy
```

## Ejecución recomendada

### 1. Preprocesar datos desde CLI

El módulo `src/data.py` es el punto único de entrada a datos crudos. Genera splits, scalers y arrays procesados:

```bash
python src/data.py --path csvs/application_train.csv --out artifacts
```

Salida esperada:

- `artifacts/preprocessing.joblib`
- `artifacts/processed_data.npz`

Para experimentar incluyendo `CODE_GENDER` como feature de entrada:

```bash
python src/data.py --path csvs/application_train.csv --out artifacts --gender-in-x
```

### 2. Ejecutar notebooks en orden

```bash
jupyter notebook
```

Orden recomendado:

| Notebook | Propósito |
|---|---|
| `notebooks/01_eda.ipynb` | EDA, desbalanceo, missingness, calibración de `k_ratio` |
| `notebooks/02_model.ipynb` | Entrenamiento del modelo principal, Keras Tuner, Pareto fairness/AUC |
| `notebooks/03_uncertainty.ipynb` | Modelo auxiliar de incertidumbre y auditoría final |

Nota práctica: `01_eda.ipynb` y `03_uncertainty.ipynb` usan `csvs/application_train.csv`. En `02_model.ipynb`, revisa la variable `DATA_PATH` antes de ejecutar y asegúrate de que apunte al CSV disponible en tu entorno.

## Diseño técnico

### Pipeline de datos

`src/data.py` aplica un split estratificado 70/15/15 sobre `TARGET` y evita fuga de información:

- Las medianas de imputación se ajustan solo con train.
- `RobustScaler` y `StandardScaler` se ajustan solo con train.
- Validación y test reciben transformaciones ya ajustadas.

El pipeline produce dos canales:

| Canal | Shape | Contenido |
|---|---:|---|
| `X_custom` | `(N, 2)` | `AMT_CREDIT`, `AMT_INCOME_TOTAL` crudos imputados |
| `X_dense` | `(N, 8)` | `AMT_ANNUITY`, `AGE`, `EXT_SOURCE_1/2/3`, flags de imputación |

Por defecto, `CODE_GENDER` no entra en las features. Se conserva como `s` y se empaqueta junto a `TARGET` mediante:

```python
from src.data import make_extended_y

y_ext = make_extended_y(y, s)
```

`y_ext[:, 0]` contiene `TARGET` y `y_ext[:, 1]` contiene la variable sensible.

### Capa customizada

`src/layers.py` define `DebtRatioCustomLayer`, una capa Keras serializable que calcula:

```text
ratio = AMT_CREDIT / AMT_INCOME_TOTAL
salida = tanh(ratio / k)
```

`k` se calibra en train como percentil 95 del ratio. La capa no tiene pesos entrenables: su objetivo es incorporar una feature interpretable de endeudamiento y comprimir suavemente la cola de valores extremos.

### Modelo principal

`src/model.py` define `FairCreditHyperModel`, un `keras_tuner.HyperModel` con dos modos:

| Modo | Qué busca |
|---|---|
| `fixed_hparams=None` | Arquitectura, dropout y learning rate con `lambda_fair=0` |
| `fixed_hparams=dict` | Arquitectura fija y exploración de `lambda_fair` |

La arquitectura tiene dos entradas Keras:

```text
input_custom -> DebtRatioCustomLayer
input_dense  -> features preprocesadas
concat       -> capas densas -> sigmoid
```

La métrica objetivo del tuner es `val_auc`, porque el dataset está fuertemente desbalanceado y la accuracy puede ser engañosa.

### Loss FAIR

`src/losses.py` implementa una función de pérdida diferenciable:

```text
Loss = BCE(y, y_pred)
     + lambda_pearson  * Pearson(y_pred, s)^2
     + lambda_spearman * Spearman_soft(y_pred, s)^2
```

Componentes:

- `BCE`: error de clasificación binaria.
- `Pearson`: dependencia lineal entre predicción y variable sensible.
- `Spearman_soft`: dependencia monótona aproximada con soft-ranks diferenciables.

`src/model.py` explora `lambda_fair` como peso de la penalización Pearson. La implementación de `make_fair_loss` también permite controlar `lambda_spearman` si se desea extender el experimento.

### Modelo de incertidumbre

`src/uncertainty.py` entrena un segundo modelo que predice el error absoluto observado del modelo principal:

```text
target_incertidumbre = |y_true - y_pred|
```

Flujo:

1. Se obtiene la predicción del modelo principal.
2. Se calcula el error absoluto.
3. Se concatena `X_custom` y `X_dense`.
4. Se entrena una red densa auxiliar con salida `softplus`, garantizando incertidumbre no negativa.

## Artefactos generados

| Artefacto | Generado por | Descripción |
|---|---|---|
| `artifacts/preprocessing.joblib` | `src/data.py` / notebook 02 | Scalers, medianas y nombres de features |
| `artifacts/processed_data.npz` | `src/data.py` / notebook 02 | Splits procesados para train/val/test |
| `artifacts/best_model.keras` | `02_model.ipynb` | Mejor arquitectura encontrada por Keras Tuner |
| `artifacts/base_model.keras` | `02_model.ipynb` | Modelo base de comparación |
| `artifacts/fair_model.keras` | `02_model.ipynb` | Modelo seleccionado tras análisis FAIR |
| `artifacts/pareto_data.csv` | `02_model.ipynb` | Puntos AUC vs dependencia FAIR |
| `artifacts/pareto_frontier.png` | `02_model.ipynb` | Gráfica del frente de Pareto |
| `artifacts/results_table.csv` | `02_model.ipynb` | Comparativa final base vs FAIR |
| `artifacts/loss_curves.png` | `02_model.ipynb` | Curvas de convergencia |
| `artifacts/flowchart_modelo.png` | `02_model.ipynb` | Diagrama del modelo principal |
| `artifacts/tuner_results/` | Keras Tuner | Trials de búsqueda de arquitectura y fairness |
| `artifacts/modelo_incertidumbre.keras` | `03_uncertainty.ipynb` | Modelo auxiliar de incertidumbre |

En el estado actual del repositorio también aparece `artifacts/uncertainty_model.keras` con tamaño 0 bytes. Para usar el modelo de incertidumbre, regenera el artefacto desde `notebooks/03_uncertainty.ipynb`.

## Resultados actuales

`artifacts/results_table.csv` contiene la comparativa final disponible:

| Modelo | AUC | Accuracy | rho_Pearson | rho2_Pearson |
|---|---:|---:|---:|---:|
| base (`lambda=0`) | 0.7205 | 0.9193 | 0.036869 | 0.001359 |
| fair (`lambda=0`) | 0.7054 | 0.9193 | 0.002065 | 0.000004 |

`artifacts/pareto_data.csv` recoge el trade-off de validación:

| `lambda_fair` | `val_auc` | `val_pearson_sq` |
|---:|---:|---:|
| 0.0 | 0.7235 | 0.0354 |
| 2.0 | 0.7070 | 0.0330 |
| 10.0 | 0.6214 | 0.0311 |
| 25.0 | 0.5743 | 0.0290 |
| 50.0 | 0.5522 | 0.0288 |

Estos resultados son artefactos del último entrenamiento guardado; pueden cambiar al reejecutar notebooks por inicialización aleatoria, entorno de TensorFlow o cambios en hiperparámetros.

## Carga de modelos

Antes de cargar modelos `.keras`, importa los módulos que registran capas, losses y métricas custom:

```python
import keras

import src.layers
import src.losses
import src.model

model = keras.models.load_model("artifacts/best_model.keras")
```

Para inferencia se debe aplicar el mismo preprocesado guardado en `artifacts/preprocessing.joblib`. No reajustes scalers con datos nuevos.

## Documentación por módulo

Además de este README, el repositorio incluye documentación específica:

| Archivo | Tema |
|---|---|
| `src/data.md` | Pipeline de datos, splits, escalado y persistencia |
| `src/layers.md` | Capa custom `DebtRatioCustomLayer` |
| `src/losses.md` | Loss FAIR, Pearson y Spearman suave |
| `src/model.md` | Arquitectura principal y Keras Tuner |
| `src/uncertainty.md` | Modelo secundario de incertidumbre |

## Notas de reproducibilidad

- Semilla base: `SEED = 42` en `src/data.py`.
- Split estratificado sobre `TARGET`.
- `k_ratio` debe calibrarse solo sobre train.
- `CODE_GENDER` no entra como feature por defecto; se usa como variable sensible en `y_ext`.
- El dataset está desbalanceado, por lo que `AUC` es la métrica principal.
- La ejecución completa de `02_model.ipynb` puede tardar por las búsquedas de Keras Tuner.
- Los modelos `.keras` dependen de objetos custom registrados en `src.layers`, `src.losses` y `src.model`.

## Licencia

Este proyecto está publicado bajo licencia MIT. Consulta `LICENSE` para el texto completo.

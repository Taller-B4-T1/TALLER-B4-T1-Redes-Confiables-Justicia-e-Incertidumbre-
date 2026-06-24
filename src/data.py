"""
src/data.py — Taller B4-T1 (Diseño de Redes Confiables: Justicia e Incertidumbre)

Responsabilidad única: todo lo que toca los datos crudos, desde el CSV
(`application_train.csv`, ~122 columnas) hasta los arrays listos para entrenar,
incluido el routing en DOS CANALES, el escalado y la persistencia de artefactos.
Nadie más lee ficheros ni hace splits.

CSV de origen: `application_train.csv`. El script SELECCIONA por nombre solo las
columnas que usa; ignora las ~110 columnas restantes.

Preprocesado por feature (spec acordado):
  - AMT_CREDIT       : solo al canal custom (crudo). NO entra al canal denso.
  - AMT_INCOME_TOTAL : solo al canal custom (crudo). NO entra al canal denso.
  - AMT_ANNUITY      : log1p + RobustScaler   (canal denso únicamente)
  - EXT_SOURCE_1/2/3 : imputación de NaN con la mediana de train (sin escalar;
                       ya vienen en ~[0,1]). Se generan además flags de
                       missingness ANTES de imputar.
  - AGE              : se deriva de DAYS_BIRTH (años = -DAYS_BIRTH / 365.25) y se
                       escala con StandardScaler.
  - CODE_GENDER      : variable sensible. Por defecto NO entra a X (solo viaja en
                       s_* / y_ext para la FAIR loss). Se puede incluir como
                       one-hot en el canal denso con include_gender_in_X=True
                       (ver nota al final del fichero).

Routing en dos canales (arquitectura funcional de Jesús):
  - Canal custom: alimenta la DebtRatioCustomLayer con AMT_CREDIT y
    AMT_INCOME_TOTAL EN CRUDO (imputadas, sin escalar). La capa calcula el ratio
    AMT_CREDIT / AMT_INCOME_TOTAL y lo satura con tanh. Escalarlas antes
    rompería la interpretación del ratio; el acotado lo hace la tanh.
  - Canal denso: AMT_ANNUITY, AGE, EXT_SOURCE_1/2/3, flags. Credit e income
    NO se repiten aquí; solo van al canal custom.

Sin fuga de información: medianas de imputación y todos los scalers se ajustan
SOLO sobre train y se aplican a val y test. Split estratificado sobre TARGET.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

# --------------------------------------------------------------------------- #
# Constantes de módulo
# --------------------------------------------------------------------------- #
SEED: int = 42
DAYS_PER_YEAR: float = 365.25

TARGET_COL: str = "TARGET"
SENSITIVE_COL: str = "CODE_GENDER"
AGE_SOURCE_COL: str = "DAYS_BIRTH"

# Canal custom (EN CRUDO). ORDEN CONTRACTUAL: layers.py extrae por índice ->
# 0 = numerador (credit), 1 = denominador (income).
CUSTOM_COLS: list[str] = ["AMT_CREDIT", "AMT_INCOME_TOTAL"]

# Canal denso, por tratamiento de escalado.
MONETARY_COLS: list[str] = ["AMT_ANNUITY"]  # log1p + RobustScaler (credit e income van solo al canal custom)
EXT_SOURCE_COLS: list[str] = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]  # solo imputación


# --------------------------------------------------------------------------- #
# Helpers de codificación de la variable sensible
# --------------------------------------------------------------------------- #
def _gender_binary(s_raw: pd.Series) -> np.ndarray:
    """CODE_GENDER -> binario 0/1 para s_* / y_ext (lo que necesita la FAIR loss).

    Convención: M -> 1, F -> 0, XNA -> 0 (coherente con el notebook del profesor).
    Acepta el CSV ya numérico o el original 'M'/'F'/'XNA'.
    """
    if pd.api.types.is_numeric_dtype(s_raw):
        return s_raw.fillna(0).astype(int).to_numpy()
    return s_raw.map({"M": 1, "F": 0, "XNA": 0}).fillna(0).astype(int).to_numpy()


def _gender_onehot(s_raw: pd.Series) -> tuple[np.ndarray, list[str]]:
    """CODE_GENDER -> one-hot [CODE_GENDER_F, CODE_GENDER_M]. XNA -> [0, 0].

    Solo se usa si include_gender_in_X=True (género como input del modelo).
    """
    if pd.api.types.is_numeric_dtype(s_raw):
        g = s_raw.map({1: "M", 0: "F"}).astype(str)
    else:
        g = s_raw.astype(str)
    f = (g == "F").astype(np.float64).to_numpy()
    m = (g == "M").astype(np.float64).to_numpy()
    return np.stack([f, m], axis=1), ["CODE_GENDER_F", "CODE_GENDER_M"]


# --------------------------------------------------------------------------- #
# Interfaz pública
# --------------------------------------------------------------------------- #
def load_and_split(
    path: str,
    seed: int = SEED,
    val_size: float = 0.15,
    test_size: float = 0.15,
    include_gender_in_X: bool = False,
):
    """Lee `application_train.csv`, imputa, escala y devuelve los arrays de los dos canales.

    Returns (en este orden):
      X_custom_train, X_custom_val, X_custom_test : (N, 2) EN CRUDO  [credit, income]
      X_dense_train,  X_dense_val,  X_dense_test  : (N, D) escalado + flags (+ género opc.)
      y_train, y_val, y_test : (N,)
      s_train, s_val, s_test : (N,)  variable sensible — NO entra al modelo
      scalers       : dict con todo lo ajustado en train (para reproducir/invertir)
      feature_names : {'custom': [...], 'dense': [...]} en el orden exacto de columnas
    """
    df = pd.read_csv(path)

    # --- Targets y variable sensible ------------------------------------- #
    y = df[TARGET_COL].astype(int).to_numpy()
    s = _gender_binary(df[SENSITIVE_COL])

    # --- Flags de missingness de EXT_SOURCE (ANTES de imputar) ----------- #
    # Per-fila y deterministas: no inducen fuga. Preservan la señal de
    # fiabilidad del perfil para el modelo de incertidumbre (Pilar 4).
    flags = pd.DataFrame(
        {f"is_imputed_{c}": df[c].isna().astype(int) for c in EXT_SOURCE_COLS},
        index=df.index,
    )
    flag_cols = list(flags.columns)

    # --- Columnas de entrada que tocamos + AGE derivada ------------------ #
    feature_cols = sorted(set(CUSTOM_COLS) | set(MONETARY_COLS) | set(EXT_SOURCE_COLS))
    Xi = df[feature_cols].copy()
    Xi["AGE"] = (-df[AGE_SOURCE_COL].to_numpy(dtype=np.float64)) / DAYS_PER_YEAR

    if include_gender_in_X:
        g_oh, g_names = _gender_onehot(df[SENSITIVE_COL])
    else:
        g_oh, g_names = None, []

    # --- Split estratificado 70 / 15 / 15 (nada ajustado todavía) -------- #
    idx = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=(val_size + test_size), random_state=seed, stratify=y
    )
    rel_test = test_size / (val_size + test_size)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=rel_test, random_state=seed, stratify=y[idx_tmp]
    )

    # --- Imputación: medianas ajustadas SOLO en train -------------------- #
    medians = {c: float(np.nanmedian(Xi[c].to_numpy()[idx_train])) for c in feature_cols}
    for c in feature_cols:
        Xi[c] = Xi[c].fillna(medians[c])
    age_median = float(np.nanmedian(Xi["AGE"].to_numpy()[idx_train]))
    Xi["AGE"] = Xi["AGE"].fillna(age_median)

    # --- Canal custom: CRUDO (imputado, sin escalar) --------------------- #
    Xc = Xi[CUSTOM_COLS].to_numpy(dtype=np.float64)  # [credit, income]

    # --- Canal denso ----------------------------------------------------- #
    # Monetaria: log1p + RobustScaler (RobustScaler ajustado solo en train).
    mono_log = np.log1p(Xi[MONETARY_COLS].to_numpy(dtype=np.float64))
    robust = RobustScaler().fit(mono_log[idx_train])
    mono_scaled = robust.transform(mono_log)

    # AGE: StandardScaler (ajustado solo en train).
    age_arr = Xi["AGE"].to_numpy(dtype=np.float64).reshape(-1, 1)
    age_scaler = StandardScaler().fit(age_arr[idx_train])
    age_scaled = age_scaler.transform(age_arr)

    # EXT_SOURCE: solo imputado (ya en ~[0,1]).
    ext_arr = Xi[EXT_SOURCE_COLS].to_numpy(dtype=np.float64)

    # Flags de imputación (0/1, sin escalar).
    flag_arr = flags.to_numpy(dtype=np.float64)

    dense_parts = [mono_scaled, age_scaled, ext_arr, flag_arr]
    dense_names = list(MONETARY_COLS) + ["AGE"] + list(EXT_SOURCE_COLS) + flag_cols
    if include_gender_in_X:
        dense_parts.append(g_oh)
        dense_names += g_names
    Xd = np.concatenate(dense_parts, axis=1)

    # --- Slicing por split ----------------------------------------------- #
    def sl(a):
        return a[idx_train], a[idx_val], a[idx_test]

    X_custom_train, X_custom_val, X_custom_test = sl(Xc)
    X_dense_train, X_dense_val, X_dense_test = sl(Xd)
    y_train, y_val, y_test = sl(y)
    s_train, s_val, s_test = sl(s)

    scalers = {
        "medians": medians,
        "age_median": age_median,
        "monetary_robust": robust,        # RobustScaler ajustado (sobre log1p de annuity)
        "monetary_cols": list(MONETARY_COLS),
        "age_scaler": age_scaler,
        "ext_source_cols": list(EXT_SOURCE_COLS),
        "include_gender_in_X": include_gender_in_X,
        "gender_feature_names": g_names,
        "seed": seed,
    }
    feature_names = {"custom": list(CUSTOM_COLS), "dense": dense_names}

    return (
        X_custom_train, X_custom_val, X_custom_test,
        X_dense_train, X_dense_val, X_dense_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test,
        scalers,
        feature_names,
    )


def make_extended_y(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """y_true de shape (N, 2) que exige la FAIR loss. Col 0 = target, col 1 = s."""
    y = np.asarray(y).reshape(-1)
    s = np.asarray(s).reshape(-1)
    y_ext = np.zeros((len(y), 2), dtype=np.float32)
    y_ext[:, 0] = y
    y_ext[:, 1] = s
    return y_ext


# --------------------------------------------------------------------------- #
# Persistencia de artefactos
# --------------------------------------------------------------------------- #
def save_preprocessing(scalers: dict, feature_names: dict, out_dir: str = "artifacts") -> str:
    """Guarda los scalers + feature_names para reusarlos en notebooks / inferencia."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "preprocessing.joblib")
    joblib.dump({"scalers": scalers, "feature_names": feature_names}, path)
    return path


def save_processed(splits: tuple, out_dir: str = "artifacts") -> str:
    """Guarda todos los arrays ya procesados en un .npz comprimido."""
    os.makedirs(out_dir, exist_ok=True)
    (Xc_tr, Xc_va, Xc_te, Xd_tr, Xd_va, Xd_te,
     y_tr, y_va, y_te, s_tr, s_va, s_te) = splits
    path = os.path.join(out_dir, "processed_data.npz")
    np.savez_compressed(
        path,
        X_custom_train=Xc_tr, X_custom_val=Xc_va, X_custom_test=Xc_te,
        X_dense_train=Xd_tr, X_dense_val=Xd_va, X_dense_test=Xd_te,
        y_train=y_tr, y_val=y_va, y_test=y_te,
        s_train=s_tr, s_val=s_va, s_test=s_te,
    )
    return path


# --------------------------------------------------------------------------- #
# CLI: python src/data.py --path application_train.csv --out artifacts
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesado Taller B4-T1")
    parser.add_argument("--path", default="application_train.csv", help="ruta al CSV de origen")
    parser.add_argument("--out", default="artifacts", help="directorio de artefactos")
    parser.add_argument("--gender-in-x", action="store_true",
                        help="incluir CODE_GENDER one-hot como input del modelo")
    args = parser.parse_args()

    out = load_and_split(args.path, include_gender_in_X=args.gender_in_x)
    *splits, scalers, feature_names = out

    pp_path = save_preprocessing(scalers, feature_names, args.out)
    dd_path = save_processed(tuple(splits), args.out)

    Xc_tr, Xc_va, Xc_te, Xd_tr, Xd_va, Xd_te, y_tr, y_va, y_te, s_tr, s_va, s_te = splits
    print("=== Preprocesado completado ===")
    print(f"  X_custom: train {Xc_tr.shape}  val {Xc_va.shape}  test {Xc_te.shape}")
    print(f"  X_dense : train {Xd_tr.shape}  val {Xd_va.shape}  test {Xd_te.shape}")
    print(f"  y       : train {y_tr.shape}  | tasa de impago train = {y_tr.mean():.4f}")
    print(f"  s        : train {s_tr.shape}  | proporción M (s=1) train = {s_tr.mean():.4f}")
    print(f"  features custom: {feature_names['custom']}")
    print(f"  features dense : {feature_names['dense']}")
    print(f"  guardado -> {pp_path}")
    print(f"  guardado -> {dd_path}")
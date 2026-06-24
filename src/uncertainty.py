"""
src/uncertainty.py - Taller B4-T1 (Diseno de Redes Confiables: Justicia e Incertidumbre)

Responsabilidad unica: todo lo relativo al modelo secundario de incertidumbre
(Pilar 4). Estima la magnitud esperada del error absoluto del modelo principal.
"""

from __future__ import annotations

import numpy as np
import keras


def _target_from_y(y: np.ndarray) -> np.ndarray:
    """Extrae TARGET desde y simple o desde y_ext[:, 0]."""
    y_arr = np.asarray(y)
    if y_arr.ndim == 2:
        y_arr = y_arr[:, 0]
    return y_arr.reshape(-1)


def compute_errors(model: keras.Model, X, y: np.ndarray) -> np.ndarray:
    """
    Calcula el error absoluto entre la prediccion del modelo y la etiqueta real.

    `y` puede ser el target simple `(N,)` o el tensor extendido `(N, 2)` generado
    por `src.data.make_extended_y`; en este ultimo caso se usa solo la columna 0.
    """
    y_true = _target_from_y(y)
    y_pred = np.asarray(model.predict(X, verbose=0)).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y y y_pred deben tener el mismo numero de muestras: "
            f"{y_true.shape[0]} != {y_pred.shape[0]}"
        )

    return np.abs(y_true - y_pred)


def build_uncertainty_model(input_dim: int) -> keras.Model:
    """
    Construye la arquitectura del estimador de incertidumbre.

    Usa activacion softplus para garantizar una salida no negativa y se optimiza
    mediante MSE contra el error absoluto observado del modelo principal.

    Nota: input_dim debe corresponder a la suma de caracteristicas (Custom + Dense).
    """
    inputs = keras.Input(shape=(input_dim,), name="input_features_unc")

    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation="relu")(x)

    outputs = keras.layers.Dense(1, activation="softplus", name="estimacion_incertidumbre")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def predict_uncertainty(uncertainty_model: keras.Model, X_concat: np.ndarray) -> np.ndarray:
    """
    Infiere la incertidumbre esperada para los perfiles procesados.

    X_concat representa la concatenacion de X_custom y X_dense (axis=1).
    """
    return uncertainty_model.predict(X_concat, verbose=0).flatten()

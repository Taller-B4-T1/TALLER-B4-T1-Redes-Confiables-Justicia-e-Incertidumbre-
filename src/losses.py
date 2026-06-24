"""
src/losses.py

Funciones de pérdida FAIR para el Taller B4-T1.

La loss principal combina:
    BCE(y, y_pred)
    + lambda_pearson  * Pearson(y_pred, s)^2

Contrato de entrada:
    y_true[:, 0] = TARGET real
    y_true[:, 1] = variable sensible s

Notas:
    - Pearson captura dependencia lineal.
"""

from __future__ import annotations

from typing import Callable

import keras
from keras import ops


_EPSILON = 1e-7


def _as_column(x):
    """
    Convierte un tensor a shape (batch, 1).

    Evita problemas de broadcasting accidental entre tensores de shape
    (batch,) y (batch, 1).
    """
    x = ops.cast(x, keras.backend.floatx())

    if len(x.shape) == 1:
        return ops.reshape(x, (-1, 1))

    return x


def pearson_corr(x, y, epsilon: float = _EPSILON):
    """
    Correlación de Pearson diferenciable entre dos tensores.

    Parameters
    ----------
    x, y:
        Tensores de shape (batch,) o (batch, 1).
    epsilon:
        Constante de estabilidad numérica.

    Returns
    -------
    Tensor escalar:
        Correlación de Pearson en el batch.
    """
    x = _as_column(x)
    y = _as_column(y)

    x_centered = x - ops.mean(x, axis=0, keepdims=True)
    y_centered = y - ops.mean(y, axis=0, keepdims=True)

    numerator = ops.sum(x_centered * y_centered)

    x_norm = ops.sqrt(ops.sum(ops.square(x_centered)) + epsilon)
    y_norm = ops.sqrt(ops.sum(ops.square(y_centered)) + epsilon)

    return numerator / (x_norm * y_norm + epsilon)


@keras.saving.register_keras_serializable(package="b4t1")
def make_fair_loss(
    lambda_pearson: float = 1.0,
    epsilon: float = _EPSILON,
) -> Callable:
    """
    Crea una FAIR loss compatible con model.compile.

    La pérdida total es:

        BCE(y, y_pred)
        + lambda_pearson  * Pearson(y_pred, s)^2

    Parameters
    ----------
    lambda_pearson:
        Peso del penalty de correlación lineal.
    epsilon:
        Constante de estabilidad numérica.

    Returns
    -------
    Callable:
        Función loss `(y_true, y_pred) -> scalar`.
    """

    def fair_loss(y_true, y_pred):
        y_true = ops.cast(y_true, keras.backend.floatx())
        y_pred = ops.cast(y_pred, keras.backend.floatx())

        y_real = y_true[:, 0:1]
        s = y_true[:, 1:2]

        y_pred = _as_column(y_pred)

        bce = keras.losses.binary_crossentropy(y_real, y_pred)
        bce = ops.mean(bce)

        rho_pearson = pearson_corr(
            y_pred,
            s,
            epsilon=epsilon,
        )

        fairness_penalty = lambda_pearson * ops.square(rho_pearson)

        return bce + fairness_penalty

    return fair_loss


def fairness_metrics(
    y_true,
    y_pred,
    epsilon: float = _EPSILON,
) -> dict:
    """
    Calcula métricas auxiliares de dependencia para análisis fuera de la loss.

    Útil en notebooks para comparar modelo base vs modelo FAIR.
    """
    y_true = ops.cast(y_true, keras.backend.floatx())
    y_pred = ops.cast(y_pred, keras.backend.floatx())

    s = y_true[:, 1:2]
    y_pred = _as_column(y_pred)

    rho_p = pearson_corr(y_pred, s, epsilon=epsilon)

    return {
        "pearson": rho_p,
        "pearson_abs": ops.abs(rho_p),
    }

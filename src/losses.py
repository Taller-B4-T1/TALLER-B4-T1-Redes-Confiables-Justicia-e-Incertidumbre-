"""
src/losses.py

Funciones de pérdida FAIR para el Taller B4-T1.

La loss principal combina:
    BCE(y, y_pred)
    + lambda_pearson  * Pearson(y_pred, s)^2
    + lambda_spearman * SoftSpearman(y_pred, s)^2

Contrato de entrada:
    y_true[:, 0] = TARGET real
    y_true[:, 1] = variable sensible s

Notas:
    - Pearson captura dependencia lineal.
    - Spearman captura dependencia monótona, usando rangos.
    - Como los rangos exactos no son diferenciables, Spearman se aproxima
      mediante soft-ranks con sigmoides pareadas.
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


def soft_rank(x, temperature: float = 0.05):
    """
    Aproximación diferenciable del ranking.

    Para cada elemento x_i, estima cuántos elementos son menores que él:

        rank_i ≈ 1 + sum_j sigmoid((x_i - x_j) / temperature)

    Cuanto menor sea `temperature`, más se parece al ranking duro.
    Cuanto mayor sea, más suave y estable es el gradiente.

    Parameters
    ----------
    x:
        Tensor de shape (batch,) o (batch, 1).
    temperature:
        Temperatura de suavizado. Valores típicos: 0.01, 0.05, 0.1.

    Returns
    -------
    Tensor de shape (batch, 1):
        Rangos suaves.
    """
    x = _as_column(x)

    pairwise_diff = x - ops.transpose(x)
    pairwise_comparisons = ops.sigmoid(pairwise_diff / temperature)

    # Suma por fila: para cada muestra, cuántas quedan "por debajo" suavemente.
    ranks = 1.0 + ops.sum(pairwise_comparisons, axis=1, keepdims=True)

    return ranks


def spearman_corr_soft(x, y, temperature: float = 0.05, epsilon: float = _EPSILON):
    """
    Correlación de Spearman diferenciable aproximada.

    Spearman = Pearson(rank(x), rank(y)).

    Como el ranking exacto no es diferenciable, usamos soft_rank.
    """
    x_rank = soft_rank(x, temperature=temperature)
    y_rank = soft_rank(y, temperature=temperature)

    return pearson_corr(x_rank, y_rank, epsilon=epsilon)


@keras.saving.register_keras_serializable(package="b4t1")
def make_fair_loss(
    lambda_pearson: float = 1.0,
    lambda_spearman: float = 1.0,
    spearman_temperature: float = 0.05,
    epsilon: float = _EPSILON,
) -> Callable:
    """
    Crea una FAIR loss compatible con model.compile.

    La pérdida total es:

        BCE(y, y_pred)
        + lambda_pearson  * Pearson(y_pred, s)^2
        + lambda_spearman * Spearman_soft(y_pred, s)^2

    Parameters
    ----------
    lambda_pearson:
        Peso del penalty de correlación lineal.
    lambda_spearman:
        Peso del penalty de correlación monótona.
    spearman_temperature:
        Temperatura para el ranking suave.
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

        rho_spearman = spearman_corr_soft(
            y_pred,
            s,
            temperature=spearman_temperature,
            epsilon=epsilon,
        )

        fairness_penalty = (
            lambda_pearson * ops.square(rho_pearson)
            + lambda_spearman * ops.square(rho_spearman)
        )

        return bce + fairness_penalty

    return fair_loss


def fairness_metrics(
    y_true,
    y_pred,
    spearman_temperature: float = 0.05,
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
    rho_s = spearman_corr_soft(
        y_pred,
        s,
        temperature=spearman_temperature,
        epsilon=epsilon,
    )

    return {
        "pearson": rho_p,
        "spearman_soft": rho_s,
        "pearson_abs": ops.abs(rho_p),
        "spearman_soft_abs": ops.abs(rho_s),
    }
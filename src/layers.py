"""
src/layers.py — Taller B4-T1 (Diseño de Redes Confiables: Justicia e Incertidumbre)

Responsabilidad única: definir la capa customizada de Keras (Pilar 1).

DebtRatioCustomLayer calcula el ratio de endeudamiento
    AMT_CREDIT / AMT_INCOME_TOTAL
como feature estructural del modelo, y lo satura con tanh(ratio / k).

Contrato con `src/data.py` (canal custom, EN CRUDO, shape (N, 2)):
    índice 0 = AMT_CREDIT        (numerador)
    índice 1 = AMT_INCOME_TOTAL  (denominador)
El orden es contractual: si cambia en data.py (CUSTOM_COLS), cambia aquí.

Implementación exclusivamente con `keras.ops` para portabilidad multi-backend
(TensorFlow, PyTorch o JAX); sin `tf.*` directo.
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="taller_b4t1")
class DebtRatioCustomLayer(keras.layers.Layer):
    """Capa customizada (Pilar 1): ratio de endeudamiento saturado.

    Recibe el canal custom (N, 2) EN CRUDO (solo imputado, sin escalar):
        x[:, 0] = AMT_CREDIT        (numerador)
        x[:, 1] = AMT_INCOME_TOTAL  (denominador)

    Calcula AMT_CREDIT / AMT_INCOME_TOTAL y aplica una saturación suave
    tanh(ratio / k). Devuelve (N, 1): una única feature (el ratio saturado)
    que `src/model.py` concatena con la salida del canal denso.

    Parámetros
    ----------
    k : float
        Constante de saturación, NO entrenable. Se calibra en train (p. ej.
        el percentil 95 del ratio) y se pasa al construir la capa. Con esta
        parametrización, ratio = k se mapea a tanh(1) ≈ 0.76: la mayor parte
        de la distribución conserva resolución y la cola alta queda comprimida.
    epsilon : float
        Término en el denominador para evitar división por cero cuando el
        ingreso imputado es 0 o muy pequeño.

    Decisión de diseño
    ------------------
    La capa NO tiene pesos entrenables (no usa `build`/`add_weight`): `k` es
    una constante calibrada, no un parámetro que la red aprenda. Un divisor
    entrenable sería un grado de libertad que rompería la interpretación
    económica del ratio y no se justifica en el contexto del taller. Misma
    lógica con la que se eligió el ratio interpretable frente al empíricamente
    más fuerte.

    Se usa `tanh` y no `clip` porque es suave y diferenciable en todo el
    dominio: limita el efecto de valores extremos sin anular el gradiente
    fuera de un rango prefijado.
    """

    def __init__(self, k: float, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        if k <= 0:
            raise ValueError(f"k debe ser positivo (escala de saturación), recibido: {k}")
        self.k = float(k)
        self.epsilon = float(epsilon)

    def call(self, x):
        # Mantener la dimensión falsa (N, 1) en cada slice evita el broadcasting
        # accidental que rompería las operaciones elemento a elemento.
        credit = x[:, 0:1]   # (N, 1)  numerador
        income = x[:, 1:2]   # (N, 1)  denominador
        ratio = credit / (income + self.epsilon)   # >= 0  (credit, income >= 0)
        return ops.tanh(ratio / self.k)             # (N, 1) en [0, 1)

    def compute_output_shape(self, input_shape):
        # La salida NO coincide con la entrada: (batch, 2) -> (batch, 1).
        return (input_shape[0], 1)

    def get_config(self) -> dict:
        # Devuelve todos los argumentos de __init__ para que
        # keras.models.load_model reconstruya la capa sin argumentos extra.
        config = super().get_config()
        config.update({"k": self.k, "epsilon": self.epsilon})
        return config

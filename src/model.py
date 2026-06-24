"""
src/model.py - Taller B4-T1 (Diseno de Redes Confiables: Justicia e Incertidumbre)

Responsabilidad unica: construir y compilar el modelo principal y el espacio de
busqueda de AutoML (Keras Tuner).
"""

from __future__ import annotations

import keras
from keras import ops
import keras_tuner as kt

from src.layers import DebtRatioCustomLayer
from src.losses import make_fair_loss, pearson_corr


def _target_from_extended_y(y_true):
    """Extrae TARGET desde y_ext, manteniendo compatibilidad con y simple."""
    if len(y_true.shape) == 1:
        return y_true
    return y_true[:, 0:1]


def fair_pearson_sq(y_true_ext, y_pred):
    """
    Metrica auxiliar para Keras Tuner.

    Extrae la variable sensible (columna 1) del tensor extendido y calcula el
    cuadrado de la correlacion de Pearson. Permite registrar el nivel de equidad
    de cada trial para la curva de Pareto.
    """
    y_true_ext = ops.cast(y_true_ext, keras.backend.floatx())
    y_pred = ops.cast(y_pred, keras.backend.floatx())

    s = y_true_ext[:, 1:2]

    rho = pearson_corr(y_pred, s)
    return ops.square(rho)


class FairAccuracy(keras.metrics.BinaryAccuracy):
    """
    Wrapper de accuracy para compatibilidad con tensores extendidos.

    Aisla la etiqueta real (columna 0) e ignora la variable sensible.
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_real = _target_from_extended_y(y_true)
        super().update_state(y_true_real, y_pred, sample_weight)


class FairAUC(keras.metrics.AUC):
    """
    Wrapper de AUC para compatibilidad con tensores extendidos.

    Aisla la etiqueta real (columna 0) para que el Tuner pueda optimizar
    objective="val_auc".
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_real = _target_from_extended_y(y_true)
        super().update_state(y_true_real, y_pred, sample_weight)


class FairCreditHyperModel(kt.HyperModel):
    """
    Orquestador de arquitectura AutoML.

    Implementa el enrutamiento dual (custom / denso) y la busqueda de
    hiperparametros bajo optimizacion de AUC de validacion (val_auc).

    Modos de uso:
      - fixed_hparams=None: busqueda de arquitectura y lr con lambda_fair=0.0.
      - fixed_hparams=dict: arquitectura y lr fijos; solo varia lambda_fair.
    """

    def __init__(
        self,
        input_shape_custom: int,
        input_shape_dense: int,
        k_ratio: float,
        fixed_hparams: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_shape_custom = input_shape_custom
        self.input_shape_dense = input_shape_dense
        self.k_ratio = k_ratio
        self.fixed_hparams = dict(fixed_hparams) if fixed_hparams is not None else None

    def _fixed_hparam(self, name: str):
        if self.fixed_hparams is None:
            raise RuntimeError("fixed_hparams no esta configurado.")
        if name not in self.fixed_hparams:
            raise KeyError(f"Falta el hiperparametro fijo requerido: {name}")
        return self.fixed_hparams[name]

    def build(self, hp):
        input_custom = keras.Input(shape=(self.input_shape_custom,), name="input_custom")
        input_dense = keras.Input(shape=(self.input_shape_dense,), name="input_dense")

        ratio_endeudamiento = DebtRatioCustomLayer(k=self.k_ratio)(input_custom)
        merged_features = keras.layers.Concatenate()([ratio_endeudamiento, input_dense])

        x = merged_features

        if self.fixed_hparams is None:
            num_layers = hp.Int("num_layers", 1, 2)
        else:
            num_layers = int(self._fixed_hparam("num_layers"))

        for i in range(num_layers):
            if self.fixed_hparams is None:
                units = hp.Int(f"units_{i}", min_value=16, max_value=64, step=16)
                dropout = hp.Float(f"dropout_{i}", 0.1, 0.4, step=0.1)
            else:
                units = int(self._fixed_hparam(f"units_{i}"))
                dropout = float(self._fixed_hparam(f"dropout_{i}"))

            x = keras.layers.Dense(units=units, activation="relu")(x)
            x = keras.layers.Dropout(dropout)(x)

        output = keras.layers.Dense(1, activation="sigmoid", name="clasificacion_final")(x)
        model = keras.Model(inputs=[input_custom, input_dense], outputs=output)

        if self.fixed_hparams is None:
            learning_rate = hp.Choice("lr", [1e-3, 5e-4])
            lambda_val = 0.0
        else:
            learning_rate = float(self._fixed_hparam("lr"))
            lambda_val = hp.Choice("lambda_fair", values=[0.0, 2.0, 10.0, 25.0, 50.0])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=make_fair_loss(lambda_pearson=lambda_val),
            metrics=[
                FairAUC(name="auc"),
                FairAccuracy(name="accuracy"),
                fair_pearson_sq,
            ],
        )

        return model

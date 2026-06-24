"""
src/model.py — Taller B4-T1 (Diseño de Redes Confiables: Justicia e Incertidumbre)

Responsabilidad única: construir y compilar el modelo principal y el espacio de 
búsqueda de AutoML (Keras Tuner). 
"""

from __future__ import annotations

import keras
from keras import ops
import keras_tuner as kt

from src.layers import DebtRatioCustomLayer
from src.losses import make_fair_loss, pearson_corr

def fair_pearson_sq(y_true_ext, y_pred):
    """
    Métrica auxiliar para Keras Tuner.
    Extrae la variable sensible (columna 1) del tensor extendido y calcula 
    el cuadrado de la correlación de Pearson. Permite al Tuner registrar 
    el nivel de equidad en cada trial para la Curva de Pareto.
    """
    y_true_ext = ops.cast(y_true_ext, keras.backend.floatx())
    y_pred = ops.cast(y_pred, keras.backend.floatx())
    
    s = y_true_ext[:, 1:2]
    
    rho = pearson_corr(y_pred, s)
    return ops.square(rho)


class FairAccuracy(keras.metrics.BinaryAccuracy):
    """
    Wrapper de Precisión (Accuracy) para compatibilidad con tensores extendidos.
    Aísla la etiqueta real (columna 0) ignorando la variable sensible en el 
    cálculo de la métrica de rendimiento.
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_real = y_true[:, 0:1]
        super().update_state(y_true_real, y_pred, sample_weight)


class FairCreditHyperModel(kt.HyperModel):
    """
    Orquestador de arquitectura AutoML.
    Implementa el enrutamiento Dual (Custom / Denso) y la búsqueda de 
    hiperparámetros bajo optimización de Precisión (Accuracy).
    """
    def __init__(self, input_shape_custom: int, input_shape_dense: int, k_ratio: float, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_custom = input_shape_custom
        self.input_shape_dense = input_shape_dense
        self.k_ratio = k_ratio

    def build(self, hp):
        input_custom = keras.Input(shape=(self.input_shape_custom,), name="input_custom")
        input_dense = keras.Input(shape=(self.input_shape_dense,), name="input_dense")

        ratio_endeudamiento = DebtRatioCustomLayer(k=self.k_ratio)(input_custom)

        merged_features = keras.layers.Concatenate()([ratio_endeudamiento, input_dense])

        x = merged_features

        for i in range(hp.Int('num_layers', 1, 2)):
            x = keras.layers.Dense(
                units=hp.Int(f'units_{i}', min_value=16, max_value=64, step=16),
                activation='relu'
            )(x)
            x = keras.layers.Dropout(hp.Float(f'dropout_{i}', 0.1, 0.4, step=0.1))(x)

        output = keras.layers.Dense(1, activation='sigmoid', name="clasificacion_final")(x)
        model = keras.Model(inputs=[input_custom, input_dense], outputs=output)

        lambda_val = hp.Choice('lambda_fair', values=[0.0, 2.0, 10.0, 25.0, 50.0])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-3, 5e-4])),
            loss=make_fair_loss(
                lambda_pearson=lambda_val,
                lambda_spearman=0.0
            ),
            metrics=[
                FairAccuracy(name='accuracy'),
                fair_pearson_sq
            ]
        )
        
        return model

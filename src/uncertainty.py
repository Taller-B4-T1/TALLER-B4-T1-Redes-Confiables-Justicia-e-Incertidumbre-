"""
src/uncertainty.py — Taller B4-T1 (Diseño de Redes Confiables: Justicia e Incertidumbre)

Responsabilidad única: todo lo relativo al modelo secundario de incertidumbre
(Pilar 4). Estima el margen de error absoluto del modelo principal.
"""

from __future__ import annotations

import numpy as np
import keras

def compute_errors(y_true_ext: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula el error absoluto entre la predicción y la etiqueta real.
    Aísla la columna 0 (TARGET) del tensor extendido generado por src/data.py.
    """
    y_true = y_true_ext[:, 0].flatten()
    y_pred = y_pred.flatten()
    return np.abs(y_true - y_pred)

def build_uncertainty_model(input_dim: int) -> keras.Model:
    """
    Construye la arquitectura del estimador de incertidumbre.
    Implementa activación 'softplus' para garantizar magnitudes estrictamente 
    positivas y se optimiza mediante el Error Cuadrático Medio (MSE).
    """
    inputs = keras.Input(shape=(input_dim,), name="input_dense_unc")
    
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    
    # Salida: Magnitud del error esperado (incertidumbre predictiva)
    outputs = keras.layers.Dense(1, activation='softplus', name="estimacion_error")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=['mae']
    )
    return model

def predict_uncertainty(uncertainty_model: keras.Model, X_dense: np.ndarray) -> np.ndarray:
    """
    Infiere el margen de error esperado para los perfiles procesados en el Canal Denso.
    """
    return uncertainty_model.predict(X_dense, verbose=0).flatten()

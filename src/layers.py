import tensorflow as tf
from tensorflow import keras

class DebtRatioLayer(keras.layers.Layer):

    """Se utiliza para calcular el ratio de deuda y penalizarlo si supera un umbral
    definido por el modelo. El ratio de deuda se calcula como la relación entre el
    crédito y los ingresos, y si este ratio excede un umbral (threshold_deuda), se
    aplica una penalización utilizando la función softplus para suavizar la transición.

    Lo que aprende durante el entrenamiento: el valor de threshold_deuda — a partir de
    qué ratio de endeudamiento el riesgo empieza a escalar de forma no lineal. Ese umbral
    no lo fijamos nosotros, lo descubre la red con los datos.
    """

    def build(self, input_shape):
        self.threshold_deuda = self.add_weight(
            name='threshold_deuda',
            shape=(),
            initializer=tf.constant_initializer(3.0),
            constraint=keras.constraints.NonNeg(),
            trainable=True
        )
        super().build(input_shape)

    def _penalize(self, ratio, threshold):
        excess = tf.nn.softplus(ratio - threshold)
        return ratio + tf.square(excess)

    def call(self, inputs):
        income = inputs[:, 1:2]
        credit = inputs[:, 2:3]

        ratio_deuda = credit / (income + 1e-7)
        r1 = self._penalize(ratio_deuda, self.threshold_deuda)

        return tf.concat([inputs, r1], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1)
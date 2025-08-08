import tensorflow as tf


class ImputerLayer(tf.keras.layers.Layer):
    """ This class will create a custom Keras preprocessing layer to replace 
    NaN values with the min across instances without NaN in this position."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_vals = None

    def adapt(self, data):
        """ Calculate the minimum of the original non-NaN, values of each 
        position in the vector input.
        Args:
            data: A tensor with potential NaN values.
        """
        data = tf.convert_to_tensor(data, dtype=tf.float32)

        # Replace NaNs with large float to ignore them in min computation
        safe_data = tf.where(tf.math.is_nan(data), tf.fill(tf.shape(data), tf.float32.max), data)
        self.min_vals = tf.reduce_min(safe_data, axis=0)

    def call(self, inputs):
        """ Replace NaN values with the minimum value calculated in the adapt
        method. 
        Args:
            inputs: A tensor with potential NaN values.
        Returns:
            A tensor with NaN values replaced by the minimum values."""
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        return tf.where(tf.math.is_nan(inputs), self.min_vals, inputs)

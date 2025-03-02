"""
Aditya Iyer, 2025
This file contains the HubLayer class, which is a custom Keras layer that wraps a TensorFlow Hub layer.
"""
# Create a custom Layer subclass to wrap hub.KerasLayer
class HubLayer(tf.keras.layers.Layer):
    def __init__(self, handle, **kwargs):
        super().__init__(**kwargs)
        self.handle = handle
        self.hub_layer = hub.KerasLayer(handle, input_shape=IMAGE_SHAPE + (3,))

    def call(self, inputs):
        return self.hub_layer(inputs)
import tensorflow as tf
from tensorflow.keras import layers
class AttentionPooling(layers.Layer):
    """Additive (Bahdanau-style) attention pooling over the time axis.
    Learns which of the 16 selected frames matter most for each clip,
    instead of only using the LSTM's final hidden state.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.supports_masking = True

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.W = self.add_weight(name="att_W", shape=(feat_dim, self.units),
                                  initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_b", shape=(self.units,),
                                  initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(self.units, 1),
                                  initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)   # (B,T,units)
        score = tf.squeeze(tf.tensordot(score, self.u, axes=1), axis=-1)  # (B,T)
        if mask is not None:
            score = score + (1.0 - tf.cast(mask, tf.float32)) * -1e9
        weights = tf.nn.softmax(score, axis=1)                            # (B,T)
        context = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)  # (B,F)
        return context

    def compute_mask(self, inputs, mask=None):
        return None   # pooled output has no time dimension left to mask
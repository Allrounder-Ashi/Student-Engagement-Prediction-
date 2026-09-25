"""
custom_layers.py
============================================================
Custom Keras layers used by the daisee_v7_improved model.
Both are required in custom_objects when loading the .keras files:

    custom_objects={
        "AttentionPooling": AttentionPooling,
        "MaskedGlobalPooling": MaskedGlobalPooling,
    }

(MaskedGlobalPooling was missing from the previous custom_layers.py --
the model architecture uses it twice, mode="mean" and mode="max", so
loading would have failed once you tried to load the real trained model.)
============================================================
"""

import tensorflow as tf
from tensorflow.keras import layers


class AttentionPooling(layers.Layer):
    """Learns a soft, content-dependent weighting over frames."""

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
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        score = tf.squeeze(tf.tensordot(score, self.u, axes=1), axis=-1)
        if mask is not None:
            score = score + (1.0 - tf.cast(mask, tf.float32)) * -1e9
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


class MaskedGlobalPooling(layers.Layer):
    """Mask-aware mean or max pooling over the time axis."""

    def __init__(self, mode="mean", **kwargs):
        super().__init__(**kwargs)
        assert mode in ("mean", "max")
        self.mode = mode
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is None:
            if self.mode == "mean":
                return tf.reduce_mean(inputs, axis=1)
            return tf.reduce_max(inputs, axis=1)
        mask_f = tf.cast(mask, tf.float32)[:, :, tf.newaxis]
        if self.mode == "mean":
            summed = tf.reduce_sum(inputs * mask_f, axis=1)
            counts = tf.maximum(tf.reduce_sum(mask_f, axis=1), 1.0)
            return summed / counts
        else:
            neg_inf = (1.0 - mask_f) * -1e9
            return tf.reduce_max(inputs + neg_inf, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"mode": self.mode})
        return cfg

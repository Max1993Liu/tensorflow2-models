"""
Reference: https://arxiv.org/abs/1612.00410
Tutorial: https://kexue.fm/archives/6181
"""
from tensorflow import keras
import tensorflow as tf


class VIB(keras.layers.Layer):

    def __init__(self, lamb, **kwargs):
        super().__init__(**kwargs)
        self.lamb = lamb

    def call(self, x, training=None):
        z_mean, z_log_var = x
        u = tf.random.normal(shape=z_mean.shape)
        kl_loss = - 0.5 * tf.reduce_sum(tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 0))
        self.add_loss(self.lamb * kl_loss)
        
        if not training:
            u = 0.0  
        return z_mean + tf.exp(z_log_var / 2) * u
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


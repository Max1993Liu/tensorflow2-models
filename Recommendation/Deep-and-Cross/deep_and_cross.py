import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




class CrossNet(layers.Layer):
    """ A CrossNet is a stack of multiple Cross Layer """
    
    def __init__(self, n_layers=3, share_weights=False, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.share_weights = share_weights
        
    def build(self, input_shape):
        dim = input_shape[1]
        
        if self.share_weights:
            self.layers = layers.Dense(1, activation='relu')
        else:
            self.layers = [layers.Dense(1, activation='relu') for _ in range(self.n_layers)]
        super().build(input_shape)    
    
    def cross_layer(self, x0, x, layer):
        # both x0 and x has shape (None, feature_dim)
        dot_product = tf.matmul(tf.expand_dims(x0, 2), tf.expand_dims(x, 1)) # (None, feature_dim, feature_dim)
        
        out = tf.squeeze(layer(dot_product)) # (None, feature_dim)
        
        # residual connection
        out = out + x
        return x
    
    def call(self, x):
        x0 = tf.identity(x)   # (None, feature_dim)
        
        for i in range(self.n_layers):
            if self.share_weights:
                x = self.cross_layer(x0, x, self.layers)
            else:
                x = self.cross_layer(x0, x, self.layers[i])            
        return x
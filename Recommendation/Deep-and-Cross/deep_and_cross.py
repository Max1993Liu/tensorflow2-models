import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np




class CrossNet(layers.Layer):
    """ A CrossNet is a stack of multiple Cross Layer """
    
    def __init__(self, n_layers=3, share_weights=False, **kwargs):
        """
        :param share_weights: Whether to share weight and bias between layers 
        """
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




# passing multiple input as dictionary is supported by specifying the name for each Input
# as demonstrated in @omalleyt12's comment under this thread: https://github.com/tensorflow/tensorflow/issues/34114

# a dictionary mapping feature name to its number of unique values
discrete_feature_size = {
    'a': 5,
    'b': 4
}

continuous_feature_size = 10


discrete_input = [layers.Input(shape=(), dtype=tf.int8, name=name) for name in discrete_feature_size]
continuous_input = layers.Input(shape=(continuous_feature_size, ), 
                                dtype=tf.float32, name='continuous_features')

embeddings = []
for name, input_dim in discrete_feature_size.items():
    output_dim = min(int(6 * np.power(input_dim, 1/4)), input_dim)
    embeddings.append(layers.Embedding(input_dim, output_dim, name='{}_embedding'.format(name)))



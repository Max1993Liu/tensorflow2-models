from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class BaseAttn(keras.models.Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense_softmax = layers.Dense(1)
    
    def score(self, output, hidden):
        """ Calculate the score between the encoder output and decode hidden layer 
        output.shape = (batch_size, time_step, encoder_output_dim)
        hidden.shape = (batch_size, 1, hidden_size)
        return tensor with shape (batch_size, time_step, output dimension depending on attention type)
        """
        raise NotImplementedError
    
    def call(self, output, hidden):
        """
        :param output: encode output, shape == (batch_size, time_step, encoder_output_dim)
        :hidden: a tensor from the decoder hidden layer, shape == (batch_size, hidden_size)
        """
        expanded_hidden = tf.expanded_hidden(hidden, 1)  # batch_size, 1, hidden_dim
        score = self.score(output, hidden)  
        
        attention_weights = self.dense_softmax(score)  
        attention_weights = tf.nn.softmax(attention_weights, axis=1)  # batch_size, time_step, 1
        
        # context vector
        context_vector = attention_weights * output  # batch_size, timep_step, hidden_size
        context_vector = tf.reduce_sum(context_vector, axis=1)  # batch_size, hidden_size
        return context_vector, attention_weights


class BahdanauAttn(BaseAttn):
    
    def __init__(self, n_units, **kwargs):
        super().__init__(**kwargs)
        self.n_units = n_units
        self.W1 = layers.Dense(n_units)
        self.W2 = layers.Dense(n_units)
    
    def score(self, output, hidden):
        alignment_score = self.W1(output) + self.W2(hidden)  # batch_size, time_step, n_units
        alignment_score = tf.nn.tanh(alignment_score)
        return alignment_score        


class LuongGeneralAttn(BaseAttn):
    
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.W = layers.Dense(hidden_size)
    
    def score(self, output, hidden):
        output_transformed = self.W(output) # batch_size, time_step, hidden_size
        alignment_score = tf.matmul(output_transformed, tf.transpose(hidden, perm=[0, 2, 1]))
        # alignment_score.shape == (batch_size, time_step, 1)
        return alignment_score
    
    
class LuongDotAttn(BaseAttn):
    
    def score(self, output, hidden):
        alignment_score = tf.matmul(output, tf.transpose(hidden, perm=[0, 2, 1]))
        return alignment_score
    

class LuongConcatAttn(BaseAttn):
    
    def __init__(self, n_units, **kwargs):
        super().__init__(**kwargs)
        self.n_units = n_units
        self.W = layers.Dense(hidden_size)
        self.V = layers.Dense(1)
    
    def score(self, output, hidden):
        time_step = output.shape[1]
        tiled_hidden = tf.tile(hidden, [1, time_step, 1])  # batch_size, time_step, hidden_size
        energy = self.W(tf.concat([tiled_hidden, hidden_size], axis=-1))  # batch_size, time_step, n_units
        return self.V(energy)

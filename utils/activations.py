import tensorflow as tf
import math


def swish(x):
    return x * tf.nn.sigmoid(x)

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))

def softplus(x):
	""" Reference: https://en.wikipedia.org/wiki/Rectifier_(neural_networks) """
	return tf.log(1 + tf.exp(x))
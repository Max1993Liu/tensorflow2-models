import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




def downsample_block(x, filters, max_pooling=False, dropout=False):
	x = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same')(x)
	x = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same')(x)
	
	

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers



def build_model(feature_size,
                field_size, 
                embedding_size=8,
                dropout_fm=[0.0, 0.0],
                deep_layers=[32, 32],
                dropout_deep=[0.3, 0.3],
                batch_norm=False,
                mode='classification'):
    """
    :param feature_size: number of total features after one-hot, denote as M
    :param field_size: number of features, denote as F
    :param embedding_size: denoted as K
    :param batch_norm: whether to use batch normalization in the deep component of the model.
    :param mode: One of ('classification', 'regression')
    """
    feature_index_input = layers.Input(shape=(field_size, ), dtype=tf.int8)
    feature_value_input = layers.Input(shape=(field_size, ), dtype=tf.float32)

    embedding_layer = layers.Embedding(input_dim=feature_size, output_dim=embedding_size, 
                                     embeddings_initializer=tf.initializers.RandomUniform(-0.01, 0.01),
                                     name='embedding')

    embeddings = embedding_layer(feature_index_input)  # (None, F, K)

    # FM part
    fm_first_order = tf.reduce_sum(embeddings, axis=2) # (None, F)
    fm_first_order = layers.Dropout(dropout_fm[0])(fm_first_order)

    sum_feature_emb = tf.reduce_sum(embeddings, 1)  # (None, K)
    sum_feature_emb_square = tf.square(sum_feature_emb) # (None, K)

    square_feature_emb = tf.square(embeddings)  # (None, F, K)
    square_sum_feature_emb = tf.reduce_sum(square_feature_emb) # (None, K)

    fm_second_order = 0.5 * (sum_feature_emb_square - square_sum_feature_emb) # (None, K)
    fm_second_order = layers.Dropout(dropout_fm[1])(fm_second_order)

    # Deep part
    embeddings = tf.multiply(embeddings, feature_value_input[..., tf.newaxis])  # (None, F, K)
    deep = layers.Flatten()(embeddings)  # (None, F * K)

    for l, d in zip(deep_layers, dropout_deep):
        if batch_norm:
            deep = layers.BatchNormalization()(deep)
        deep = layers.Dense(l, activation='relu')(deep)
        deep = layers.Dropout(d)(deep)
        
    # Deep FM
    out = layers.Concatenate()([fm_first_order, fm_second_order, deep])
    activation = 'sigmoid' if mode == 'classification' else None
    out = layers.Dense(1, activation=activation)(out)

    model = keras.models.Model(inputs=[feature_index_input, feature_value_input], 
                               outputs=out,
                               name='DeepFM')
    return model

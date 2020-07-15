import tensorflow as tf


def spectral_norm(w, r=5):
    w_shape = tf.shape(w).numpy()
    in_dim = np.prod(w_shape[:-1])
    out_dim = w_shape[-1]
    
    w = tf.reshape(w, (in_dim, out_dim))
    u = tf.ones((1, in_dim), dtype='float32')
    
    for _ in range(r):
        v = tf.nn.l2_normalize(tf.matmul(u, w))
        u = tf.nn.l2_normalize(tf.matmul(v, tf.transpose(w)))
    return tf.reduce_sum(tf.matmul(tf.matmul(u, w), tf.transpose(v)))



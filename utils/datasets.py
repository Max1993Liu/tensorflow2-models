import tensorflow as tf

def create_imdb_dataset(n_vocab=5000, max_len=100, batch_size=32, shuffle=True):
    """ Return (train_dataset, test_dataset), labels are one-hot encoded """
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.utils import to_categorical

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n_vocab)

    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    if shuffle:
        train_dataset = train_dataset.shuffle(10000)
        test_dataset = test_dataset.shuffle(10000)

    return train_dataset, test_dataset

import tensorflow as tf
import numpy as np

class Fashion_mnist:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        self.fashion_mnist = tf.keras.datasets.fashion_mnist

    def get_data(self):
        # load fashion_mnist data
        (x_train, y_train), (x_test, y_test) = self.fashion_mnist.load_data()
        # (x_train, y_train), (x_test, y_test) = self.mnist.load_data()

        # adjusting to 0 ~ 1.0
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        print('x_train_before: {}'.format(x_train.shape))
        print('x_test_before: {}'.format(x_test.shape))

        # x_train = np.expand_dims(x_train, axis=-1)
        # x_train = tf.image.resize(x_train, [224, 224])
        #
        # x_test = np.expand_dims(x_test, axis=-1)
        # x_test = tf.image.resize(x_test, [224, 224])
        #
        # x_train = x_train.reshape(-1, 224, 224, 1)
        # x_test = x_test.reshape(-1, 224, 224, 1)
        #
        # print('x_train: {}'.format(x_train.shape))
        # print('x_test: {}'.format(x_test.shape))


        return x_train, y_train, x_test, y_test
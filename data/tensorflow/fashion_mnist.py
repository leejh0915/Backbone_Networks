import tensorflow as tf
import numpy as np

class Fashion_mnist:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        self.fashion_mnist = tf.keras.datasets.fashion_mnist

    def get_data(self):
        # load fashion_mnist data
        (x_train, y_train), (x_test, y_test) = self.fashion_mnist.load_data()

        # adjusting to 0 ~ 1.0
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        return x_train, y_train, x_test, y_test
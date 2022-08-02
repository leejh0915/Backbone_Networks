# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# # load fashion_mnist data
# fashion_mnist = tf.keras.datasets.cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
#
# # adjusting to 0 ~ 1.0
# x_train = x_train / 255.0
# x_test = x_test / 255.0
#
# print(x_train.shape, x_test.shape)
#
# # reshaping
# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)
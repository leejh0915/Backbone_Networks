import tensorflow as tf

class ResNet(tf.keras.Model):
  def __init__(self):
    super(ResNet, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(kernel_size=(7,7), filters=64, stride=2, activation='relu',
                                        initializer=tf.keras.initializers.HeUniform(),
                                        kernel_regularizer=tf.keras.regularizers.L2())
    self.pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), stride=2)
    pass

  def call(self, x):
    pass


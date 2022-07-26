import tensorflow as tf

class DenseNet(tf.keras.Model):
  def __init__(self):
    super(DenseNet, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(kernel_size=(7,7), filters=64, activation='relu')

  def call(self, x):
    pass

#참고
#https://pasus.tistory.com/18
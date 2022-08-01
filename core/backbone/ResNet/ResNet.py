import tensorflow as tf
from core.backbone.ResNet.BottleNeck import BottleNeckBlock_3n

class ResNet50(tf.keras.Model):
  def __init__(self, num_classes):
    super(ResNet50, self).__init__()
    #self.resize = tf.keras.layers.Resizing(224, 224)

    self.conv = tf.keras.layers.Conv2D(kernel_size=(7, 7), filters=64, strides=2)
    self.bn = tf.keras.layers.BatchNormalization()
    self.act = tf.keras.layers.Activation('relu')
    self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)

    self.bottle1 = BottleNeckBlock_3n(filters=64, kernel_size=1)
    self.bottle2 = BottleNeckBlock_3n(filters=128, kernel_size=1)
    self.bottle3 = BottleNeckBlock_3n(filters=256, kernel_size=1)
    self.bottle4 = BottleNeckBlock_3n(filters=512, kernel_size=1)

    self.avg = tf.keras.layers.GlobalAveragePooling2D()
    self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

  def call(self, inputs):
      #x = self.resize(inputs)

      x = self.conv(inputs)
      x = self.bn(x)
      x = self.act(x)
      x = self.max_pool(x)

      for i in range(3):
          x = self.bottle1(x)

      for i in range(4):
          x = self.bottle2(x)

      for i in range(6):
          x = self.bottle3(x)

      for i in range(3):
          x = self.bottle4(x)

      x = self.global_pool(x)
      return self.classifier(x)

class ResNet56(tf.keras.Model): #for cifar10
  def __init__(self, num_classes):
    super(ResNet56, self).__init__()
    self.conv = tf.keras.layers.Conv2D(kernel_size=(7, 7), filters=64, strides=2, padding='same')
    self.bn = tf.keras.layers.BatchNormalization()
    self.act = tf.keras.layers.Activation('relu')
    self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

    self.bottle1 = BottleNeckBlock_3n(filters=64, kernel_size=1)
    self.bottle2 = BottleNeckBlock_3n(filters=128, kernel_size=1)
    self.bottle3 = BottleNeckBlock_3n(filters=256, kernel_size=1)
    self.bottle4 = BottleNeckBlock_3n(filters=512, kernel_size=1)

    self.avg = tf.keras.layers.GlobalAveragePooling2D()
    self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

  def call(self, inputs):
      x = self.conv(inputs)
      x = self.bn(x)
      x = self.act(x)
      x = self.max_pool(x)

      print('max_pool: {}'.format(x))

      for i in range(3):
          x = self.bottle1(x)

      for i in range(4):
          x = self.bottle2(x)

      for i in range(6):
          x = self.bottle3(x)

      for i in range(3):
          x = self.bottle4(x)

      x = self.global_pool(x)
      return self.classifier(x)
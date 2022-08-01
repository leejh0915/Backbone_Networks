import tensorflow as tf

class BottleNeckBlock_3n(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(BottleNeckBlock_3n, self).__init__(name='')
        self.conv1 =tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size*3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 =tf.keras.layers.Conv2D(filters*4, kernel_size, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.shortcut_conv = tf.keras.layers.Conv2D(filters*4, kernel_size, padding='same')
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):

        shortcut = input_tensor

        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        if x.shape[3] != input_tensor.shape[3]:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut)

        print('x: {}'.format(x.shape))
        print('input_tensor: {}'.format(shortcut.shape))

        x = self.add([x, shortcut])
        x = self.act(x)

        return x
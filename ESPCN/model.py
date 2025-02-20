import tensorflow as tf
from tensorflow import keras

class ESPCN(keras.Model):
    def __init__(self, upscale_factor):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(128, 5, padding='same', activation='tanh', kernel_initializer='orthogonal')
        self.conv2 = keras.layers.Conv2D(64, 3, padding='same', activation='tanh', kernel_initializer='orthogonal')
        self.conv3 = keras.layers.Conv2D(64, 3, padding='same', activation='tanh', kernel_initializer='orthogonal')
        self.conv4 = keras.layers.Conv2D((upscale_factor ** 2)*3, 3, padding='same', activation='tanh', kernel_initializer='orthogonal')
        self.upscale_factor = upscale_factor

    def call(self, x):
        image_height = x.shape[1]
        image_width = x.shape[2]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = tf.nn.depth_to_space(x, self.upscale_factor)
        return x

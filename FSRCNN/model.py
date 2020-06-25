import tensorflow as tf
from tensorflow.keras import Model, layers

class FSRCNN(Model):

    def __init__(self, image_size=33, label_size=21, batch_size=128, c_dim=3):
        super(FSRCNN, self).__init__()

        input_shape= [batch_size, image_size, image_size,  c_dim]

        self.conv1 = layers.Conv2D(56, kernel_size=5, padding='same', input_shape=input_shape)
        self.conv2 = layers.Conv2D(16, kernel_size=1, padding='same')
        self.conv3 = layers.Conv2D(12, kernel_size=3, padding='same')
        self.conv4 = layers.Conv2D(12, kernel_size=3, padding='same')
        self.conv5 = layers.Conv2D(12, kernel_size=3, padding='same')
        self.conv6 = layers.Conv2D(12, kernel_size=3, padding='same')
        self.conv7 = layers.Conv2D(56, kernel_size=1, padding='same')
        self.conv8 = layers.Conv2DTranspose(3, kernel_size=9, padding='same')

    def call(self, x, is_training = False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return x

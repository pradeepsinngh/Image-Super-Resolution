import tensorflow as tf
from tensorflow.keras import Model, layers

class SRCNN(Model):

    def __init__(self, image_size=33, label_size=21, batch_size=128, c_dim=3):
        super(SRCNN, self).__init__()

        input_shape= [batch_size, image_size, image_size,  c_dim]
        self.conv1 = layers.Conv2D(64, kernel_size=9, activation=tf.nn.relu, padding='same', input_shape=input_shape)
        self.conv2 = layers.Conv2D(32, kernel_size=1, activation=tf.nn.relu, padding='same')
        self.conv3 = layers.Conv2D(3, kernel_size=5, activation=tf.nn.relu, padding='same')

    def call(self, x, is_training = True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

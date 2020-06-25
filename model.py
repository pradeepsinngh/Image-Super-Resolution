from utils import (read_data, input_setup, imsave, merge)
import time
import os
import matplotlib.pyplot as plt
import numpy as np
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



class ESPCN(Model):
    def __init__(self, upscale_factor):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(128, 5, padding='same', activation='relu', kernel_initializer='orthogonal')
        self.conv2 = keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu, kernel_initializer='orthogonal')
        self.conv3 = keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu, kernel_initializer='orthogonal')
        self.conv4 = keras.layers.Conv2D((upscale_factor ** 2)*3, 3, padding='same', activation=tf.nn.relu, kernel_initializer='orthogonal')
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

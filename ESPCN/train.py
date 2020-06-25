import argparse
import os
import tensorflow as tf
import numpy as np

from model import ESPCN
from data import get_training_set, get_test_set
from os.path import join, exists
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('-upscale_factor', default=2, type=int)
parser.add_argument('-num_epochs', default=105, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-seed', default=123, type=int)
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-output_dir', default='outputs', type=str)
parser.add_argument('-is_train', type=bool, default= False, help='True for training, False for testing [True]')


args = parser.parse_args()
tf.random.set_seed(args.seed)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.lr,
    decay_steps=400,
    decay_rate=0.99,
    staircase=True)


# model & optimizer
model = ESPCN(args.upscale_factor)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = model(x)
        # Compute loss.
        loss = tf.reduce_mean(tf.math.squared_difference(pred, y))
        # Variables to update, i.e. trainable variables.
        trainable_variables = model.trainable_variables
        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)
        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, trainable_variables))


def train():

    train_dataset = get_training_set(args.upscale_factor).shuffle(200).batch(args.batch_size)

    for epoch in range(args.num_epochs):
        for inputs in train_dataset:
            ds_image, image = inputs
            run_optimization(ds_image, image)

            pred = model(ds_image)
            loss = tf.reduce_mean(tf.math.squared_difference(pred, image))
            print("step: %i, loss: %f" %(epoch, loss))

def test():

    test_dataset = get_test_set(args.upscale_factor).batch(args.batch_size)
    for inputs in test_dataset:
        ds_image, image = inputs
        result = model(ds_image)

    for i in range(2):
        img = np.reshape(result[i], (256, 256, 3))
        img = Image.fromarray(img, 'RGB')
        img.save('my.png')
        img.show()

if __name__ == '__main__':

    if args.is_train:
        train()
    else:
        test()

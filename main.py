
from model import SRCNN, FSRCNN, ESPCN
from utils import input_setup
from utils import (read_data, input_setup, imsave, merge)
from metrics import mse, psnr1, psnr

import numpy as np
import tensorflow as tf
import argparse
import pprint
import os
import time


def main():

    parser = argparse.ArgumentParser( formatter_class= argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=55, help='Number of epoch [15000]')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch images [128]')
    parser.add_argument('--image_size', type=int, default= 33, help='The size of image to use [33]')
    parser.add_argument('--label_size', type=int, default= 33, help='The size of label [33]')
    parser.add_argument('--learning_rate', type=int, default= 1e-4, help='The learning rate of gradient descent algorithm [1e-4]')
    parser.add_argument('--c_dim', type=int, default= 3, help='Dimension of image color [3]')
    parser.add_argument('--scale', type=int, default= 3, help='The size of scale factor for preprocessing input image [3]')
    parser.add_argument('--stride', type=int, default= 14, help='The size of stride to apply input image [14]')
    parser.add_argument('--checkpoint_dir', type=str, default= 'checkpoint', help='Name of checkpoint directory [checkpoint]')
    parser.add_argument('--sample_dir', type=str, default= 'sample', help='Name of sample directory [sample]')
    parser.add_argument('--is_train', type=bool, default= False, help='True for training, False for testing [True]')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    srcnn = FSRCNN(image_size=args.image_size, label_size=args.label_size,
                    batch_size=args.batch_size, c_dim=args.c_dim)

    # Stochastic gradient descent optimizer.
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    # Optimization process.
    def run_optimization(x, y):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass.
            pred = srcnn(x, is_training=True)
            # Compute loss.
            loss = mse(pred, y)
            # Variables to update, i.e. trainable variables.
            trainable_variables = srcnn.trainable_variables
            # Compute gradients.
            gradients = g.gradient(loss, trainable_variables)
            # Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, trainable_variables))


    def train(args):

        if args.is_train:
            input_setup(args)
        else:
            nx, ny = input_setup(args)

        counter = 0
        start_time = time.time()

        if args.is_train:
            print("Training...")
            data_dir = os.path.join('./{}'.format(args.checkpoint_dir), "train.h5")
            train_data, train_label = read_data(data_dir)

            display_step = 5
            for step in range(args.epochs):
                batch_idxs = len(train_data) // args.batch_size

                for idx in range(0, batch_idxs):

                    batch_images = train_data[idx * args.batch_size : (idx + 1) * args.batch_size]
                    batch_labels = train_label[idx * args.batch_size : (idx + 1) * args.batch_size]
                    run_optimization(batch_images, batch_labels)

                    if step % display_step == 0:
                        pred = srcnn(batch_images)
                        loss = mse(pred, batch_labels)
                        #psnr_loss = psnr(batch_labels, pred)
                        #acc = accuracy(pred, batch_y)

                        #print("step: %i, loss: %f", "psnr_loss: %f" %(step, loss, psnr_loss))
                        #print("Step:'{0}', Loss:'{1}', PSNR: '{2}'".format(step, loss, psnr_loss))

                        print("step: %i, loss: %f" %(step, loss))

        else:
            print("Testing...")
            data_dir = os.path.join('./{}'.format(args.checkpoint_dir), "test.h5")
            test_data, test_label = read_data(data_dir)

            result = srcnn(test_data)
            result = merge(result, [nx, ny])
            result = result.squeeze()

            image_path = os.path.join(os.getcwd(), args.sample_dir)
            image_path = os.path.join(image_path, "test_image.png")
            print(result.shape)
            imsave(result, image_path)

    train(args)


if __name__ == '__main__':
    main()

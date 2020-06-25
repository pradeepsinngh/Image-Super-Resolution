import numpy as np
import tensorflow as tf
import math

def mse(labels, pred):
    return tf.reduce_mean(tf.square(labels - pred))

def psnr1(img1, img2):
    return tf.image.psnr(img1, img2, max_val=255)

# Compute Peak Signal to Noise Ratio
# PSNR = 20 * log (MAXi / root(MSE))
def psnr(label, image, max_val=1.):
    print(np.array(label).shape)
    h, w, _ = np.array(label).shape

    diff = image - label
    rmse = math.sqrt(np.mean(diff ** 2))
    if rmse == 0:
        return 100
    else:
        return 20 * math.log10(max_val / rmse)

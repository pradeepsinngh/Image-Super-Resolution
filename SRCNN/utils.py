import os, glob
import h5py, random
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image
import scipy.misc
import scipy.ndimage
import numpy as np
import tensorflow as tf
#import imageio

def read_data(path):
  """
  Read h5 format data file

  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3):
  """
  Preprocess single image file
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path)
  label_ = modcrop(image, scale)

  # Must be normalized
  image = np.asarray(image) / 255.
  label_ = np.asarray(label_) / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

def prepare_data(args, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset

    For train dataset, output data would be ['.../t1.png', '.../t2.png', ..., '.../t99.png']
  """
  if args.is_train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.png"))
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = glob.glob(os.path.join(data_dir, "*.png"))

  return data

def make_data(args, data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if args.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  return Image.open(path)
  #return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  image = np.array(image)
  if image.ndim == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup(args):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path
  if args.is_train:
    data = prepare_data(args, dataset="Train")
  else:
    data = prepare_data(args, dataset="Test")

  sub_input_sequence = []
  sub_label_sequence = []
  padding = abs(args.image_size - args.label_size) / 2

  if args.is_train:
    for i in range(len(data)):
      input_, label_ = preprocess(data[i], args.scale)

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape

      for x in range(0, h-args.image_size+1, args.stride):
        for y in range(0, w-args.image_size+1, args.stride):
          sub_input = input_[x:x+args.image_size, y:y+args.image_size] # [33 x 33]
          sub_label = label_[x+int(padding):x+int(padding)+args.label_size, y+int(padding):y+int(padding)+args.label_size] # [21 x 21]

          # Make channel value
          sub_input = sub_input.reshape([args.image_size, args.image_size, 3])
          sub_label = sub_label.reshape([args.label_size, args.label_size, 3])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)

  else:
    input_, label_ = preprocess(data[5], args.scale)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0
    for x in range(0, h-args.image_size+1, args.stride):
      nx += 1; ny = 0
      for y in range(0, w-args.image_size+1, args.stride):
        ny += 1
        sub_input = input_[x:x+args.image_size, y:y+args.image_size] # [33 x 33]
        sub_label = label_[x+int(padding):x+int(padding)+args.label_size, y+int(padding):y+int(padding)+args.label_size] # [21 x 21]

        sub_input = sub_input.reshape([args.image_size, args.image_size, 3])
        sub_label = sub_label.reshape([args.label_size, args.label_size, 3])

        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 3]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 3]

  make_data(args, arrdata, arrlabel)

  if not args.is_train:
    return nx, ny

def imsave(image, path):
    #img = Image.fromarray(image)
    min_val = np.min(image)
    max_val = np.max(image)
    image_clamped = (image - min_val) / (max_val - min_val)
    return matplotlib.image.imsave(path, image_clamped)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img

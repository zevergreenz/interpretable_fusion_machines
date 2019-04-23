from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import os

from mnist.vae import *
from mnist.latent_classifier import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"  # specify which GPU(s) to be used


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

original_dim = x_train.shape[1] ** 2
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

encoder, decoder, vae = train_vae(x_train, y_train)
batch_size = 128

plot_results((encoder, decoder),
             (x_test, y_test),
             batch_size=batch_size,
             model_name="vae_mlp")

fit_gmm((encoder, decoder),
        (x_test, y_test),
        batch_size=batch_size,
        model_name="vae_mlp")
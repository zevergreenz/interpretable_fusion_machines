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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"  # specify which GPU(s) to be used


def generate_latent_dataset(encoder, x_train, y_train, x_test, y_test, name='latent_dataset.npy'):
    z_train, _, _ = encoder.predict(x_train)
    z_test, _, _ = encoder.predict(x_test)
    np.save(name, [z_train, y_train, z_test, y_test])
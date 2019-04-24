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


def generate_latent_dataset(encoder, x_train, y_train, x_test, y_test, name='latent_dataset.npy'):
    z_train, z_log_var_train, _ = encoder.predict(x_train)
    z_test, z_log_var_test, _ = encoder.predict(x_test)
    np.save(name, [z_train, z_log_var_train, y_train, z_test, z_log_var_test, y_test])
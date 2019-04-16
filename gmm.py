from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from sklearn.mixture.gaussian_mixture import GaussianMixture
from sklearn.datasets.samples_generator import make_blobs

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import os


# Generate data from mixture of 5 Gaussians
X, y = make_blobs(n_samples=500, centers=3, n_features=2)
figure = plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
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

from numpy import matmul, transpose, log, sqrt
from numpy.linalg import det, inv

from common import *


def activation_score(z_mean, z_log_var, pattern):
    z_var = np.diag(np.exp(z_log_var)[0])
    p_mean, p_var = pattern
    return Bhattacharyya_coeff(np.ravel(z_mean), z_var, p_mean, p_var)


class RecomposedClassifier(object):
    def __init__(self, specilized_clfs, num_labels):
        self.specialized_clfs = specilized_clfs
        self.num_labels = num_labels

    def p_label(self, encoder, x):
        z_mean, z_log_var, _ = encoder.predict(x)
        sum_scores = 0

        p_label = np.array([0.] * self.num_labels)
        for clf in self.specialized_clfs:
            activation = activation_score(z_mean, z_log_var, clf.pattern)
            p_label += clf.p_label * activation
            sum_scores += activation

        return p_label / sum_scores
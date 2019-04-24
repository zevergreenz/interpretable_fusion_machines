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


class SpecializedClassifier(object):
    def __init__(self, latent_clf, pattern, z_train, z_log_var_train, y_train):
        self.pattern = pattern
        self.labels = np.unique(y_train)
        self.p_label = np.array([0.0] * len(self.labels))

        y_pred = latent_clf.predict(z_train)

        for i in range(z_train.shape[0]):
            self.p_label += y_pred[i, :] * activation_score(z_train[i:i+1], z_log_var_train[i:i+1], self.pattern)

        self.p_label /= z_train.shape[0]

    def p_label(self, label):
        return self.p_label[label]
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


def make_ellipses(gmm, ax):
    for n in range(len(gmm.means_)):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color='b', alpha=0.7)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ell.set_alpha(0.3)
        ax.set_aspect('equal', 'datalim')

def kl_dist(mu1, sigma1, mu2, sigma2):
    kl_dist = np.linalg.slogdet(sigma2) - np.linalg.slogdet(sigma1)
    kl_dist += - mu1.shape[-1]
    kl_dist += np.trace(np.linalg.inv(sigma2))


# def Bhattacharyya_coeff(mu1, sigma1, mu2, sigma2):
#     DB = 0.5 * np.log(det(sigma1+sigma2) / sqrt(det(sigma1)*det(sigma2)))
#     DB += 1/8 * matmul(transpose(mu1-mu2), matmul(inv(sigma1+sigma2), mu1-mu2))
#     return np.exp(-DB)
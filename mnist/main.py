from keras.layers import Lambda, Input, Dense
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import keras
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import os

from mnist.vae import *
from mnist.latent_classifier import *
from mnist.specialized_classifier import *
from mnist.recomposed_classifier import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"  # specify which GPU(s) to be used


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

original_dim = x_train.shape[1] ** 2
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# latent_dims = [2, 5, 10]
# latent_dims = [15, 20, 30]
latent_dims = [3]
test_accs = []
for latent_dim in latent_dims:
    print('training nn ', latent_dim)
    encoder, decoder, vae = train_vae(x_train, y_train, latent_dim=latent_dim, weights='mnist_vae_%d.h5' % latent_dim)
    batch_size = 128

    generate_latent_dataset(encoder, x_train, y_train, x_test, y_test, 'mnist_latent.npy')
    z_train, z_log_var_train, _, z_test, z_log_var_test, _ = np.load('mnist_latent.npy')

    from sklearn.gaussian_process import GaussianProcessClassifier
    # clf = GaussianProcessClassifier(n_restarts_optimizer=5, multi_class='one_vs_rest')
    # clf.fit(z_train, y_train)
    # print('latent dim: ', latent_dim, 'score: ', clf.score(z_test, y_test))

    import keras
    latent_clf = keras.Sequential([
        keras.layers.Dense(latent_dim, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    latent_clf.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    latent_clf.fit(z_train, y_train, epochs=10)
    test_loss, test_acc = latent_clf.evaluate(z_test, y_test)
    test_accs.append(test_acc)
    print('latent dim: ', latent_dim, 'score: ', test_acc)

    num_patern = 20
    gmm = fit_gmm((encoder, decoder), (x_train, y_train), num_patern)

    specialized_clfs = []
    for i in range(num_patern):
        clf = SpecializedClassifier(latent_clf,
                                    (gmm.means_[i], gmm.covariances_[i]),
                                    z_train,
                                    z_log_var_train,
                                    y_train)
        specialized_clfs.append(clf)

    recomposed_clf = RecomposedClassifier(specialized_clfs, 10)

    pred = []
    for i in range(x_test.shape[0]):
        label = np.argmax(recomposed_clf.p_label(encoder, x_test[i:i+1, :]))
        pred.append(label)
    pred = np.array(pred)
    print(np.count_nonzero(pred == y_test))
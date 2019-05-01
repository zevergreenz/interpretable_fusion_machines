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


results = []
def run_experiment(latent_dim, num_pattern):
    print('Running experiment with latent dim: %d; num patterns: %d', (latent_dim, num_pattern))
    encoder, decoder, vae = train_vae(x_train, y_train, latent_dim=latent_dim, weights='mnist_vae_%d.h5' % latent_dim)

    # Train the latent classifier.
    generate_latent_dataset(encoder, x_train, y_train, x_test, y_test, 'mnist_latent.npy')
    z_train, z_log_var_train, _, z_test, z_log_var_test, _ = np.load('mnist_latent.npy')

    latent_clf = keras.Sequential([
        keras.layers.Dense(latent_dim, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    latent_clf.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    latent_clf.fit(z_train, y_train, epochs=10, verbose=0)
    _, latent_clf_acc = latent_clf.evaluate(z_test, y_test)


    # Train the specialized classifier.
    gmm = fit_gmm((encoder, decoder), (x_train, y_train), num_pattern)

    specialized_clfs = []
    for i in range(num_pattern):
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
    recomposed_clf_acc = float(np.count_nonzero(pred == y_test)) / y_test.shape[0]

    results.append((latent_dim, num_pattern, latent_clf_acc, recomposed_clf_acc))


# latent_dims = range(1, 21)
num_paterns = range(10, 201, 10)
latent_dims = [30]
# num_paterns = [10]
for latent_dim in latent_dims:
    for num_patern in num_paterns:
        print('Latent dim %d num_pattern %d' % (latent_dim, num_patern))
        run_experiment(latent_dim, num_patern)
        np.save('results_%d.npy' % latent_dim, np.array(results))
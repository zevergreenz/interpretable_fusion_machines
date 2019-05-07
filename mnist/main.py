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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

original_dim = x_train.shape[1] ** 2
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


results = []
specialized_clfs = []
def run_experiment(latent_dim, num_pattern):
    digit_size = 28

    print('Running experiment with latent dim: %d; num patterns: %d', (latent_dim, num_pattern))
    encoder, decoder, vae = train_vae(x_train, y_train, latent_dim=latent_dim, weights='mnist_vae_%d.h5' % latent_dim)

    # Train the latent classifier.
    print('Training latent classifier...')
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
    print('Training specialized classifiers...')
    gmm = fit_gmm((encoder, decoder), (x_train, y_train), num_pattern)

    for i in range(num_pattern):
        print('Training pattern %d...' % i)
        clf = SpecializedClassifier(latent_clf,
                                    (gmm.means_[i], gmm.covariances_[i]),
                                    z_train,
                                    z_log_var_train,
                                    y_train)
        specialized_clfs.append(clf)

    recomposed_clf = RecomposedClassifier(specialized_clfs, 10)


    # Visualize cluster centroid =======================================================================================
    print("Visualize cluster centre...")
    for count, clf in enumerate(specialized_clfs):
        clf = specialized_clfs[count]
        mean, var = clf.pattern
        mean = np.reshape(mean, (1, latent_dim))
        decoded_mean = decoder.predict(mean)
        decoded_mean = np.reshape(decoded_mean, (digit_size, digit_size))
        plt.clf()
        plt.imshow(decoded_mean)
        plt.savefig("pattern_%d.png" % count, dpi=300)


    # Visualize top training examples from each specialized classifier =================================================
    # for count, clf in enumerate(specialized_clfs):
    #     activations = []
    #     for i in range(z_train.shape[0]):
    #         activation = activation_score(z_train[i:i + 1], z_log_var_train[i:i + 1], clf.pattern)
    #         activations.append(activation)
    #     activations = np.array(activations)
    #     res = activations.argsort()[-16:][::-1]
    #     # display a 4x4 2D manifold of digits
    #     n = 4
    #     figure = np.zeros((digit_size * n, digit_size * n))
    #     # linearly spaced coordinates corresponding to the 2D plot
    #     # of digit classes in the latent space
    #     grid_x = np.linspace(-4, 4, n)
    #     grid_y = np.linspace(-4, 4, n)[::-1]
    #
    #     idx = 0
    #     for i, yi in enumerate(grid_y):
    #         for j, xi in enumerate(grid_x):
    #             digit = x_train[res[idx]].reshape(digit_size, digit_size)
    #             idx += 1
    #             figure[i * digit_size: (i + 1) * digit_size,
    #             j * digit_size: (j + 1) * digit_size] = digit
    #
    #     plt.figure(figsize=(10, 10))
    #     start_range = digit_size // 2
    #     end_range = n * digit_size + start_range + 1
    #     pixel_range = np.arange(start_range, end_range, digit_size)
    #     sample_range_x = np.round(grid_x, 1)
    #     sample_range_y = np.round(grid_y, 1)
    #     plt.xticks(pixel_range, sample_range_x)
    #     plt.yticks(pixel_range, sample_range_y)
    #     plt.xlabel("z[0]")
    #     plt.ylabel("z[1]")
    #     plt.imshow(figure, cmap='Greys_r')
    #     plt.savefig('pattern_%d.png' % count, dpi=300)


    # Evaluate recomposed model ========================================================================================
    # pred = []
    # for i in range(x_test.shape[0]):
    #     label = np.argmax(recomposed_clf.p_label(encoder, x_test[i:i+1, :]))
    #     pred.append(label)
    # pred = np.array(pred)
    # print(np.count_nonzero(pred == y_test))
    # recomposed_clf_acc = float(np.count_nonzero(pred == y_test)) / y_test.shape[0]
    #
    # results.append((latent_dim, num_pattern, latent_clf_acc, recomposed_clf_acc))



# latent_dims = range(1, 21)
# num_paterns = range(1000, 5001, 1000)
latent_dims = [10]
num_paterns = [200]
for latent_dim in latent_dims:
    for num_patern in num_paterns:
        print('Latent dim %d num_pattern %d' % (latent_dim, num_patern))
        run_experiment(latent_dim, num_patern)
        np.save('results_%d.npy' % latent_dim, np.array(results))
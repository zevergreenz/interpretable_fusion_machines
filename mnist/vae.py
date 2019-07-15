import os

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import backend as K
from keras import layers
from keras.layers import Lambda, Input, Dense
from keras.losses import mse, sparse_categorical_crossentropy
from keras.models import Model
from keras.utils import plot_model

from sklearn.mixture.gaussian_mixture import GaussianMixture


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

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

def fit_gmm(models,
            data,
            n_components=10,
            batch_size=128,
            model_name="vae_mnist"):
    encoder, decoder = models
    x_test, y_test = data
    # x_test = x_test[y_test == 1]
    # y_test = y_test[y_test == 1]
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full').fit(z_mean)

    # figure, (ax) = plt.subplots(1, 1, figsize=(10, 10))
    # ax.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    # plt.colorbar()
    # ax.xlabel("z[0]")
    # ax.ylabel("z[1]")
    # make_ellipses(gmm, ax)
    # plt.show()
    return gmm

def train_vae(x_train, y_train, latent_dim=2, weights='mnist_vae.h5'):
    # network parameters
    original_dim = x_train.shape[1]
    input_shape = (original_dim, )
    intermediate_dim = 512
    batch_size = 128
    epochs = 50

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    # encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    # decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = mse(inputs, outputs)
    # reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    if os.path.isfile(weights):
        print("Found saved weights, loading from file ...")
        vae.load_weights(weights)
    else:
        # train the autoencoder
        print("Saving weights to %s" % weights)
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
                # validation_data=(x_test, None))
        vae.save_weights(weights)

    return encoder, decoder, vae

def train_cnn_vae(x_train, y_train, latent_dim=2, weights='cnn_vae.h5'):
    # network parameters
    original_dim = x_train.shape[1]
    input_shape = (original_dim, )
    intermediate_dim = 1024
    batch_size = 256
    epochs = 70

    """
      Convolutional structure for the encoder net
    """

    encoder = keras.Sequential([
        layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same', input_shape=x_train.shape[1:], batch_size=batch_size),
        layers.Conv2D(filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
        layers.Conv2D(filters=512, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
        layers.Flatten()
    ])

    """
      DeConv structure for the decoder net
    """

    decoder = keras.Sequential([
        layers.Dense(2048, input_shape=(1024,)),
        layers.Reshape(target_shape=(4, 4, 128), input_shape=(None, 1024)),
        layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
        layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
        layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
    ])
    # x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
    x = Input(shape=[32, 32, 3])

    encoded = encoder(x)

    mean = layers.Dense(1024, activation=tf.nn.softplus)(encoded)
    sigma = layers.Dense(1024, activation=tf.nn.relu)(encoded)

    z = Lambda(sampling, output_shape=(1024,), name='z')([mean, sigma])
    # z = mean + tf.multiply(tf.sqrt(tf.exp(sigma)),
    #                        tf.random_normal(shape=(batch_size, 1024)))

    x_reco = decoder(z)

    vae = Model(x, x_reco, name='vae_cnn')

    # reconstruction_term = -tf.reduce_sum(tfp.distributions.MultivariateNormalDiag(
    #     layers.Flatten()(x_reco), scale_identity_multiplier=0.05).log_prob(layers.Flatten()(x)))
    reconstruction_term = tf.reduce_sum(keras.losses.categorical_crossentropy(x, x_reco), axis=[1, 2])

    kl_divergence = tf.reduce_sum(tf.keras.metrics.kullback_leibler_divergence(x, x_reco), axis=[1, 2])

    cost = tf.reduce_mean(reconstruction_term + kl_divergence)

    vae.add_loss(cost)
    vae.compile(optimizer='adam')

    # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    #
    # runs = 20
    # n_minibatches = int(x_train.shape[0] / batch_size)
    #
    # print("Number of minibatches: ", n_minibatches)
    #
    # sess = tf.InteractiveSession()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    #
    # for epoch in range(runs):
    #     pbar = tf.contrib.keras.utils.Progbar(n_minibatches)
    #     for i in range(n_minibatches):
    #         x_batch = x_train[i * batch_size:(i + 1) * batch_size] / 255.
    #         cost_, _ = sess.run((cost, optimizer), feed_dict={x: x_batch})
    #
    #         pbar.add(1, [("cost", cost_)])

    if os.path.isfile(weights):
        print("Found saved weights, loading from file ...")
        vae.load_weights(weights)
    else:
        # train the autoencoder
        print("Saving weights to %s" % weights)
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
                # validation_data=(x_test, None))
        vae.save_weights(weights)

    return encoder, decoder, vae
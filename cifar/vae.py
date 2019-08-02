import keras
import keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings

batch_size = 256

"""
  Convolutional structure for the encoder net
"""

encoder = keras.Sequential([
    layers.Conv2D(filters=64 , kernel_size=4, strides=2, activation=tf.nn.relu, padding='same', input_shape=(32, 32, 3), batch_size=batch_size),
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
    layers.Conv2DTranspose(filters=64 , kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
    layers.Conv2DTranspose(filters=3  , kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
])

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

# x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
# x = layers.Input(batch_shape=(batch_size, 32, 32, 3))
x = layers.Input(shape=(32, 32, 3))

encoded = encoder(x)

mean = layers.Dense(1024, activation=tf.nn.softplus)(encoded)
sigma = layers.Dense(1024, activation=tf.nn.relu)(encoded)

# z = mean + tf.multiply(tf.sqrt(tf.exp(sigma)),
#                        tf.random_normal(shape=(batch_size, 1024)))
z = layers.Lambda(sampling)([mean, sigma])
my_encoder = keras.models.Model(x, [mean, sigma, z])

latent_inputs = layers.Input(shape=(1024,))

x_reco = decoder(latent_inputs)
my_decoder = keras.models.Model(latent_inputs, x_reco)

x_reco = my_decoder(my_encoder(x)[2])
my_vae = keras.models.Model(x, x_reco)

reconstruction_term = -tf.reduce_sum(tfp.distributions.MultivariateNormalDiag(
    layers.Reshape((3072,))(x_reco), scale_identity_multiplier=0.05).log_prob(layers.Reshape((3072,))(x)))

kl_divergence = tf.reduce_sum(tf.keras.metrics.kullback_leibler_divergence(x, x_reco), axis=[1, 2])

cost = tf.reduce_mean(reconstruction_term + kl_divergence)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

from keras.datasets.cifar10 import load_data
(X_train, y_train), (X_test, y_test) = load_data()


runs = 1
n_minibatches = int(X_train.shape[0] / batch_size)

print("Number of minibatches: ", n_minibatches)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(runs):
    pbar = tf.contrib.keras.utils.Progbar(n_minibatches)
    for i in range(n_minibatches):
        x_batch = X_train[i * batch_size:(i + 1) * batch_size] / 255.
        cost_, _ = sess.run((cost, optimizer), feed_dict={x: x_batch})

        pbar.add(1, [("cost", cost_)])

import matplotlib.pyplot as plt
import numpy as np

n_rec = 10

x_batch = X_train[:batch_size]

plt.figure(figsize=(n_rec + 6, 4))

pred_img = sess.run(x_reco, feed_dict={x: x_batch})
pred_img = pred_img.reshape(batch_size, 32, 32, 3)
pred_img = pred_img.astype(np.int32)

for i in range(n_rec):
    plt.subplot(2, n_rec, i + 1)
    plt.imshow(x_batch[i])

    plt.subplot(2, n_rec, n_rec + i + 1)
    plt.imshow(pred_img[i])

plt.tight_layout()

plt.savefig("vae2.png")

def train_cnn_vae():
    return my_encoder, my_decoder, my_vae
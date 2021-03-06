import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Lambda, Input, Dense
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.utils import plot_model
from sklearn.datasets.samples_generator import make_blobs

from common import make_ellipses


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


def fit_gmm(models,
            data,
            batch_size=128,
            model_name="vae_mnist"):
    encoder, decoder = models
    x_test, y_test = data
    # x_test = x_test[y_test == 1]
    # y_test = y_test[y_test == 1]
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    from sklearn.mixture.gaussian_mixture import GaussianMixture
    gmm = GaussianMixture(n_components=5, covariance_type='full').fit(z_mean)

    figure, (ax) = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    # plt.colorbar()
    # ax.xlabel("z[0]")
    # ax.ylabel("z[1]")
    make_ellipses(gmm, ax)
    plt.show()

# Generate data from mixture of 5 Gaussians
x, y = make_blobs(n_samples=5000, centers=5, n_features=6)
# figure = plt.figure(figsize=(10, 10))
# plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.show()

x_train = x[:4000, :]
x_test = x[4000:, :]
y_train = y[:4000]
y_test = y[4000:]

# network parameters
original_dim = 6
input_shape = (original_dim,)
intermediate_dim = 6
batch_size = 128
latent_dim = 2
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

if __name__ == '__main__':
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(inputs, outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    # vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    # train the autoencoder
    vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    # vae.save_weights('vae_mlp_mnist.h5')

    fit_gmm(models,
            data,
            batch_size=batch_size,
            model_name="vae_mlp")

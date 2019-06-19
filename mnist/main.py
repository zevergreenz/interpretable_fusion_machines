import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
from keras.datasets import mnist
from sklearn.mixture.gaussian_mixture import GaussianMixture
from tensorflow.contrib.factorization.python.ops import gmm as gmm_lib

from mnist.vae import train_vae

tfb = tfp.bijectors

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"  # specify which GPU(s) to be used
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings


def Bhattacharyya_coeff(mu1, sigma1, mu2, sigma2):
    N = tf.shape(mu1)[0]
    M = tf.shape(mu2)[0]
    Z = tf.shape(mu1)[1]
    mu1 = tf.reshape(mu1, [N, 1, Z])
    sigma1 = tf.reshape(sigma1, [N, 1, Z, Z])
    mu1 = tf.tile(mu1, [1, M, 1])
    sigma1 = tf.tile(sigma1, [1, M, 1, 1])
    mu2 = tf.broadcast_to(mu2, [N, M, Z])
    sigma2 = tf.broadcast_to(sigma2, [N, M, Z, Z])
    mu1 = tf.reshape(mu1, [N, M, Z, 1])
    mu2 = tf.reshape(mu2, [N, M, Z, 1])
    sigma = (sigma1 + sigma2) / 2.0
    # DB = 0.5 * tf.log(tf.linalg.det(sigma) / tf.sqrt(tf.linalg.det(sigma1)*tf.linalg.det(sigma2)))
    DB = 1/2 * tf.linalg.logdet(sigma) - 1/4 * tf.linalg.logdet(sigma1) - 1/4 * tf.linalg.logdet(sigma2)
    DB += 1/8 * tf.reshape(tf.matmul(tf.linalg.transpose(mu1-mu2), tf.matmul(tf.linalg.inv(sigma), mu1-mu2)), [N, M])
    # D_KL = tf.log(tf.linalg.det(sigma2) / tf.linalg.det(sigma1)) - Z.value
    # D_KL += tf.linalg.trace(tf.matmul(tf.linalg.inv(sigma2), sigma1))
    # D_KL += tf.reshape(tf.matmul(tf.linalg.transpose(mu1-mu2), tf.matmul(tf.linalg.inv(sigma2), mu1-mu2)), [N, M])
    # D_KL *= 1/2
    return tf.exp(-DB)


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

original_dim = x_train.shape[1] ** 2
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Build a black-box model and get its predictions
black_box_model_weights_filename = 'black_box.h5'
black_box_model = keras.Sequential([
    keras.layers.Dense(784, input_shape=(28*28,), activation=tf.nn.relu),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(10)
])
black_box_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
if os.path.isfile(black_box_model_weights_filename):
    print("Loading black-box model weights...")
    black_box_model.load_weights(black_box_model_weights_filename)
else:
    print("Training black-box model...")
    black_box_model.fit(x_train, y_train, epochs=10, verbose=0)
    print("Saving weights...")
    black_box_model.save_weights(black_box_model_weights_filename)
print("Black-box model accuracy: %.4f" % black_box_model.evaluate(x_test, y_test)[1])
true_pred = black_box_model.predict(x_train)


latent_dim = 5
num_pattern = 200
N = x_train.shape[0]
M = num_pattern
L = 10
D = 784
Z = latent_dim
B = 8192
E = 1000

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

encoder, decoder, vae = train_vae(x_train, y_train, latent_dim=Z, weights='mnist_vae_%d.h5' % Z)
z_train, z_log_var_train, _ = encoder.predict(x_train)
z_test, z_log_var_test, _ = encoder.predict(x_test)

# Creating one full datasets and two sub-datasets
full_dataset = tf.data.Dataset.from_tensor_slices((x_train, z_train, z_log_var_train, y_train))
indices1 = np.argwhere(
    np.logical_or.reduce((y_train == 0, y_train == 1, y_train == 2, y_train == 3, y_train == 4)))[:, 0]
dataset1 = tf.data.Dataset.from_tensor_slices(
    (x_train[indices1], z_train[indices1], z_log_var_train[indices1], y_train[indices1]))
indices2 = np.argwhere(
    np.logical_or.reduce((y_train == 5, y_train == 6, y_train == 7, y_train == 8, y_train == 9)))[:, 0]
dataset2 = tf.data.Dataset.from_tensor_slices(
    (x_train[indices2], z_train[indices2], z_log_var_train[indices2], y_train[indices2]))

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, z_test, z_log_var_test, y_test))
test_indices1 = np.argwhere(
    np.logical_or.reduce((y_test == 0, y_test == 1, y_test == 2, y_test == 3, y_test == 4)))[:, 0]
test_dataset1 = tf.data.Dataset.from_tensor_slices(
    (x_test[test_indices1], z_test[test_indices1], z_log_var_test[test_indices1], y_test[test_indices1]))
test_indices2 = np.argwhere(
    np.logical_or.reduce((y_test == 5, y_test == 6, y_test == 7, y_test == 8, y_test == 9)))[:, 0]
test_dataset2 = tf.data.Dataset.from_tensor_slices(
    (x_test[test_indices2], z_test[test_indices2], z_log_var_test[test_indices2], y_test[test_indices2]))


class AgentFactory(object):
    def __init__(self):
        """
        Objects which are shared across many agents
        (e.g. vae, ...)
        """
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, full_dataset.batch(B).output_types,
                                                       full_dataset.batch(B).output_shapes)
        x_train_ph, z_mean_ph, z_log_var_ph, y_train_ph = self.iterator.get_next()

        # x_train_ph = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]])
        # y_train_ph = tf.placeholder(tf.float32, shape=[None,])
        # z_mean_ph = tf.placeholder(tf.float32, shape=[None, z_train.shape[1]])
        # z_log_var_ph = tf.placeholder(tf.float32, shape=[None, z_log_var_train.shape[1]])
        z_cov = tf.matrix_diag(tf.exp(z_log_var_ph + 1e-10))

        # Train the latent classifier ==================================================================================
        print('Training latent classifier...')
        self.latent_clf = keras.Sequential([
            keras.layers.InputLayer(input_tensor=z_mean_ph, input_shape=(Z,)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        latent_loss = K.sparse_categorical_crossentropy(y_train_ph, self.latent_clf.output, from_logits=True)
        self.latent_train_step = tf.train.AdamOptimizer().minimize(latent_loss)

        # Train the specialized classifiers ================================================================================
        print('Training specialized classifiers...')
        self.scale_to_unconstrained = tfb.Chain([
            # step 3: flatten the lower triangular portion of the matrix
            tfb.Invert(tfb.FillTriangular(validate_args=True)),
            # step 2: take the log of the diagonals
            tfb.TransformDiagonal(tfb.Invert(tfb.Exp(validate_args=True))),
            # # step 1: decompose the precision matrix into its Cholesky factors
            # tfb.Invert(tfb.CholeskyOuterProduct(validate_args=True)),
        ])

        # random_init = False
        # if random_init:
        #     means = tf.Variable(initial_value=tf.random_uniform(means_.shape), trainable=True, dtype=tf.float32)
        #     scales_unconstrained = tf.Variable(
        #         initial_value=scale_to_unconstrained.forward(np.linalg.cholesky(covariances_)),
        #         trainable=True, dtype=tf.float32)
        #     scales_unconstrained = tf.Variable(initial_value=tf.random_uniform(scales_unconstrained.shape),
        #                                        trainable=True,
        #                                        dtype=tf.float32)
        #     scales = scale_to_unconstrained.inverse(scales_unconstrained)
        # else:
        #     means = tf.Variable(initial_value=means_, trainable=True, dtype=tf.float32)
        #     scales_unconstrained = tf.Variable(
        #         initial_value=scale_to_unconstrained.forward(np.linalg.cholesky(covariances_)),
        #         trainable=True, dtype=tf.float32)
        #     scales = scale_to_unconstrained.inverse(scales_unconstrained)

        self.means = tf.get_variable(name='gmm_means', shape=(M, Z), trainable=True, dtype=tf.float32)
        self.scales_unconstrained = tf.get_variable(
            name='gmm_scale',
            shape=(M, (Z*Z + Z) / 2),
            trainable=True,
            dtype=tf.float32
        )
        scales = self.scale_to_unconstrained.inverse(self.scales_unconstrained)

        covariances = tf.matmul(scales, tf.linalg.transpose(scales))
        p = tfp.distributions.MultivariateNormalTriL(
            loc=self.means,
            scale_tril=scales + tf.eye(Z, Z, batch_shape=(M,)) * 1e-5,
            validate_args=True
        )
        S_label_pattern = tfp.monte_carlo.expectation(
            f=lambda x: self.latent_clf(x),
            samples=p.sample(1000),
            log_prob=p.log_prob,
            use_reparametrization=(p.reparameterization_type == tfp.distributions.FULLY_REPARAMETERIZED)
        )

        coeffs = Bhattacharyya_coeff(z_mean_ph, z_cov, self.means, covariances)
        coeffs = tf.reshape(coeffs, [tf.shape(x_train_ph)[0], M, 1])
        coeffs_sum = tf.reduce_sum(coeffs, axis=[1, 2])
        coeffs = tf.tile(coeffs, [1, 1, L])
        # S_label_pattern = tf.reshape(S_label_pattern, [1, M, L])
        # S_label_pattern = tf.tile(S_label_pattern, [tf.shape(x_train_ph)[0], 1, 1])

        S_label_x = coeffs * S_label_pattern + (1 - coeffs) * (1 / L)

        # Combine specialized classifiers to obtained recomposed model =====================================================
        coeffs_sum = tf.reshape(coeffs_sum, [tf.shape(x_train_ph)[0], 1, 1])
        L_label_x = (coeffs / coeffs_sum) * S_label_x
        L_label_x = tf.reduce_sum(L_label_x, axis=1)

        # Construct loss function and optimizer ============================================================================
        # true_pred_ph = tf.placeholder(tf.float32, shape=[None, true_pred.shape[1]])
        # loss = tf.reduce_mean(
        #     tf.reduce_mean(tf.square(tf.log(tf.clip_by_value(L_label_x, 1e-10, 1.0)) - true_pred_ph), axis=1))
        # loss = tf.debugging.check_numerics(
        #     loss,
        #     'loss'
        # )

        # optimizer = tf.train.AdamOptimizer()
        # # opt = optimizer.minimize(loss, var_list=[scales_unconstrained, means])
        # grads_and_vars = optimizer.compute_gradients(loss, var_list=[self.scales_unconstrained, self.means])
        # # clipped_grads_and_vars = [(tf.clip_by_norm(g, 1), v) for g, v in grads_and_vars if g is not None]
        # grads_and_vars = [(tf.debugging.check_numerics(g, 'gradient'), v) for g, v in grads_and_vars]
        # opt = optimizer.apply_gradients(grads_and_vars)

        self.S_label_pattern = S_label_pattern

    def spawn(self, sess, dataset):
        """
        Run through the computational graph with a dataset
        to create an agent
        """
        # 1. Train the latent classifier
        print('Step 1...')
        dataset_string = sess.run(dataset.repeat(1000).batch(B).make_one_shot_iterator().string_handle())
        try:
            while True:
                sess.run(self.latent_train_step, feed_dict={self.handle: dataset_string})
        except tf.errors.OutOfRangeError:
            pass

        # 2. Train the GMM
        print('Step 2...')
        dataset_string = sess.run(dataset.batch(N).make_one_shot_iterator().string_handle())
        _, z_mean, _, _ = sess.run(self.iterator.get_next(), feed_dict={self.handle: dataset_string})
        gmm = GaussianMixture(n_components=M, covariance_type='full').fit(z_mean)
        means_ = gmm.means_.astype(np.float32)
        scales_ = self.scale_to_unconstrained.forward(np.linalg.cholesky(gmm.covariances_.astype(np.float32)))
        sess.run([self.means.assign(means_), self.scales_unconstrained.assign(scales_)], feed_dict={self.handle: dataset_string})

        # 3. Compute S_labels_patterns
        print('Step 3...')
        S_label_pattern_ = sess.run(self.S_label_pattern)
        patterns = (means_, gmm.covariances_.astype(np.float32))

        return Agent(patterns, S_label_pattern_)

    @staticmethod
    def fuse(agent1, agent2):
        gmm = GaussianMixture(n_components=M).fit(np.concatenate((agent1.patterns[0], agent2.patterns[0])))
        s1 = agent1.S_label_pattern
        s2 = agent2.S_label_pattern
        idx1 = gmm.predict(agent1.patterns[0])
        idx2 = gmm.predict(agent2.patterns[0])

        s = np.ones((M, 10))
        for j in range(M):
            i1 = np.argwhere(idx1 == j)[:, 0]
            for i in i1:
                s[j, :] *= s1[i, :]
            i2 = np.argwhere(idx2 == j)[:, 0]
            for i in i2:
                s[j, :] *= s2[i, :]
        normalization_const = np.sum(s, axis=1, keepdims=True)
        # normalization_const = np.reshape(normalization_const, (M, 10))
        normalization_const = np.tile(normalization_const, (1, 10))
        s /= normalization_const

        return Agent((gmm.means_.astype(np.float32), gmm.covariances_.astype(np.float32)), s)


class Agent(object):

    def __init__(self, patterns, S_label_pattern):
        self.patterns = patterns
        self.S_label_pattern = S_label_pattern


    def predict(self, x):
        z_mean, z_log_var, _ = encoder.predict(x)
        means, covariances = self.patterns
        mu1 = tf.placeholder(tf.float32, shape=(None, Z))
        sigma1 = tf.placeholder(tf.float32, shape=(None, Z))
        sigma1_transformed = tf.matrix_diag(tf.exp(sigma1 + 1e-10))
        mu2 = tf.placeholder(tf.float32, shape=(None, Z))
        sigma2 = tf.placeholder(tf.float32, shape=(None, Z, Z))
        coeffs_ph = Bhattacharyya_coeff(mu1, sigma1_transformed, mu2, sigma2)
        coeffs = sess.run(coeffs_ph, feed_dict={
            mu1: z_mean,
            sigma1: z_log_var,
            mu2: means,
            sigma2: covariances
        })
        coeffs_sum = np.sum(coeffs, axis=1)
        coeffs = coeffs[:, :, np.newaxis]
        coeffs = np.tile(coeffs, [1, 1, L])

        S_label_pattern = np.reshape(self.S_label_pattern, [1, M, L])
        S_label_pattern = np.tile(S_label_pattern, [x.shape[0], 1, 1])
        S_label_x = coeffs * S_label_pattern + (1 - coeffs) * (1 / L)

        coeffs_sum = np.reshape(coeffs_sum, [x.shape[0], 1, 1])
        L_label_x = (coeffs / coeffs_sum) * S_label_x
        L_label_x = np.sum(L_label_x, axis=1)

        return L_label_x

    def evaluate(self, x_test, y_test):
        pred = self.predict(x_test)
        pred = np.argmax(pred, axis=1)
        return np.count_nonzero(pred == y_test)


agent_factory = AgentFactory()
sess.run(tf.global_variables_initializer())
agent1 = agent_factory.spawn(sess, dataset1)
agent2 = agent_factory.spawn(sess, dataset2)
agent = AgentFactory.fuse(agent1, agent2)

vae.load_weights('mnist_vae_%d.h5' % Z)
print('Agent 1: ', agent1.evaluate(x_test, y_test))
print('Agent 2: ', agent2.evaluate(x_test, y_test))
print('Agent  : ', agent.evaluate(x_test, y_test))
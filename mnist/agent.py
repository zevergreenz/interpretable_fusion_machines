import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
from sklearn.mixture.gaussian_mixture import GaussianMixture
tfb = tfp.bijectors


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


class AgentFactory(object):
    def __init__(self, full_dataset, batch_size=1024, latent_dim=5, num_pattern=200, num_labels=10):
        """
        Objects which are shared across many agents
        (e.g. vae, ...)
        """
        self.handle = tf.placeholder(tf.string, shape=[])
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.num_pattern = num_pattern
        self.num_labels = num_labels
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, full_dataset.batch(self.batch_size).output_types,
                                                       full_dataset.batch(self.batch_size).output_shapes)
        x_train_ph, z_mean_ph, z_log_var_ph, y_train_ph = self.iterator.get_next()

        # x_train_ph = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]])
        # y_train_ph = tf.placeholder(tf.float32, shape=[None,])
        # z_mean_ph = tf.placeholder(tf.float32, shape=[None, z_train.shape[1]])
        # z_log_var_ph = tf.placeholder(tf.float32, shape=[None, z_log_var_train.shape[1]])
        z_cov = tf.matrix_diag(tf.exp(z_log_var_ph + 1e-10))

        # Train the latent classifier ==================================================================================
        print('Training latent classifier...')
        self.latent_clf = keras.Sequential([
            keras.layers.InputLayer(input_tensor=z_mean_ph, input_shape=(self.latent_dim,)),
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

        self.means = tf.get_variable(name='gmm_means', shape=(self.num_pattern, self.latent_dim), trainable=True, dtype=tf.float32)
        self.scales_unconstrained = tf.get_variable(
            name='gmm_scale',
            shape=(self.num_pattern, (self.latent_dim*self.latent_dim + self.latent_dim) / 2),
            trainable=True,
            dtype=tf.float32
        )
        scales = self.scale_to_unconstrained.inverse(self.scales_unconstrained)

        covariances = tf.matmul(scales, tf.linalg.transpose(scales))
        p = tfp.distributions.MultivariateNormalTriL(
            loc=self.means,
            scale_tril=scales + tf.eye(self.latent_dim, self.latent_dim, batch_shape=(self.num_pattern,)) * 1e-5,
            validate_args=True
        )
        S_label_pattern = tfp.monte_carlo.expectation(
            f=lambda x: self.latent_clf(x),
            samples=p.sample(1000),
            log_prob=p.log_prob,
            use_reparametrization=(p.reparameterization_type == tfp.distributions.FULLY_REPARAMETERIZED)
        )

        coeffs = Bhattacharyya_coeff(z_mean_ph, z_cov, self.means, covariances)
        coeffs = tf.reshape(coeffs, [tf.shape(x_train_ph)[0], self.num_pattern, 1])
        coeffs_sum = tf.reduce_sum(coeffs, axis=[1, 2])
        coeffs = tf.tile(coeffs, [1, 1, self.num_labels])
        # S_label_pattern = tf.reshape(S_label_pattern, [1, self.num_pattern, self.num_labels])
        # S_label_pattern = tf.tile(S_label_pattern, [tf.shape(x_train_ph)[0], 1, 1])

        S_label_x = coeffs * S_label_pattern + (1 - coeffs) * (1 / self.num_labels)

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

    def spawn(self, sess, dataset, num_data=None):
        """
        Run through the computational graph with a dataset
        to create an agent
        """
        # 1. Train the latent classifier
        print('Step 1...')
        dataset_string = sess.run(dataset.repeat(1000).batch(self.batch_size).make_one_shot_iterator().string_handle())
        try:
            while True:
                sess.run(self.latent_train_step, feed_dict={self.handle: dataset_string})
        except tf.errors.OutOfRangeError:
            pass

        # 2. Train the GMM
        print('Step 2...')
        dataset_string = sess.run(dataset.batch(num_data).make_one_shot_iterator().string_handle())
        _, z_mean, _, _ = sess.run(self.iterator.get_next(), feed_dict={self.handle: dataset_string})
        gmm = GaussianMixture(n_components=self.num_pattern, covariance_type='full').fit(z_mean)
        means_ = gmm.means_.astype(np.float32)
        scales_ = self.scale_to_unconstrained.forward(np.linalg.cholesky(gmm.covariances_.astype(np.float32)))
        sess.run([self.means.assign(means_), self.scales_unconstrained.assign(scales_)], feed_dict={self.handle: dataset_string})

        # 3. Compute S_labels_patterns
        print('Step 3...')
        S_label_pattern_ = sess.run(self.S_label_pattern)
        patterns = (means_, gmm.covariances_.astype(np.float32))

        return Agent(sess, patterns, S_label_pattern_)

    def fuse(self, agent1, agent2):
        gmm = GaussianMixture(n_components=self.num_pattern).fit(np.concatenate((agent1.patterns[0], agent2.patterns[0])))
        s1 = agent1.S_label_pattern
        s2 = agent2.S_label_pattern
        idx1 = gmm.predict(agent1.patterns[0])
        idx2 = gmm.predict(agent2.patterns[0])

        s = np.ones((self.num_pattern, 10))
        for j in range(self.num_pattern):
            i1 = np.argwhere(idx1 == j)[:, 0]
            for i in i1:
                s[j, :] *= s1[i, :]
            i2 = np.argwhere(idx2 == j)[:, 0]
            for i in i2:
                s[j, :] *= s2[i, :]
        normalization_const = np.sum(s, axis=1, keepdims=True)
        # normalization_const = np.reshape(normalization_const, (self.num_pattern, 10))
        normalization_const = np.tile(normalization_const, (1, 10))
        s /= normalization_const

        return Agent(agent1.sess, (gmm.means_.astype(np.float32), gmm.covariances_.astype(np.float32)), s)


class Agent(object):

    def __init__(self, sess, patterns, S_label_pattern):
        self.sess = sess
        self.patterns = patterns
        self.S_label_pattern = S_label_pattern
        self.num_pattern = patterns[0].shape[0]
        self.latent_dim = patterns[0].shape[1]
        self.num_labels = S_label_pattern.shape[1]


    def predict(self, z_mean, z_log_var):
        # z_mean, z_log_var, _ = encoder.predict(x)
        means, covariances = self.patterns
        mu1 = tf.placeholder(tf.float32, shape=(None, self.latent_dim))
        sigma1 = tf.placeholder(tf.float32, shape=(None, self.latent_dim))
        sigma1_transformed = tf.matrix_diag(tf.exp(sigma1 + 1e-10))
        mu2 = tf.placeholder(tf.float32, shape=(None, self.latent_dim))
        sigma2 = tf.placeholder(tf.float32, shape=(None, self.latent_dim, self.latent_dim))
        coeffs_ph = Bhattacharyya_coeff(mu1, sigma1_transformed, mu2, sigma2)
        coeffs = self.sess.run(coeffs_ph, feed_dict={
            mu1: z_mean,
            sigma1: z_log_var,
            mu2: means,
            sigma2: covariances
        })
        coeffs_sum = np.sum(coeffs, axis=1)
        coeffs = coeffs[:, :, np.newaxis]
        coeffs = np.tile(coeffs, [1, 1, self.num_labels])

        S_label_pattern = np.reshape(self.S_label_pattern, [1, self.num_pattern, self.num_labels])
        S_label_pattern = np.tile(S_label_pattern, [z_mean.shape[0], 1, 1])
        S_label_x = coeffs * S_label_pattern + (1 - coeffs) * (1 / self.num_labels)

        coeffs_sum = np.reshape(coeffs_sum, [z_mean.shape[0], 1, 1])
        L_label_x = (coeffs / coeffs_sum) * S_label_x
        L_label_x = np.sum(L_label_x, axis=1)

        return L_label_x

    def evaluate(self, z_mean, z_log_var, y):
        pred = self.predict(z_mean, z_log_var)
        pred = np.argmax(pred, axis=1)
        return np.count_nonzero(pred == y)
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
os.environ["CUDA_VISIBLE_DEVICES"]="3"  # specify which GPU(s) to be used
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings


def Bhattacharyya_coeff(mu1, sigma1, mu2, sigma2):
    N = tf.shape(mu1)[0]
    M = mu2.shape[0]
    Z = mu1.shape[1]
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
        iterator = tf.data.Iterator.from_string_handle(self.handle, full_dataset.batch(B).output_types,
                                                       full_dataset.batch(B).output_shapes)
        x_train_ph, z_mean_ph, z_log_var_ph, y_train_ph = iterator.get_next()

        # x_train_ph = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]])
        # y_train_ph = tf.placeholder(tf.float32, shape=[None,])
        # z_mean_ph = tf.placeholder(tf.float32, shape=[None, z_train.shape[1]])
        # z_log_var_ph = tf.placeholder(tf.float32, shape=[None, z_log_var_train.shape[1]])
        z_cov = tf.matrix_diag(tf.exp(z_log_var_ph + 1e-10))

        self.gmm = gmm_lib.GMM(num_clusters=M, covariance_type='full')
        # gmm.fit(x=z_mean_ph, y=None)
        self.gmm_train_step = self.gmm._get_train_ops(z_mean_ph, None)
        means_ = self.gmm.clusters()
        covariances_ = self.gmm.covariances()

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
        scale_to_unconstrained = tfb.Chain([
            # step 3: flatten the lower triangular portion of the matrix
            tfb.Invert(tfb.FillTriangular(validate_args=True)),
            # step 2: take the log of the diagonals
            tfb.TransformDiagonal(tfb.Invert(tfb.Exp(validate_args=True))),
            # # step 1: decompose the precision matrix into its Cholesky factors
            # tfb.Invert(tfb.CholeskyOuterProduct(validate_args=True)),
        ])

        random_init = False
        if random_init:
            means = tf.Variable(initial_value=tf.random_uniform(means_.shape), trainable=True, dtype=tf.float32)
            scales_unconstrained = tf.Variable(
                initial_value=scale_to_unconstrained.forward(np.linalg.cholesky(covariances_)),
                trainable=True, dtype=tf.float32)
            scales_unconstrained = tf.Variable(initial_value=tf.random_uniform(scales_unconstrained.shape),
                                               trainable=True,
                                               dtype=tf.float32)
            scales = scale_to_unconstrained.inverse(scales_unconstrained)
        else:
            means = tf.Variable(initial_value=means_, trainable=True, dtype=tf.float32)
            scales_unconstrained = tf.Variable(
                initial_value=scale_to_unconstrained.forward(np.linalg.cholesky(covariances_)),
                trainable=True, dtype=tf.float32)
            scales = scale_to_unconstrained.inverse(scales_unconstrained)

        covariances = tf.matmul(scales, tf.linalg.transpose(scales))
        p = tfp.distributions.MultivariateNormalTriL(
            loc=means,
            scale_tril=scales + tf.eye(Z, Z, batch_shape=(M,)) * 1e-5,
            validate_args=True
        )
        S_label_pattern = tfp.monte_carlo.expectation(
            f=lambda x: self.latent_clf(x),
            samples=p.sample(1000),
            log_prob=p.log_prob,
            use_reparametrization=(p.reparameterization_type == tfp.distributions.FULLY_REPARAMETERIZED)
        )

        coeffs = Bhattacharyya_coeff(z_mean_ph, z_cov, means, covariances)
        coeffs = tf.reshape(coeffs, [tf.shape(x_train_ph)[0], M, 1])
        coeffs_sum = tf.reduce_sum(coeffs, axis=[1, 2])
        coeffs = tf.tile(coeffs, [1, 1, L])
        S_label_pattern = tf.reshape(S_label_pattern, [1, M, L])
        S_label_pattern = tf.tile(S_label_pattern, [tf.shape(x_train_ph)[0], 1, 1])

        S_label_x = coeffs * S_label_pattern + (1 - coeffs) * (1 / L)

        # Combine specialized classifiers to obtained recomposed model =====================================================
        coeffs_sum = tf.reshape(coeffs_sum, [tf.shape(x_train_ph)[0], 1, 1])
        L_label_x = (coeffs / coeffs_sum) * S_label_x
        L_label_x = tf.reduce_sum(L_label_x, axis=1)

        # Construct loss function and optimizer ============================================================================
        true_pred_ph = tf.placeholder(tf.float32, shape=[None, true_pred.shape[1]])
        loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(tf.log(tf.clip_by_value(L_label_x, 1e-10, 1.0)) - true_pred_ph), axis=1))
        loss = tf.debugging.check_numerics(
            loss,
            'loss'
        )

        optimizer = tf.train.AdamOptimizer()
        # opt = optimizer.minimize(loss, var_list=[scales_unconstrained, means])
        grads_and_vars = optimizer.compute_gradients(loss, var_list=[scales_unconstrained, means])
        # clipped_grads_and_vars = [(tf.clip_by_norm(g, 1), v) for g, v in grads_and_vars if g is not None]
        grads_and_vars = [(tf.debugging.check_numerics(g, 'gradient'), v) for g, v in grads_and_vars]
        opt = optimizer.apply_gradients(grads_and_vars)

        self.S_label_pattern = S_label_pattern


    def spawn(self, sess, dataset):
        """
        Run through the computational graph with a dataset
        to create an agent
        :param sess:
        :param x:
        :param y:
        :return:
        """
        # 1. Train the latent classifier
        dataset_string = sess.run(dataset.make_one_shot_iterator().string_handle())
        try:
            while True:
                sess.run(self.latent_train_step, feed_dict={self.handle: dataset_string})
        except tf.errors.OutOfRangeError:
            pass

        # 2. Train the GMM
        try:
            while True:
                sess.run(self.gmm_train_step, feed_dict={self.handle: dataset_string})
        except tf.errors.OutOfRangeError:
            pass

        # 3. Compute S_labels_patterns
        try:
            S_label_pattern_ = sess.run(self.S_label_pattern, feed_dict={self.handle: dataset_string})
            while True:
                s = sess.run(self.S_label_pattern, feed_dict={self.handle: dataset_string})
                S_label_pattern_ = np.concatenate((S_label_pattern_, s))
        except tf.errors.OutOfRangeError:
            pass

        patterns = sess.run(self.gmm.clusters())

        return Agent(patterns, S_label_pattern_)


class Agent(object):
    def __init__(self, patterns, S_label_pattern):
        self.patterns = patterns
        self.S_label_pattern = S_label_pattern

    def predict(self, x):
        pass

    def evaluate(self, dataset):
        pass


class SpecializedModel(object):
    def __init__(self):
        self.centroids = None
        self.labels_centroids = None


class FusionMachine(object):
    def __init__(self):
        pass


agent_factory = AgentFactory()
agent1 = agent_factory.spawn(sess)
agent2 = agent_factory.spawn(sess)




print('Training GMM model...')
gmm1 = GaussianMixture(n_components=M, covariance_type='full').fit(z_train[indices1])
gmm2 = GaussianMixture(n_components=M, covariance_type='full').fit(z_train[indices2])
gmm_combine = GaussianMixture(n_components=M, covariance_type='full').fit(np.concatenate((gmm1.means_, gmm2.means_)))
# means_, covariances_ = gmm_combine.means_.astype(np.float32), gmm.covariances_.astype(np.float32)

# x_means_ = decoder.predict(means_)
# for i in range(L):
#     plt.clf()
#     plt.imshow(np.reshape(x_means_[i], (28, 28)))
#     plt.savefig('new_mean_%d.png' % i, dpi=300)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


# Tensorflow session ========================================================================================
sess.run(tf.global_variables_initializer())


full_dataset_string = sess.run(dataset1.repeat(E).batch(B).make_one_shot_iterator().string_handle())
try:
    while True:
        sess.run(latent_train_step, feed_dict={handle: full_dataset_string})
except tf.errors.OutOfRangeError:
    pass
sess.run([means.assign(gmm1.means_.astype(np.float32)), scales_unconstrained.assign(scale_to_unconstrained.forward(np.linalg.cholesky(gmm1.covariances_.astype(np.float32))))])
dataset_string = sess.run(test_dataset.batch(B).make_one_shot_iterator().string_handle())
s1 = sess.run(S_label_pattern, feed_dict={handle: dataset_string})

full_dataset_string = sess.run(dataset2.repeat(E).batch(B).make_one_shot_iterator().string_handle())
try:
    while True:
        sess.run(latent_train_step, feed_dict={handle: full_dataset_string})
except tf.errors.OutOfRangeError:
    pass
sess.run([means.assign(gmm2.means_.astype(np.float32)), scales_unconstrained.assign(scale_to_unconstrained.forward(np.linalg.cholesky(gmm2.covariances_.astype(np.float32))))])
dataset_string = sess.run(test_dataset.batch(B).make_one_shot_iterator().string_handle())
s2 = sess.run(S_label_pattern, feed_dict={handle: dataset_string})

idx1 = gmm_combine.predict(gmm1.means_)
idx2 = gmm_combine.predict(gmm2.means_)
s = np.ones((s1.shape[0], M, 10))
for j in range(M):
    i1 = np.argwhere(idx1 == j)[:, 0]
    for i in i1:
        s[:, j, :] *= s1[:, i, :]
    i2 = np.argwhere(idx2 == j)[:, 0]
    for i in i2:
        s[:, j, :] *= s2[:, i, :]
normalization_const = np.sum(s, axis=1)
normalization_const = np.reshape(normalization_const, (B, 1, 10))
normalization_const = np.tile(normalization_const, (1, M, 1))
s /= normalization_const

sess.run([means.assign(gmm_combine.means_.astype(np.float32)), scales_unconstrained.assign(scale_to_unconstrained.forward(np.linalg.cholesky(gmm_combine.covariances_.astype(np.float32))))])
dataset_string = sess.run(test_dataset.batch(B).make_one_shot_iterator().string_handle())
coeffs_, coeffs_sum_ = sess.run([coeffs, coeffs_sum], feed_dict={handle: dataset_string})
S_label_x_ = coeffs_ * s + (1 - coeffs_) * (1 / L)
L_label_x_ = (coeffs_ / coeffs_sum_) * S_label_x_
L_label_x_ = np.sum(L_label_x_, axis=1)

full_dataset_string = sess.run(dataset1.repeat(E).batch(B).make_one_shot_iterator().string_handle())
try:
    while True:
        sess.run(latent_train_step, feed_dict={handle: full_dataset_string})
except tf.errors.OutOfRangeError:
    pass

full_dataset_string = sess.run(dataset1.batch(B).make_one_shot_iterator().string_handle())
acc = 0
count = 0
try:
    while True:
        pred, ground_truth = sess.run([latent_clf(z_mean_ph), y_train_ph], feed_dict={handle: full_dataset_string})
        pred = np.argmax(pred, axis=1)
        acc += float(np.count_nonzero(pred == ground_truth))
        count += ground_truth.shape[0]
except tf.errors.OutOfRangeError:
    acc /= count
    print('Latent model train accuracy: ', acc)
test_dataset_string = sess.run(test_dataset.batch(B).make_one_shot_iterator().string_handle())
acc = 0
count = 0
try:
    while True:
        pred, ground_truth = sess.run([latent_clf(z_mean_ph), y_train_ph], feed_dict={handle: test_dataset_string})
        pred = np.argmax(pred, axis=1)
        acc += float(np.count_nonzero(pred == ground_truth))
        count += ground_truth.shape[0]
except tf.errors.OutOfRangeError:
    acc /= count
    print('Latent model test accuracy: ', acc)


dataset1_string = sess.run(dataset1.batch(B).make_one_shot_iterator().string_handle())
acc = 0
try:
    while True:
        pred, ground_truth = sess.run([L_label_x, y_train_ph], feed_dict={handle: dataset1_string})
        pred = np.argmax(pred, axis=1)
        acc += float(np.count_nonzero(pred == ground_truth))
except tf.errors.OutOfRangeError:
    acc /= y_train.shape[0]
    print('Recomposed model train accuracy: ', acc)
test_dataset_string = sess.run(test_dataset.batch(B).make_one_shot_iterator().string_handle())
acc = 0
try:
    while True:
        pred, ground_truth = sess.run([L_label_x, y_train_ph], feed_dict={handle: test_dataset_string})
        pred = np.argmax(pred, axis=1)
        acc += float(np.count_nonzero(pred == ground_truth))
except tf.errors.OutOfRangeError:
    acc /= y_test.shape[0]
    print('Recomposed model test accuracy: ', acc)


# Evaluation ================================================================================================
# full_dataset_string = sess.run(full_dataset.batch(B).make_one_shot_iterator().string_handle())
# acc = 0
# try:
#     while True:
#         pred, ground_truth = sess.run([L_label_x, y_train_ph], feed_dict={handle: full_dataset_string})
#         pred = np.argmax(pred, axis=1)
#         acc += float(np.count_nonzero(pred == ground_truth))
# except tf.errors.OutOfRangeError:
#     acc /= y_train.shape[0]
#     print('Recomposed model train accuracy: ', acc)
# test_dataset_string = sess.run(test_dataset.batch(B).make_one_shot_iterator().string_handle())
# acc = 0
# try:
#     while True:
#         pred, ground_truth = sess.run([L_label_x, y_train_ph], feed_dict={handle: test_dataset_string})
#         pred = np.argmax(pred, axis=1)
#         acc += float(np.count_nonzero(pred == ground_truth))
# except tf.errors.OutOfRangeError:
#     acc /= y_test.shape[0]
#     print('Recomposed model test accuracy: ', acc)

# scales_grads = []
# means_grads = []
# for j in range(0):
#     loss_ = 0
#     for i in range(0, N, B):
#         _, loss_i, grads_and_vars_ = sess.run([opt, loss, grads_and_vars], feed_dict={
#             x_train_ph: x_train[i:i + B],
#             y_train_ph: y_train[i:i + B],
#             z_mean_ph: z_train[i:i + B],
#             z_log_var_ph: z_log_var_train[i:i + B],
#             true_pred_ph: true_pred[i:i + B]
#         })
#         loss_ += loss_i
#         scales_grads.append(grads_and_vars_[0])
#         means_grads.append(grads_and_vars_[1])
#     print(j, loss_, np.sum(grads_and_vars_[0][0]), np.sum(grads_and_vars_[1][0]))


# print("Loss 2: ", sess.run(loss, feed_dict=feed_dict))
# Evaluation ================================================================================================
# acc = 0
# for i in range(0, N, B):
#     pred = sess.run(L_label_x, feed_dict={
#         x_train_ph: x_train[i:i + B],
#         y_train_ph: y_train[i:i + B],
#         z_mean_ph: z_train[i:i + B],
#         z_log_var_ph: z_log_var_train[i:i + B],
#         true_pred_ph: true_pred[i:i + B]
#     })
#     pred = np.argmax(pred, axis=1)
#     acc += float(np.count_nonzero(pred == y_train[i:i + B]))
# acc /= y_train.shape[0]
# print('Recomposed model accuracy: ', acc)
# acc = 0
# for i in range(0, x_test.shape[0], B):
#     pred = sess.run(L_label_x, feed_dict={
#         x_train_ph: x_test[i:i + B],
#         y_train_ph: y_test[i:i + B],
#         z_mean_ph: z_test[i:i + B],
#         z_log_var_ph: z_log_var_test[i:i + B],
#     })
#     pred = np.argmax(pred, axis=1)
#     acc += float(np.count_nonzero(pred == y_test[i:i + B]))
# acc /= y_test.shape[0]
# print('Recomposed model accuracy: ', acc)


# Interprete the results ==============================================================================================
# encoder, decoder, vae = train_vae(x_train, y_train, latent_dim=Z, weights='mnist_vae_%d.h5' % Z)
# # First, let us reduce to L centroids and visualize them
# gmm2 = GaussianMixture(n_components=L)
# clustering = gmm2.fit_predict(means_)
#
# # display a 10x10 2D manifold of digits
# n = 10
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # linearly spaced coordinates corresponding to the 2D plot
# # of digit classes in the latent space
# grid_x = np.linspace(-4, 4, n)
# grid_y = np.linspace(-4, 4, n)[::-1]
#
# idx = 0
# for i, yi in enumerate(grid_y):
#     indices = np.where(clustering == i)[0]
#     for j, xi in enumerate(grid_x):
#         if j >= len(indices):
#             break
#         elif j == 0:
#             digit = decoder.predict(gmm2.means_[i:i+1]).reshape(digit_size, digit_size)
#         else:
#             digit = decoder.predict(means_[indices[j]:indices[j]+1]).reshape(digit_size, digit_size)
#         idx += 1
#         figure[i * digit_size: (i + 1) * digit_size,
#         j * digit_size: (j + 1) * digit_size] = digit
#
# plt.figure(figsize=(10, 10))
# start_range = digit_size // 2
# end_range = n * digit_size + start_range + 1
# pixel_range = np.arange(start_range, end_range, digit_size)
# sample_range_x = np.round(grid_x, 1)
# sample_range_y = np.round(grid_y, 1)
# plt.xticks(pixel_range, sample_range_x)
# plt.yticks(pixel_range, sample_range_y)
# # plt.xlabel("z[0]")
# # plt.ylabel("z[1]")
# plt.imshow(figure)
# plt.savefig('temp.png', dpi=300)
#
#
# indices = np.random.choice(x_test.shape[0], B, replace=False)
# x_test_sample = x_test[indices, :]
# y_test_sample = y_test[indices]
# z_test_sample, z_log_var_test_sample, _ = encoder.predict(x_test_sample)
# # z_log_var_test_sample = z_log_var_test[indices, :]
# coeffs_test_sample = sess.run(coeffs, feed_dict={
#     x_train_ph: x_test_sample,
#     y_train_ph: y_test_sample,
#     z_mean_ph: z_test_sample,
#     z_log_var_ph: z_log_var_test_sample
# })
# i = 0
# coeffs_train = sess.run(coeffs, feed_dict={
#     x_train_ph: x_train[i:i + B],
#     y_train_ph: y_train[i:i + B],
#     z_mean_ph: z_train[i:i + B],
#     z_log_var_ph: z_log_var_train[i:i + B],
#     true_pred_ph: true_pred[i:i + B]
# })
# for i in range(B, N, B):
#     coeffs_train = np.concatenate([
#         coeffs_train,
#         sess.run(coeffs, feed_dict={
#             x_train_ph: x_train[i:i + B],
#             y_train_ph: y_train[i:i + B],
#             z_mean_ph: z_train[i:i + B],
#             z_log_var_ph: z_log_var_train[i:i + B],
#             true_pred_ph: true_pred[i:i + B]
#         })
#     ], axis=0)
#
#
# # display a 10x10 2D manifold of digits
# n = 10
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # linearly spaced coordinates corresponding to the 2D plot
# # of digit classes in the latent space
# grid_x = np.linspace(-4, 4, n)
# grid_y = np.linspace(-4, 4, n)[::-1]
#
# idx = 0
# for i, yi in enumerate(grid_y):
#     pattern_idx = np.argmax(coeffs_test_sample[i, :, 0])
#     new_pattern_idx = clustering[pattern_idx]
#     similar_idx = ((coeffs_train[:, new_pattern_idx, 0]).argsort())[::-1]
#     for j, xi in enumerate(grid_x):
#         if j >= len(indices):
#             break
#         elif j == 0:
#             digit = x_test_sample[i:i+1].reshape(digit_size, digit_size)
#         elif j == 1:
#             digit = decoder.predict(gmm2.means_[new_pattern_idx:new_pattern_idx+1]).reshape(digit_size, digit_size)
#         else:
#             digit = x_train[similar_idx[j-2]].reshape(digit_size, digit_size)
#         idx += 1
#         figure[i * digit_size: (i + 1) * digit_size,
#         j * digit_size: (j + 1) * digit_size] = digit
#
# plt.figure(figsize=(10, 10))
# start_range = digit_size // 2
# end_range = n * digit_size + start_range + 1
# pixel_range = np.arange(start_range, end_range, digit_size)
# sample_range_x = np.round(grid_x, 1)
# sample_range_y = np.round(grid_y, 1)
# plt.xticks(pixel_range, sample_range_x)
# plt.yticks(pixel_range, sample_range_y)
# plt.imshow(figure)
# plt.savefig('temp2.png', dpi=300)
import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from cifar.blackbox import train_blackbox
from cifar.vae import train_cnn_vae
from agent import AgentFactory

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings


# CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Build a black-box model and get its predictions
black_box_model = train_blackbox()

latent_dim = 1024
num_pattern = 10
N = x_train.shape[0]
M = num_pattern
L = 10
D = 32*32*3
Z = latent_dim
B = 64
# E = 1000

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

# original_dim = np.prod(x_train.shape[1:])
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])

encoder, decoder, _ = train_cnn_vae()
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


agent_factory = AgentFactory(full_dataset, latent_dim=Z, num_pattern=200, num_labels=10)
sess.run(tf.global_variables_initializer())
agent1 = agent_factory.spawn(sess, dataset1, num_data=len(indices1))
agent2 = agent_factory.spawn(sess, dataset2, num_data=len(indices2))
agent = agent_factory.fuse(agent1, agent2)

print('Agent 1: ', agent1.evaluate(z_test, z_log_var_test, y_test))
print('Agent 2: ', agent2.evaluate(z_test, z_log_var_test, y_test))
print('Agent  : ', agent.evaluate(z_test, z_log_var_test, y_test))
import h5py
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# Parameter to change
num_datasets = 10
dataset_size = 40
epochs = 500
f = h5py.File('uci_dataset.hdf5', 'w')

# MNIST dataset
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y)

#Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
# x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

f.create_dataset('train/x', data=x_train)
f.create_dataset('train/y', data=y_train)
f.create_dataset('test/x', data=x_test)
f.create_dataset('test/y', data=y_test)


def random_split():
    global x_train, y_train
    idx = np.arange(0, x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]
    for dataset_i in range(num_datasets):
        start = dataset_i * dataset_size
        end = start + dataset_size
        f.create_dataset("train_%d/x" % dataset_i, data=x_train[start:end])
        f.create_dataset("train_%d/y" % dataset_i, data=y_train[start:end])

random_split()
f.close()

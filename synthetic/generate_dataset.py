import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Parameter to change
x_dim = 5
y_dim = 3
num_data = 60000
num_datasets = 10
dataset_size = 400
f = h5py.File('synthetic_dataset.hdf5', 'w')

# Generate synthetic dataset
x = torch.rand((num_data, x_dim))
alpha = torch.rand((y_dim, x_dim))
distances = torch.pow(x.unsqueeze(1) - alpha.unsqueeze(0), 2).sum(-1)
y = torch.distributions.Categorical(distances).sample().view((num_data, 1))

x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy())

f.create_dataset('train/x', data=x_train)
f.create_dataset('train/y', data=y_train)
f.create_dataset('test/x', data=x_test)
f.create_dataset('test/y', data=y_test)
f.create_dataset('alpha', data=alpha.numpy())


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

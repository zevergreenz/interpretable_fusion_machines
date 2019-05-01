import numpy as np

latent_dims = [1, 2, 3, 5, 10, 15, 20, 30]
latent_accs = []
recomposed_accs = []
num_components = []

for latent_dim in latent_dims:
    result = np.load('results_%d.npy' % latent_dim)
    latent_accs.append(np.average(result, axis=0)[2])
    recomposed_accs.append(np.max(result, axis=0)[3])
    num_components.append(np.argmax(result, axis=0)[3])
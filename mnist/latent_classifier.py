import numpy as np


def generate_latent_dataset(encoder, x_train, y_train, x_test, y_test, name='latent_dataset.npy'):
    z_train, z_log_var_train, _ = encoder.predict(x_train)
    z_test, z_log_var_test, _ = encoder.predict(x_test)
    np.save(name, [z_train, z_log_var_train, y_train, z_test, z_log_var_test, y_test])
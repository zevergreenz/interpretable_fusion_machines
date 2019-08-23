import GPy
import time
from gaus_marginal_matching import match_local_atoms
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
from GPy.inference.latent_function_inference.posterior import Posterior
#new_posterior = Posterior(mean = variational_mean, cov = variational_cov)

np.random.seed(101)

def generate_partitioned_data():
    X_partition = []
    Y_partition = []
    X_test = []
    Y_test = []
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Partition 1
    x = np.linspace(0.5, 5, 100)[:, np.newaxis]
    y = x.ravel() + np.sin(x.ravel()) + 0.2 * np.random.randn(100)

    x_test = np.random.uniform(0, 1, 20)[:, np.newaxis]
    y_test = x_test.ravel() + np.sin(x_test.ravel()) + 0.2 * np.random.randn(20)

    axs[0].plot(x.ravel(), y, 'b*-')
    axs[0].plot(x_test.ravel(), y_test, 'k*')

    # Store first partition
    X_partition.append(x)
    Y_partition.append(y[:, np.newaxis])
    X_test.append(x_test)
    Y_test.append(y_test[:, np.newaxis])

    # Partition 2
    x = np.linspace(4.5, 9, 100)[:, np.newaxis]
    y = x.ravel() + np.sin(x.ravel()) + 0.2 * np.random.randn(100)

    x_test = np.random.uniform(8.5, 9.5, 20)[:, np.newaxis]
    y_test = x_test.ravel() + np.sin(x_test.ravel()) + 0.2 * np.random.randn(20)

    axs[1].plot(x.ravel(), y, 'b*-')
    axs[1].plot(x_test.ravel(), y_test, 'k*')

    # Store first partition
    X_partition.append(x)
    Y_partition.append(y[:, np.newaxis])
    X_test.append(x_test)
    Y_test.append(y_test[:, np.newaxis])

    plt.tight_layout()
    #plt.show(block=True)

    return X_partition, Y_partition, X_test, Y_test

def construct_local_sgps(X_partition, Y_partition):
    print('Constructing local SGPs ...')
    local_gps = []
    for i in range(len(X_partition)):
        print('Constructing SGP no. %d' % (i + 1))
        P = int(np.sqrt(X_partition[i].shape[0]))  # no. of inducing point = sqrt(no. of data points)
        Z = np.random.rand(P, X_partition[i].shape[1]) * 1.0  # randomly initialize inducing points
        m_gp = GPy.models.SparseGPRegression(X_partition[i], Y_partition[i], Z=Z) # compile a SGP
        m_gp.optimize('bfgs') # optimize it
        local_gps.append(m_gp)
    return local_gps # return the list of SGP

def extract_inducing_point(sgp, l):
    v = sgp.inducing_inputs[l, :].reshape((sgp.inducing_inputs.shape[1], 1))
    return v

def rmse(truth, prediction):
    #print("Hello")
    diff = truth[:, 0] - prediction[:, 0]
    err = (np.dot(diff, diff) * 1.0 / truth.shape[0]) ** 0.5
    return err

def deep_fusion(sgps):  # generate q(u | D1, D2, ..., Dm) from q(u | D1) ... q(u | Dm)
    new_sgps = []
    F1 = np.zeros(sgps[0].posterior.covariance.shape)
    F2 = np.zeros(sgps[0].posterior.mean.shape)

    print(sgps[0].posterior.mean.shape)

    for sgp in sgps:
        mean = sgp.posterior.mean
        cov = sgp.posterior.covariance
        E1 = np.linalg.inv(cov)
        E2 = np.dot(E1, mean)
        F1 += E1
        F2 += E2

    fused_cov = np.linalg.inv(F1)
    fused_mean = np.dot(fused_cov, F2)

    for sgp in sgps:
        sgp.posterior._mean = fused_mean
        sgp.posterior._covariance = fused_cov
        new_sgps.append(sgp)
    return new_sgps

def fusion(sgps, X_partition, Y_partition, reopt=False):
    atoms = [sgps[i].inducing_inputs for i in range(len(sgps))]

    print(atoms[0].shape)

    est_atoms, popularity_counts = match_local_atoms(local_atoms=atoms, sigma=5., sigma0=5., gamma=10., it=20,
                                                     optimize_hyper=True)
    print('Updating local models with new inducing points ...')
    new_sgps = [GPy.models.SparseGPRegression(X_partition[i], Y_partition[i], Z=est_atoms) for i in range(len(sgps))]

    for i in range(len(new_sgps)):
        plt.figure()
        new_sgps[i].plot(xlim=[0,10])
        #plt.show()

    if reopt:
        for i in range(len(sgps)):
            print('Re-optimizing local model no. %d' % (i + 1))
            sgps[i].inducing_inputs.fix()
            sgps[i].optimize('bfgs')  # re-optimize the inducing atoms

    print('Performing deeper fusion ...')
    new_sgps = deep_fusion(new_sgps)

    for i in range(len(new_sgps)):
        #sgps[i].plot()
        plt.figure()
        new_sgps[i].plot(xlim=[0,10])
        #plt.show()

    pool_atom = atoms[0]
    for i in range(1, len(atoms)):
        pool_atom = np.concatenate((pool_atom, atoms[i]), axis=0)
    return new_sgps, est_atoms, pool_atom

def kmean_baseline(X_train, Y_train, X_test, Y_test, pool_atoms, n_est_atoms):
    print('Evaluating K-mean baseline ...')

    X_test_all = X_test[0]
    Y_test_all = Y_test[0]
    for i in range(1, len(X_test)):
        X_test_all = np.concatenate((X_test_all, X_test[i]), axis = 0)
        Y_test_all = np.concatenate((Y_test_all, Y_test[i]), axis = 0)

    start = time.time()
    U = kmeans(whiten(pool_atoms), n_est_atoms)[0]
    kmean_model = GPy.models.SparseGPRegression(X_train, Y_train, Z=U)
    kmean_rmse = rmse(Y_test_all, kmean_model.predict(X_test_all)[0])
    print('K-mean baseline performance : %f' % kmean_rmse)
    end = time.time()
    kmean_time = end - start
    print('K-mean baseline time : %f' % kmean_time)
    return kmean_rmse, kmean_time

def centralized_gp_baseline(X_train, Y_train, X_test, Y_test, est_atoms):
    print('Evaluating centralized GP baseline with centralized training data + optimizing inducing points ...')
    X_test_all = X_test[0]
    Y_test_all = Y_test[0]
    for i in range(1, len(X_test)):
        X_test_all = np.concatenate((X_test_all, X_test[i]), axis=0)
        Y_test_all = np.concatenate((Y_test_all, Y_test[i]), axis=0)
    start = time.time()
    P = est_atoms.shape[0]
    U = np.random.rand(P, X_train.shape[1]) * 1.0
    centralized_baseline = GPy.models.SparseGPRegression(X_train, Y_train, Z=U)
    centralized_baseline.optimize('bfgs')
    centralized_rmse = rmse(Y_test_all, centralized_baseline.predict(X_test_all)[0])
    print('Centralized GP baseline performance : %f' % centralized_rmse)
    end = time.time()
    centralized_time = end - start
    print('Centralized GP baseline time : %f' % centralized_time)
    return centralized_rmse, centralized_time

def fuse_gp(X_train, Y_train, X_test, Y_test, est_atoms, overhead=0):
    print('Evaluating fused GP model (using fused inducing points) with centralized training data ...')
    X_test_all = X_test[0]
    Y_test_all = Y_test[0]
    for i in range(1, len(X_test)):
        X_test_all = np.concatenate((X_test_all, X_test[i]), axis=0)
        Y_test_all = np.concatenate((Y_test_all, Y_test[i]), axis=0)
    start = time.time()
    fused_model = GPy.models.SparseGPRegression(X_train, Y_train, Z=est_atoms)

    plt.figure()
    fused_model.plot()
    #plt.show()


    fused_rmse = rmse(Y_test_all, fused_model.predict(X_test_all)[0])
    print('Fused GP model performance : %f' % fused_rmse)
    end = time.time()
    fused_time = overhead + (end - start)
    print('Fused GP model time : %f' % fused_time)
    return fused_rmse, fused_time

def local_gp_evaluation(sgps, new_sgps, X_test, Y_test):
    print('Evaluating local GP models with and without using the fused inducing point on localized training data ...')
    old_accuracy = []
    new_accuracy = []
    old_p_ave = 0.0
    new_p_ave = 0.0

    X_test_all = X_test[0]
    Y_test_all = Y_test[0]
    for i in range(1, len(X_test)):
        X_test_all = np.concatenate((X_test_all, X_test[i]), axis=0)
        Y_test_all = np.concatenate((Y_test_all, Y_test[i]), axis=0)

    for i in range(len(sgps)):
        old_p = rmse(Y_test_all, sgps[i].predict(X_test_all)[0])
        new_p = rmse(Y_test_all, new_sgps[i].predict(X_test_all)[0])
        old_p_ave = old_p_ave + old_p
        new_p_ave = new_p_ave + new_p
        old_accuracy.append(old_p)
        new_accuracy.append(new_p)
        print('Evaluation for local model no. %d (pre-match, post-match) : (%f, %f)' % (i, old_p, new_p))
    old_p_ave = old_p_ave / len(sgps)
    new_p_ave = new_p_ave / len(sgps)
    print('Averaged (pre-match, post-match) over all local models : (%f, %f)' % (old_p_ave, new_p_ave))

def experiment(n_gp):
    X_partition, Y_partition, X_test, Y_test = generate_partitioned_data()
    X_train = X_partition[0]
    Y_train = Y_partition[0]
    for i in range(1, len(X_partition)):
        X_train = np.concatenate((X_train, X_partition[i]), axis=0)
        Y_train = np.concatenate((Y_train, Y_partition[i]), axis=0)

    start = time.time()
    sgps = construct_local_sgps(X_partition, Y_partition)
    new_sgps, est_atoms, pool_atoms = fusion(sgps, X_partition, Y_partition, reopt=True)
    end = time.time()
    overhead = end - start

    kmean_rmse, kmean_time = kmean_baseline(X_train, Y_train, X_test, Y_test, pool_atoms, est_atoms.shape[0])
    centralized_rmse, centralized_time = centralized_gp_baseline(X_train, Y_train, X_test, Y_test, est_atoms)
    fused_rmse, fused_time = fuse_gp(X_train, Y_train, X_test, Y_test, est_atoms, overhead=overhead)
    local_gp_evaluation(sgps, new_sgps, X_test, Y_test)

    plt.show()

    return kmean_rmse, kmean_time, centralized_rmse, centralized_time, fused_rmse, fused_time


experiment(2)
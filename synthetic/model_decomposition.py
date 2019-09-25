import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Parameter to change
x_dim = 5
y_dim = 3
num_data = 60000
num_datasets = 10
dataset_size = 200

f = h5py.File('synthetic_dataset.hdf5', 'r')
x_train = torch.Tensor(np.array(f["train/x"])).cuda()
y_train = torch.Tensor(np.array(f["train/y"])).long().cuda()
x_test = torch.Tensor(np.array(f["test/x"])).cuda()
y_test = torch.Tensor(np.array(f["test/y"])).long().cuda()
true_alpha = torch.Tensor(np.array(f["alpha"])).cuda()
f.close()

def maximum_likelihood():
    global x_train, y_train, x_test, y_test
    y_train_one_hot = torch.nn.functional.one_hot(y_train).float().squeeze()
    log_alpha = torch.rand((y_dim, x_dim), requires_grad=True)
    opt = torch.optim.Adam([log_alpha])
    for epoch in range(100000):
        opt.zero_grad()
        alpha_cuda = torch.exp(log_alpha).cuda()
        p = torch.pow(x_train.unsqueeze(1) - alpha_cuda.unsqueeze(0), 2).sum(-1)
        p2 = p / p.sum(1, keepdims=True)
        neg_loglikelihood = - torch.log((p2 * y_train_one_hot).sum(1)).sum(0)
        if epoch % 10000 == 0:
            print(epoch, neg_loglikelihood)
        neg_loglikelihood.backward()
        opt.step()
    return torch.exp(log_alpha).cuda()

def learning():
    logBc = torch.Tensor(np.load('softmax_outputs_train.npy')).cuda()
    alpha = torch.rand((y_dim, x_dim), requires_grad=True)
    opt = torch.optim.Adam([alpha])
    for e in range(10000):
        opt.zero_grad()
        alpha_cuda = alpha.cuda()
        distance_mat = torch.pow(x_train.unsqueeze(1) - alpha_cuda.unsqueeze(0), 2).sum(-1)
        p = distance_mat / distance_mat.sum(1, keepdims=True)
        loss = torch.pow(logBc - p, 2).sum(1).sum(0)
        if e % 1000 == 0:
            print(e, loss)
        loss.backward()
        opt.step()
    return alpha.cuda()

def evaluate(alpha):
    # Evaluate the accuracy on train set
    alpha_cuda = alpha.cuda()
    p = torch.pow(x_train.unsqueeze(1) - alpha_cuda.unsqueeze(0), 2).sum(-1)
    p2 = p / p.sum(1, keepdims=True)
    pred = p2.argmax(dim=1)
    correct = pred.eq(y_train.view_as(pred)).sum().item()
    print("Train accuracy: ", float(correct) / y_train.shape[0])

    # Evaluate the accuracy on test set
    p = torch.pow(x_test.unsqueeze(1) - alpha_cuda.unsqueeze(0), 2).sum(-1)
    p2 = p / p.sum(1, keepdims=True)
    pred = p2.argmax(dim=1)
    correct = pred.eq(y_test.view_as(pred)).sum().item()
    print("Test accuracy: ", float(correct) / y_test.shape[0])

alpha_mle = learning()
evaluate(alpha_mle)
evaluate(true_alpha)
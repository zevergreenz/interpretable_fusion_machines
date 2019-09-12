import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="4"

# Parameters to change
num_models = 2
m = 6
n = 40

f = h5py.File('uci_dataset.hdf5', 'r')
x_test = torch.Tensor(np.array(f["test/x"])).cuda()
y_test = torch.Tensor(np.array(f["test/y"])).long().cuda()

x_trains = []
y_trains = []
for i in range(num_models):
    x_train = torch.Tensor(np.array(f["train_{}/x".format(i)])).cuda()
    y_train = torch.Tensor(np.array(f["train_{}/y".format(i)])).long().cuda()
    x_trains.append(x_train)
    y_trains.append(y_train)
f.close()

def r(loggamma):
    gamma = torch.exp(loggamma)
    return m * torch.sqrt(gamma) / np.sqrt(n) / torch.sqrt(gamma.sum(0))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class VAE(nn.Module):
    def __init__(self, x_dim=30, h_dim1=128, h_dim2=64, z_dim=2):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 30))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def sample_z(self, x):
        mu, log_var = self.encoder(x.view(-1, 30))
        z = self.sampling(mu, log_var)
        return z


class ConvNet(object):
    def __init__(self, i,
                 model_ckp="uci_models.pth",
                 vae_ckp="uci_vaes.pth"):
        # Path to model weight
        self._model_ckp = model_ckp
        self._vae_ckp = vae_ckp
        self._i = i

        self._model = Net().cuda()
        self._model.load_state_dict(torch.load(self._model_ckp)['model_{}'.format(self._i)])
        self._vae = VAE(x_dim=2).cuda()
        self._vae.load_state_dict(torch.load(self._vae_ckp)['vae_{}'.format(self._i)])

    def evaluate(self):
        pred = self._model(x_test).argmax(dim=1)
        correct = pred.eq(y_test.view_as(pred)).sum().item()
        return float(correct) / y_test.shape[0]

    def predict(self, x):
        return self._model(x)

    # sample w' ~ p
    # w'_i = 1 / distance between z and z_i
    def sample_p(self, x, x_train):
        # z = self._vae.sample_z(x) # 1 x latent_dim
        # z_train = self._vae.sample_z(x_train) # N x latent_dim
        mean = ((self.predict(x) - self.predict(x_train)) ** 2).mean(1)
        sample = torch.distributions.Normal(loc=mean, scale=torch.ones_like(mean)).rsample((1,))
        return sample.squeeze()

    def check_E_g(self, x, nsamples=100):
        assert x.shape[0] == 1
        n = x_trains[self._i].shape[0]
        log_gamma = torch.zeros(n, requires_grad=True)
        log_gamma_gpu = log_gamma.cuda()
        log_Bc = self.predict(x_trains[self._i])
        # E_g with w_i_prime
        E_g = 0
        for k in range(nsamples):
            w_prime = self.sample_p(x, x_trains[self._i]).cuda()
            w_prime /= w_prime.sum()
            E_h = w_prime.unsqueeze(1) * log_Bc * r(log_gamma_gpu).unsqueeze(1)
            E_h2 = ((w_prime.unsqueeze(1) * log_Bc) ** 2) * r(log_gamma_gpu).unsqueeze(1)
            E_g += E_h2.sum(1).sum(0)
            for j in range(n):
                for i in range(j):
                    E_g += 2 * (E_h[j] * E_h[i]).sum()
            E_g -= 2 * (self.predict(x).squeeze() * E_h.sum(0)).sum()
            E_g += (self.predict(x).squeeze() ** 2).sum()
        E_g /= nsamples
        # True E_g with w_i
        unit_normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        true_E_g = 0
        for k in range(nsamples):
            w_prime = self.sample_p(x, x_trains[self._i]).cuda()
            w_prime /= w_prime.sum()
            u = unit_normal.rsample((n,)).cuda()
            w = w_prime * (u.squeeze() <= unit_normal.icdf(r(log_gamma_gpu).cpu()).cuda()).float()
            a = (w.unsqueeze(1) * log_Bc).sum(0)
            true_E_g += ((a - self.predict(x).squeeze()) ** 2).sum()
        true_E_g /= nsamples
        print("True E_g: ", true_E_g)
        print("E_g: ", E_g)
        return E_g - true_E_g

    def predict_fuse(self, x):
        assert x.shape[0] == 1
        n = x_trains[self._i].shape[0]
        log_gamma = torch.rand(n, requires_grad=True)
        log_gamma_gpu = log_gamma.cuda()
        log_Bc = self.predict(x_trains[self._i])
        opt = torch.optim.Adam([log_gamma])
        for epoch in range(10):
            opt.zero_grad()
            w_prime = self.sample_p(x, x_trains[self._i])
            E_h = w_prime.unsqueeze(1) * log_Bc * r(log_gamma_gpu).unsqueeze(1)
            E_h2 = ((w_prime.unsqueeze(1) * log_Bc)**2) * r(log_gamma_gpu).unsqueeze(1)
            E_g = E_h2.sum(1).sum(0)
            for j in range(n):
                for i in range(j):
                    E_g += 2 * (E_h[j] * E_h[i]).sum()
            E_g -= 2 * (self.predict(x).squeeze() * E_h.sum(0)).sum()
            E_g += (self.predict(x).squeeze() ** 2).sum()
            # if epoch % 10 == 0:
            #     print(epoch, E_g.item())
            #     print(log_gamma)
            E_g.backward(retain_graph=True)
            opt.step()
        unit_normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        w_prime = self.sample_p(x, x_trains[self._i])
        u = unit_normal.rsample((n,)).cuda()
        w = w_prime * (u.squeeze() <= unit_normal.icdf(r(log_gamma_gpu).cpu()).cuda()).float()
        non_zero_idx = torch.where(w != 0)
        # return w[non_zero_idx], log_Bc[non_zero_idx]
        return w_prime, log_Bc


def fuse(models, x, m=10):
    assert x.shape[0] == 1
    w = []
    logBc = []
    for model in models:
        w_, logBc_ = model.predict_fuse(x)
        w.append(w_)
        logBc.append(logBc_)

    G = torch.zeros((m, num_models, n))
    w_l = torch.cat(w, dim=0)[:m]
    logB_l = torch.cat(logBc, dim=0)[:m]
    for i in range(num_models):
        for j in range(len(w[i])):
            G[(i+j) % m, i, j] = 1
    for _ in range(100):
        # w-step
        for l in range(m):
            term1 = 0
            for i in range(num_models):
                for j in range(len(w[i])):
                    term1 += (G[l, i, j] * w[i][j] * logBc[i][j]).sum()
            term2 = G[l, :, :].sum(1).sum(0)
            term3 = logB_l.sum()
            if term2 != 0:
                w_l[l] = term1 / term2 / term3
                if torch.isnan(w_l).any():
                    print("HERE1", term1, term2, term3)
        # p-step
        for l in range(m):
            term1 = 0
            for i in range(num_models):
                for j in range(len(w[i])):
                    term1 += G[l, i, j] * w[i][j] * logBc[i][j]
            term2 = w_l[l] * G[l, :, :].sum(1).sum(0)
            if term2 != 0:
                logB_l[l] = term1 / term2
                if torch.isnan(logB_l).any():
                    print("HERE2", term1, term2)
        # g-step
        for i in range(num_models):
            for j in range(len(w[i])):
                import math
                best_value, best_idx = math.inf, None
                for l in range(m):
                    value = ((w_l[l] * logB_l[l] - w[i][j] * logBc[i][j]).sum(0) ** 2).item()
                    if value < best_value:
                        best_value, best_idx = value, l
                # print(l, best_idx)
                G[:, i, j] = torch.zeros(m)
                G[best_idx, i, j] = 1
        for l in range(m):
            if G[l, :, :].sum(1).sum(0) == 0:
                for k in range(m):
                    if k != l and G[k, :, :].sum(1).sum(0) > 1:
                        w_l[l] = w[i][j]
                        logB_l[l] = logBc[i][j]
                        G[:, i, j] = torch.zeros(m)
                        G[l, i, j] = 1
    return (w_l.unsqueeze(1) * logB_l).sum(0)


def PoE(models):
    with torch.no_grad():
        pred = torch.zeros(x_test.shape[0], 10).cuda()
        for model in models:
            pred += model.predict(x_test)
        pred = pred.argmax(dim=1)
        correct = pred.eq(y_test.view_as(pred)).sum().item()
    return float(correct) / y_test.shape[0]


def fuse2(models, x, m=2):
    assert x.shape[0] == 1
    w = []
    logBc = []
    for model in models:
        w_, logBc_ = model.predict_fuse(x)
        w.append(w_)
        logBc.append(logBc_)
    w = torch.cat(w)
    logBc = torch.cat(logBc)
    return (w.unsqueeze(1) * logBc).sum(0)


models = []
for i in range(num_models):
    model = ConvNet(i)
    models.append(model)
    print(model.evaluate())

correct = 0
for i in range(x_test.shape[0]):
    pred = fuse(models, x_test[i:i+1], m=6)
    pred = pred.argmax()
    if pred == y_test[i]:
        correct += 1
print(correct)
print(x_test.shape[0])
print(float(correct) / x_test.shape[0])
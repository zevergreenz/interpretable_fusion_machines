import torch
import numpy as np
import pyro
from torch.autograd import Variable

T = 2
delta = 0.6
m = 20
n = 54000
alpha = torch.ones(n).cuda()
loggamma = torch.zeros(n, requires_grad=True).cuda()

pi = pyro.distributions.Normal(torch.ones(n, dtype=torch.float32) * 10, torch.ones(n, dtype=torch.float32), validate_args=True)

def p(pi, alpha):
    a = pyro.sample("a", pyro.distributions.Uniform(torch.zeros_like(alpha), torch.ones_like(alpha), validate_args=True)).cuda()
    w = pyro.sample("w", pi).cuda()
    return w * (a < alpha).float()

def r(loggamma):
    gamma = torch.exp(loggamma)
    return m * torch.sqrt(gamma) / np.sqrt(n) / torch.sqrt(gamma.sum(0))

def drdgamma(loggamma):
    res = []
    for i in range(n):
        r_gamma = r(loggamma)
        gr = torch.autograd.grad(r_gamma[i], loggamma, allow_unused=True)[0]
        res.append(gr[i])
    return torch.Tensor(res)

def E_h():
    w_prime = p(pi, alpha) # n x 1
    return w_prime.unsqueeze(1) * torch.log(Bc) * r(loggamma).unsqueeze(1)

def E_h2():
    w_prime = p(pi, alpha) # n x 1
    return ((w_prime.unsqueeze(1) * torch.log(Bc))**2) * r(loggamma).unsqueeze(1)

def E_g(Bcx):
    res = E_h2().sum(1).sum(0)
    E_h_ = E_h()
    for j in range(n):
        for i in range(j):
            res += 2 * (E_h_[j] * E_h_[i]).sum()
    res -= 2 * (torch.log(Bcx) * E_h_.sum(0)).sum()
    res += (torch.log(Bcx) ** 2).sum()
    return res

def dE_hdgamma():
    w_prime = p(pi, alpha) # n x 1
    return w_prime.unsqueeze(1) * torch.log(Bc) * drdgamma(loggamma).unsqueeze(1)

def dE_h2dgamma():
    w_prime = p(pi, alpha) # n x 1
    return ((w_prime.unsqueeze(1) * torch.log(Bc))**2) * drdgamma(loggamma).unsqueeze(1)

# def g(w):
#     temp = torch.mm(w.unsqueeze(0), torch.log(Bc))  # (1xK)
#     if temp.sum(1) != 0:
#         temp2 = temp / temp.sum(1)
#     else:
#         temp2 = temp
#     return ((temp2 - torch.log(Bc)) ** 2).sum(1).sum(0)
#
# def L(gamma):
#     r_gamma = r(gamma)
#     w_prime = p(pi, alpha)
#     g_w_prime = g(w_prime)
#     g_0 = g(torch.zeros_like(w_prime))
#     return g_w_prime * r_gamma + g_0 * (1 - r_gamma)
#
# def dLdgamma(gamma):
#     r_gamma = r(gamma)
#     w_prime = p(pi, alpha)
#     g_w_prime = g(w_prime)
#     g_0 = g(torch.zeros_like(w_prime))
#     drdgamma_ = drdgamma(gamma)
#     return g_w_prime * drdgamma_ + g_0 * (1 - drdgamma_)


Bc = torch.tensor(np.load('softmax_outputs_train.npy')).cuda()  # B(c|x_i)
Bc = Bc[:n]  # TODO: Remove this on real experiment


# Parameterization trick with path-wise sampling
# class Our_Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pi = pyro.distributions.Normal(torch.ones(n, dtype=torch.float32), torch.ones(n, dtype=torch.float32),
#                                             validate_args=True)
#         self.gamma_unconstrained = torch.nn.Parameter(torch.ones(n), requires_grad=True)
#         self.unit_gaussian = pyro.distributions.Normal(torch.zeros_like(alpha, dtype=torch.float32),
#                                                        torch.ones_like(alpha, dtype=torch.float32), validate_args=True)
#
#     def forward(self):
#         u = pyro.sample("u", self.unit_gaussian)
#         w = p(self.pi, alpha)
#         gamma = torch.exp(self.gamma_unconstrained)
#         r = delta * k / (2 * np.sqrt(n)) * torch.sqrt(gamma) / torch.sqrt(gamma.sum(0))
#         deterministic_transform_fn = make_logistic(self.unit_gaussian.icdf(r))
#         w1 = w * deterministic_transform_fn(u)
#         w2 = (w1 / w1.sum()).unsqueeze(0)
#         Bc_hat = torch.mm(w2, torch.log(Bc))
#
#         Bc_drop = (u <= self.unit_gaussian.icdf(r)).float().unsqueeze(1) * Bc
#         return torch.mm(w.unsqueeze(0), torch.log(Bc_drop))
#
# def get_model():
#     model = Our_Model()
#     return model, torch.optim.Adam(model.parameters(), lr=0.001)
#
#
# model, opt = get_model()

# for epoch in range(100000):
#     model.train()
#
#     loss = Rin_loss_func(model(), Bc)  # + complexity_loss_func(model.gamma)
#     loss.backward()
#     opt.step()
#     opt.zero_grad()
#     if epoch % 10000 == 0:
#         print(loss)
#         print(model.gamma_unconstrained[:7])
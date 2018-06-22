import torch
import numpy as np
from torch import nn
from torch.nn import Module, Parameter
from torch.nn import functional as F


torch.manual_seed(5)


def update_parameters(m):

    if isinstance(m, MyLinear):

        if m.reg_flag:
            dg_lik = -m.weight.grad.data
            m.mu.add_(m.lr*dg_lik)
            m.weight.data = m.mu
        else:
            dg_lik = - m.weight.grad.data

            C2 = m.C.mul(m.C)
            Cmu = C2 + torch.pow(m.mu, 2)
            dmu = dg_lik - m.mu / Cmu
            dC = dg_lik.mul(m.z) + 1 / m.C - m.C / Cmu

            # Stochastic update of the parameters
            m.mu.add_(m.lr * dmu)
            m.C.add_((0.1 * m.lr), dC)
            m.C[m.C < 1e-4] = 1e-4
            m.z = torch.randn(m.out_features, m.in_features)
            m.weight.data = m.C.mul(m.z) + m.mu


class DsviArd(Module):

    def __init__(self, n,  in_features, out_features, lr, reg_flag):
        super(DsviArd, self).__init__()
        self.n = n
        self.reg_flag = reg_flag
        self.lr = lr
        self.in_features = in_features
        self.out_features = out_features
        self.linear_in = MyLinear(self.n, self.in_features, 50, self.lr, self.reg_flag)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.linear_out = MyLinear(self.n, 50, self.out_features, self.lr, self.reg_flag)
        self.softmax = nn.Softmax()
        self.criterion = nn.CrossEntropyLoss(size_average=False)

    def forward(self, t, X):

        h = self.tanh(self.linear_in(X))
        # h = self.relu(self.linear_in(X))
        y = self.linear_out(h)
        s = self.softmax(y)
        logsumexp = self.criterion(y, t)
        return logsumexp, s


class MyLinear(Module):

    def __init__(self, n,  in_features, out_features, lr, reg_flag, bias=True):
        super(MyLinear, self).__init__()
        self.n = n
        self.reg_flag = reg_flag
        self.lr = lr
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.C = torch.FloatTensor(out_features, in_features)
        self.mu = torch.FloatTensor(out_features, in_features)
        self.z = torch.FloatTensor(out_features, in_features)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        if self.reg_flag:
            self.mu = torch.zeros(self.out_features, self.in_features)
            self.C = torch.zeros(self.out_features, self.in_features)
            self.weight.data = self.mu
        else:
            self.z = torch.randn(self.out_features, self.in_features)
            self.C = 0.01 * torch.ones(self.out_features, self.in_features)

            self.mu = torch.zeros(self.out_features, self.in_features)
            self.weight.data = self.C.mul(self.z) + self.mu

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


def dsviard(X, T, D, D_out, reg_flag):

    dec = 0.95
    ro = 1e-5

    niter = 5000
    iters = 3

    # If reg_flag is set to True the model runs without the regularization parameter
    model = DsviArd(X.shape[0], D, D_out, ro, reg_flag)
    F = np.zeros((iters*niter, 1))

    for i in range(1, iters+1):
        Ftmp = np.zeros((niter, 1))

        for epoch in range(niter):
            print('Epoch {}/{}'.format(epoch, niter - 1))
            print('-' * 10)

            loss_fn, y_pred = model(T, X)
            model.zero_grad()
            loss_fn.backward()
            if reg_flag:
                model.apply(update_parameters)
                Ftmp[epoch] = np.array((-loss_fn.data))
            else:

                C_in_2 = model.linear_in.C.mul(model.linear_in.C)
                Cmu_in = C_in_2 + torch.pow(model.linear_in.mu, 2)
                C_out_2 = model.linear_out.C.mul(model.linear_out.C)
                Cmu_out = C_out_2 + torch.pow(model.linear_out.mu, 2)

                model.apply(update_parameters)
                Ftmp[epoch] = np.array((-loss_fn.data + 0.5 * sumnorm(C_in_2,  C_out_2, Cmu_in, Cmu_out)))

            print('==>>> epoch: {}, train loss: {}'.format(epoch, Ftmp[epoch]))
            print('===>> mu sum : {}, C sum: {}'.format(model.linear_in.mu.sum(), model.linear_in.C.sum()))
        F[(i - 1) * niter:i * niter] = Ftmp
        model.lr = dec * model.lr
        print(model.lr)
        print('Iters {}/{} mean {} '.format(i, iters, np.mean(Ftmp)))

    return F, model.linear_in.mu, model.linear_out.C, model.linear_out.mu, model.linear_out.C


def sumnorm(C1, C2, Cm1, Cm2):
    norm1 = torch.sum(torch.log(C1/Cm1))
    norm2 = torch.sum(torch.log(C2/Cm2))
    return norm1 + norm2


import torch
import numpy as np


def softmax(y):
    m, _ = torch.max(y, 1)
    y = y - m.view(m.shape[0], 1)
    y = torch.exp(y)
    s = torch.div(y, torch.sum(y, 1).view(y.shape[0], 1))
    return s


def softmax_n(y):

    k = y.shape[1]

    m = np.max(y, 1)
    m = m.reshape(m.shape[0], 1)
    y = y - np.tile(m, k)

    y = np.exp(y)

    p = np.sum(y, 1)
    p = p.reshape(p.shape[0], 1)
    s = np.divide(y, np.tile(p, k))
    return s


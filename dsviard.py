import torch
import numpy as np
from loglik import loglikelihood


def dsviard(X, T, mu, C, ro, D, D_out, niter):
    F = np.zeros((niter, 1))

    for i in range(niter):
        z = torch.FloatTensor(np.random.standard_normal(size=(D, D_out)))
        theta = C.mul(z) + mu

        g_lik, dg_lik = loglikelihood(theta, X, T)

        C2 = C.mul(C)
        Cmu = C2 + torch.pow(mu, 2)

        """Stochastic gradient update of the parameters"""
        dmu = dg_lik - torch.div(mu, Cmu)
        dC = (dg_lik.mul(z)) + 1/C - torch.div(C, Cmu)

        mu = mu + ro*dmu
        C = C + (0.1*ro)*dC
        C[C <= 1e-4] = 1e-4

        F[i] = np.array(g_lik + 0.5*torch.sum(torch.log(torch.div(C2, Cmu))))

    return F, mu, C








import torch.nn as nn
import torch
import numpy as np
from typing import Union, Iterable, Tuple, Dict
from torchdiffeq import odeint as odeint

torch.pi = torch.tensor(np.pi)


class HNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(HNN, self).__init__()
        self.dt = deltat
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, output_size,bias=None)
        self.nonlin = torch.tanh
        self.M = self.get_perm_mat(input_dim)
        for l in [self.mlp1, self.mlp2, self.mlp3]:
            torch.nn.init.orthogonal_(l.weight)

    def renorm(self,xnow):
        # firsts = torch.remainder(xnow[:, :2] +torch.pi ,2 * torch.pi) - torch.pi
        # return torch.cat([firsts,xnow[:,2:]],1)
        return xnow

    def rk4(self, dx_dt_fn, x_t):
        k1 = self.dt * dx_dt_fn(x_t)
        k2 = self.dt * dx_dt_fn(x_t + (1. / 2) * k1)
        k3 = self.dt * dx_dt_fn(x_t + (1. / 2) * k2)
        k4 = self.dt * dx_dt_fn(x_t + k3)
        x_tp1 = x_t + (1. / 6) * (k1 + k2 * 2. + k3 * 2. + k4)
        return x_tp1

    def get_H(self, x):
        h = self.nonlin(self.mlp1(x))
        h = self.nonlin(self.mlp2(h))
        fin = self.mlp3(h)
        return fin

    def next_step(self, x):
        init_param = self.renorm(x)
        next_set = self.rk4(self.time_deriv, init_param)
        return self.renorm(next_set)

    def time_deriv(self, x):
        init_x = x
        F2 = self.get_H(init_x)
        dF2 = torch.autograd.grad(F2.sum(), init_x, create_graph=True)[0]
        return dF2@self.M

    def get_perm_mat(self,n):
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])
        return M*(-1)
import torch.nn as nn
import torch
import numpy as np
from typing import Union, Iterable, Tuple, Dict

torch.pi = torch.tensor(np.pi)


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


def tensor(v: Union[float, np.ndarray, torch.Tensor],
           min_ndim=1,
           device=None,
           **kwargs):
    """
    Construct a tensor if the input is not; otherwise return the input as is,
    but return None as is for convenience when input is not passed.
    Same as enforce_tensor
    :param v:
    :param min_ndim:
    :param device:
    :param kwargs:
    :return:
    """
    # if device is None:
    #     device = device0

    if v is None:
        pass
    else:
        if not torch.is_tensor(v):
            v = torch.tensor(v, **kwargs)
        if v.ndimension() < min_ndim:
            v = v.expand(v.shape
                         + torch.Size([1] * (min_ndim - v.ndimension())))
    return v


class HNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(HNN, self).__init__()
        self.dt = deltat
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, output_size)
        self.nonlin = torch.tanh
        for l in [self.mlp1, self.mlp2, self.mlp3]:
            torch.nn.init.orthogonal_(l.weight)

    def rk4(self, dx_dt_fn, x_t):
        k1 = self.dt * dx_dt_fn(x_t)
        k2 = self.dt * dx_dt_fn(x_t + (1. / 2) * k1)
        k3 = self.dt * dx_dt_fn(x_t + (1. / 2) * k2)
        k4 = self.dt * dx_dt_fn(x_t + k3)
        x_tp1 = x_t + (1. / 6) * (k1 + k2 * 2. + k3 * 2. + k4)
        return x_tp1

    def forward(self, x):
        h = self.nonlin(self.mlp1(x))
        h = self.nonlin(self.mlp2(h))
        fin = self.mlp3(h)
        return fin

    # def rebase(self, q):
    #     return torch.fmod((q + torch.pi) , (2 * torch.pi)) - torch.pi

    def next_step(self, x):
        init_param = x
        next_set = self.rk4(self.time_deriv, init_param)
        return next_set

    def time_deriv(self, x):
        F2 = self.forward(x)
        dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]
        return torch.cat([dF2[:, 1:], -dF2[:, :1]], 1)

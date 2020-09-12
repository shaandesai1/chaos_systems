import torch.nn as nn
import torch
import numpy as np
from typing import Union, Iterable, Tuple, Dict
from torchdiffeq import odeint as odeint

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


class HNN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(HNN2, self).__init__()
        self.dt = deltat
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, output_size)
        self.nonlin = torch.cos

        for l in [self.mlp1, self.mlp2, self.mlp3]:
            torch.nn.init.orthogonal_(l.weight)

    def get_H(self,x):
        h = self.nonlin(self.mlp1(x))
        h = self.nonlin(self.mlp2(h))
        F2 = self.mlp3(h)
        return F2

    def forward(self, t,x):
        # x.requires_grad = True
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float32, device='cpu', requires_grad=True)
            x = one * x
            F2 = self.get_H(x)
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]
            return torch.cat([dF2[0,1].reshape(1,1),-dF2[0,0].reshape(1,1)],1).reshape(1,2)#torch.cat([dF2[ 1:].reshape(1,1), -dF2[ :1].reshape(1,1)], 1)

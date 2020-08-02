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

class MODEL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, assume_canonical_coords=True,
                 field_type='solenoidal'):
        super(MODEL, self).__init__()
        self.dt = 0.01
        self.assume_canonical_coords = assume_canonical_coords
        self.field_type = field_type
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, output_size, bias=None)
        self.nonlin = torch.tanh

        for l in [self.mlp1, self.mlp2, self.mlp3]:
            torch.nn.init.orthogonal_(l.weight)

        self.mat1 = nn.Linear(2, 10)
        self.mat2 = nn.Linear(10, 10)
        self.mat3 = nn.Linear(10, 4)

        self.M = None
        self.Minv = None

    def rk4(self, dx_dt_fn, x_t, dt):
        k1 = dt * dx_dt_fn(x_t)
        k2 = dt * dx_dt_fn(x_t + (1 / 2) * k1)
        k3 = dt * dx_dt_fn(x_t + (1 / 2) * k2)
        k4 = dt * dx_dt_fn(x_t + k3)
        x_tp1 = x_t + (1 / 6) * (k1 + k2 * 2 + k3 * 2 + k4)
        return x_tp1

    def M_matrix(self, q):
        h = self.nonlin(self.mat1(q))
        h = self.nonlin(self.mat2(h))
        out = self.mat3(h)
        self.M = self.block_diag(out.reshape(-1, 2, 2))
        self.Minv = torch.inverse(self.M)
        return self.M
    def block_diag(self, m):
        """
        Make a block diagonal matrix along dim=-3
        EXAMPLE:
        block_diag(torch.ones(4,3,2))
        should give a 12 x 8 matrix with blocks of 3 x 2 ones.
        Prepend batch dimensions if needed.
        You can also give a list of matrices.
        :type m: torch.Tensor, list
        :rtype: torch.Tensor
        """
        if type(m) is list:
            m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

        d = m.dim()
        n = m.shape[-3]
        siz0 = m.shape[:-3]
        siz1 = m.shape[-2:]
        m2 = m.unsqueeze(-2)
        eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
        return (m2 * eye).reshape(
            siz0 + torch.Size(tensor(siz1) * n)
        )

    def forward(self, x):

        h = self.nonlin(self.mlp1(x))
        h = self.nonlin(self.mlp2(h))
        fin = self.mlp3(h)

        return fin

    def rebase(self,q):

        return (q + torch.pi) % (2 * torch.pi) - torch.pi
    def next_step(self, x):

        # step 1, get p
        _ = self.M_matrix(self.rebase(x[:,:2]))
        p_init = (self.M@(x[:,2:].reshape(-1,1))).reshape(-1,2)
        # L = self.forward(x)  # traditional forward pass
        # dlds = torch.autograd.grad(L.sum(), x, create_graph=True)[0]
        # p_init = dlds[:, 2:]
        q_init = self.rebase(x[:, :2])
        init_param = torch.cat([q_init, p_init], 1)
        next_set = self.rk4(self.time_deriv, init_param,self.dt)
        qdot = (self.Minv @ (next_set[:, 2:].reshape(-1, 1))).reshape(-1, 2)
        return torch.cat([self.rebase(next_set[:,:2]),qdot],1)

    def Lpartials(self,x):
        F2 = self.forward(x)  # traditional forward pass
        dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]  # gradients for solenoidal field
        return dF2
    def Hpartials(self,x):
        F2 = self.forward(x)  # traditional forward pass
        dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]  # gradients for solenoidal field
        return torch.cat([dF2[:,2:],-dF2[:,:2]],1)
    def time_deriv(self, x, p=False, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''

        # step 1, get qdot back
        # q = self.rebase(x[:,:2])
        # qdot = (self.Minv@(x[:,2:].reshape(-1,1))).reshape(-1,2)
        # Lpartials = self.Lpartials(torch.cat([q, qdot], 1))
        # pdot = Lpartials[:,:2]
        # return torch.cat([qdot,pdot],1)

        hp = self.Hpartials(x)
        return hp

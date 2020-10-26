import torch.nn as nn
import torch
import numpy as np

import torch.nn as nn
import torch
import numpy as np
from .baseline import *


class TDHNN1(base_model):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(TDHNN1, self).__init__(input_dim, hidden_dim, output_size, deltat)
        self.f1 = nn.Linear(1, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f2_ = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, 1,bias=None)
        self.nonlin = torch.tanh
        for l in [self.f1, self.f2, self.f2_, self.f3]:
            torch.nn.init.orthogonal_(l.weight)

    def get_F(self, x):
        h = self.nonlin(self.f1(x))
        h = self.nonlin(self.f2(h))
        h = self.nonlin(self.f2_(h))
        h = self.f3(h)
        return h

    def time_deriv(self, x, t):
        H0, F = self.get_H(x), self.get_F(t.reshape(-1, 1))
        H_full = H0 + (x[:, 0] * F[:, 0]).reshape(-1, 1)
        dF2 = torch.autograd.grad(H_full.sum(), x, create_graph=True)[0]
        return dF2 @ self.M.t()


import torch.nn as nn
import torch
import numpy as np

import torch.nn as nn
import torch
import numpy as np
from .baseline import *


class TDHNN4(base_model):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(TDHNN4, self).__init__(input_dim, hidden_dim, output_size, deltat)
        self.f1 = nn.Linear(1, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f2_ = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, 1, bias=False)
        self.nonlin = torch.sin
        # for l in [self.f1, self.f2, self.f2_, self.f3]:
        #     torch.nn.init.orthogonal_(l.weight)

        # self.W = nn.Parameter(torch.zeros(1, 1))
        # self.W = nn.init.kaiming_normal_(self.W)

        # self.Wd = nn.Parameter(torch.zeros(1, 1))
        # self.Wd = nn.init.kaiming_normal_(self.Wd)

        self.d1 = nn.Parameter(torch.zeros(1, 1))
        self.d1 = nn.init.kaiming_normal_(self.d1)
#nn.Linear(1, 1, bias=None)

    def get_F(self, x):
        h = self.nonlin(self.f1(x))
        h = self.nonlin(self.f2(h))
        h = self.nonlin(self.f2_(h))
        h = self.f3(h)
        return h

    def get_D(self):
        return self.d1

    def time_deriv(self, x, t):
        H0, F, D = self.get_H(x), self.get_F(t.reshape(-1, 1)), self.get_D()
        dH0 = torch.autograd.grad(H0.sum(), x, create_graph=True)[0]
        derivs = dH0 @ self.M.t()
        qdot = derivs[:,0].reshape(-1,1)
        pdot = derivs[:,1].reshape(-1,1)

        new_pdot = pdot + D*qdot + F
        fin_deriv = torch.cat([qdot, new_pdot], 1)
        return fin_deriv

    def get_weight(self):
        return  # torch.relu(torch.sign(torch.sigmoid(self.W) - 0.5)),torch.relu(torch.sign(torch.sigmoid(self.Wd) - 0.5))

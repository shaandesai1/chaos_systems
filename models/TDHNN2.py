import torch.nn as nn
import torch
import numpy as np

import torch.nn as nn
import torch
import numpy as np
from .baseline import *

class TDHNN2(base_model):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(TDHNN2, self).__init__(input_dim,hidden_dim,output_size,deltat)
        self.f1 = nn.Linear(1, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f2_ = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, 1,bias=None)
        self.nonlin = torch.tanh
        for l in [self.f1, self.f2, self.f2_, self.f3]:
            torch.nn.init.orthogonal_(l.weight)

        self.W = nn.Parameter(torch.zeros(1, 1))
        self.W = nn.init.kaiming_normal_(self.W)

    def get_F(self, x):
        h = self.nonlin(self.f1(x))
        h = self.nonlin(self.f2(h))
        h = self.nonlin(self.f2_(h))
        h = self.f3(h)
        return h

    def time_deriv(self, x, t):
        H0, F = self.get_H(x), self.get_F(t.reshape(-1, 1))
        wbin = torch.sigmoid(self.W)#1#torch.nn.LeakyReLU()(torch.sigmoid(self.W)-0.5)
        H_full = H0 +torch.mm((x[:,0]*F[:,0]).reshape(-1,1), wbin) #(x[:,0]*F[:,0]).reshape(-1,1)#
        dF2 = torch.autograd.grad(H_full.sum(), x, create_graph=True)[0]
        return dF2@self.M.t()

    def get_weight(self):
        return torch.sigmoid(self.W)











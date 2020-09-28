import torch.nn as nn
import torch
import numpy as np
from typing import Union

torch.pi = torch.tensor(np.pi)

class HNN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(HNN2, self).__init__()
        self.dt = deltat
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, output_size)
        self.nonlin = torch.tanh

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

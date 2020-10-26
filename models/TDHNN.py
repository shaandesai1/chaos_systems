import torch.nn as nn
import torch
from .baseline import *

class TDHNN(base_model):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(TDHNN, self).__init__(input_dim,hidden_dim,output_size,deltat)
        self.f1 = nn.Linear(1, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, 1,bias=None)
        self.nonlin = torch.tanh
        for l in [self.f1, self.f2,self.f3]:
            torch.nn.init.orthogonal_(l.weight)

    def time_deriv(self, x, t):
        input_vec = torch.cat([x,t.reshape(-1,1)],1)
        H = self.get_H(input_vec)
        dF2 = torch.autograd.grad(H.sum(), input_vec, create_graph=True,allow_unused=True)[0]
        return dF2[:,:2]@self.M.t()













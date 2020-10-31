import torch.nn as nn
import torch
import numpy as np
from .baseline import *
class HNN_dpend(base_model):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(HNN_dpend, self).__init__(input_dim,hidden_dim,output_size,deltat)

    def time_deriv(self, x,t):
        q1 = x[:,0]
        q2 = x[:,1]
        p1 = x[:,2].reshape(-1,1)
        p2 = x[:,3].reshape(-1,1)
        q1c = torch.cos(q1).reshape(-1,1)
        q1s = torch.sin(q1).reshape(-1,1)
        q2c = torch.cos(q2).reshape(-1,1)
        q2s = torch.sin(q2).reshape(-1,1)
        input_vec = torch.cat([q1c,q1s,q2c,q2s,p1,p2],1)
        F2 = self.get_H(input_vec)
        dF2 = torch.autograd.grad(F2.sum(), input_vec, create_graph=True)[0]
        qdot = dF2[:,4:6].reshape(-1,2)
        pdot1 = q1s*dF2[:,0].reshape(-1,1) - q1c*dF2[:,1].reshape(-1,1)
        pdot2 = q2s * dF2[:, 2].reshape(-1, 1) - q2c * dF2[:, 3].reshape(-1, 1)
        print(qdot.shape,pdot1.shape,pdot2.shape)
        return torch.cat([qdot,pdot1,pdot2],1)

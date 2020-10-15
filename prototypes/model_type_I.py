import torch.nn as nn
import torch
import numpy as np

class HNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(HNN, self).__init__()
        self.dt = deltat
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp4 = nn.Linear(hidden_dim, output_size)
        self.nonlin = torch.tanh
        self.M = self.permutation_tensor(2)
        for l in [self.mlp1, self.mlp2, self.mlp4,self.mlp3]:
            torch.nn.init.orthogonal_(l.weight)

    def rk4(self, dx_dt_fn, x_t,t):
        k1 = self.dt * dx_dt_fn(x_t,t)
        k2 = self.dt * dx_dt_fn(x_t + (1. / 2) * k1,t+.5*self.dt)
        k3 = self.dt * dx_dt_fn(x_t + (1. / 2) * k2,t+.5*self.dt)
        k4 = self.dt * dx_dt_fn(x_t + k3,t+self.dt)
        x_tp1 = x_t + (1. / 6) * (k1 + k2 * 2. + k3 * 2. + k4)
        return x_tp1

    def get_H(self, x):
        h = self.nonlin(self.mlp1(x))
        h = self.nonlin(self.mlp2(h))
        h = self.nonlin(self.mlp3(h))
        fin = self.mlp4(h)
        return fin.split(1,1)

    def permutation_tensor(self, n):
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])
        return M

    def next_step(self, x,t):
        init_param = x
        next_set = self.rk4(self.time_deriv, init_param,t)
        return next_set

    def time_deriv(self, x,t):
        input_vec = torch.cat([x, t.reshape(-1, 1)], 1)
        H0,F = self.get_H(input_vec)
#         print(H0.shape,F.shape)
        dF2 = torch.autograd.grad(H0.sum(), x, create_graph=True)[0]
        dqdt = dF2[:,1]
        dpdt = -dF2[:,0].reshape(-1,1) + F
        return torch.cat([dqdt.reshape(-1,1), dpdt.reshape(-1,1)], 1)

    def time_deriv2(self, x,t):
        q = x[:,0].reshape(-1,1)
        p = x[:,1].reshape(-1,1)
        tvec = t.reshape(-1,1)
        H0,F = self.get_H(torch.cat([q,p,tvec],1))
   
        
        dHdt = torch.autograd.grad(H0.sum(), tvec, create_graph=True,allow_unused=True)[0]
        dFdq = torch.autograd.grad(F.sum(), q, create_graph=True,allow_unused=True)[0]
        dFdp = torch.autograd.grad(F.sum(), p, create_graph=True,allow_unused=True)[0]
        return dHdt,dFdq,dFdp
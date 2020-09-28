import torch.nn as nn
import torch
import numpy as np

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
        fin = self.mlp3(h)
        return fin

    def next_step(self, x,t):
        init_param = x
        next_set = self.rk4(self.time_deriv, init_param,t)
        return next_set

    def time_deriv(self, x,t):
        input_vec = torch.cat([x, t.reshape(-1, 1)], 1)
        F2 = self.get_H(input_vec)
        dF2 = torch.autograd.grad(F2.sum(), input_vec, create_graph=True,allow_unused=True)[0]
        return torch.cat([dF2[:, 1].reshape(-1,1), -dF2[:, 0].reshape(-1,1)], 1)

    def time_deriv2(self, x,t):
        F2 = self.get_H(torch.cat([x,t.reshape(-1,1)],1))
        dF2 = torch.autograd.grad(F2.sum(), t, create_graph=True)[0]
        return dF2

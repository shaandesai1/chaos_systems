"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""
import torch
from data_builder import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
n_test_traj = 10
T_max_t = 2.01
dt = 0.01
srate = 0.01
noise_std = 0.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

valid_data = dpend_adapted(n_test_traj, T_max_t, dt, srate,2)
vnow, vnext,venergy,dvnow = nownext(valid_data, n_test_traj, T_max_t, dt, srate)
valdat = pendpixdata(vnow, vnext,venergy,dvnow)
val_dataloader = DataLoader(valdat, batch_size=int(T_max_t//srate), num_workers=2, shuffle=False)

data_dict = {'valid': val_dataloader}
torch.pi = torch.tensor(np.pi)

_,_,xpred,ypred = theta_to_cart(vnow)
plt.figure()
plt.scatter(xpred,ypred, label='predicted')
plt.legend()
plt.show()



def renorm(xnow):
    firsts = torch.remainder(xnow[:, :2]+torch.pi, 2 * torch.pi) - torch.pi
    return torch.cat([firsts, xnow[:, 2:]], 1)
    #return xnow

model= torch.load('mdl_s1')
model.eval()

for batch_i, (q, q_next, energy_, dq) in enumerate(data_dict['valid']):

    q, q_next, dq = q.float(), q_next.float(), dq.float()

    q.to(device)
    q_next.to(device)
    energy_.to(device)
    dq.to(device)
    q.requires_grad = True
    preds = []
    qinit = q[0].reshape(1, -1)
    for i in range(len(q_next)):
        next_step_pred = model.next_step(renorm(qinit))
        preds.append(next_step_pred)
        qinit = next_step_pred
    preds = torch.cat(preds)
    print(torch.mean((preds - q_next) ** 2))
    preds = preds.detach().numpy()
    q_next = q_next.detach().numpy()

    _, _, xpred, ypred = theta_to_cart(preds)
    _, _, xtrue, ytrue = theta_to_cart(q_next)
    plt.figure()
    plt.scatter(xpred, ypred, label='predicted')
    plt.scatter(xtrue, ytrue, label='true')
    plt.legend()
    plt.show()

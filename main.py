"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import *
from utils import *
from model import *
from model2 import *
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

# %%
from torchdiffeq import odeint_adjoint as odeint

num_trajectories = 1
n_test_traj = 1
num_nodes = 2
T_max = 5.01
dt = 0.01
srate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = pendulum(num_trajectories, T_max, dt, srate, noise_std=0, seed=3)
valid_data = pendulum(n_test_traj, T_max, dt, srate, noise_std=0, seed=5)

print(valid_data['energy'])
tnow, tnext,tenergy,_ = nownext(train_data, num_trajectories, T_max, dt, srate)
vnow, vnext,venergy,_ = nownext(valid_data, n_test_traj, T_max, dt, srate)
traindat = pendpixdata(tnow, tnext,tenergy)
train_dataloader = DataLoader(traindat, batch_size=200, num_workers=2, shuffle=False)
valdat = pendpixdata(vnow, vnext,venergy)
val_dataloader = DataLoader(valdat, batch_size=200, num_workers=2, shuffle=False)

data_dict = {'train': train_dataloader, 'valid': val_dataloader}
running_losses = 0.
loss_collater = {'train': [], 'valid': []}

torch.pi = torch.tensor(np.pi)



def train_model(model, optimizer, num_epochs=1,energy_term=False,adj_method=False,reg_grad=False):
    for epoch in range(num_epochs):
        print('epoch:{}'.format(epoch))

        # Each epoch has a training and validation phase
        for phase in ['train','valid']:
            if phase == 'train':
                # scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for batch_i, (q, q_next,energy_) in enumerate(data_dict[phase]):
                if phase == 'train':
                    optimizer.zero_grad()
                q, q_next = q.float(), q_next.float()
                if adj_method:
                    batchsz = len(q)
                    teval = torch.linspace(0, 0.01 * (batchsz - 1),batchsz)
                    q.to(device)
                    q_next.to(device)
                    teval.to(device)
                    energy_.to(device)
                    loss = 0
                #q.requires_grad = True
                    next_step_pred = odeint(model,q[0].reshape(1,2),teval,method='dopri5')#model.next_step(q)
                    state_loss = torch.mean((next_step_pred - q)**2) #+ (1/100)*torch.norm(model.time_deriv(q))
                else:
                    q.to(device)
                    q_next.to(device)
                    energy_.to(device)
                    batchsz = len(q)
                    teval = torch.linspace(0, 0.01 * (batchsz - 1), batchsz)
                    teval.to(device)
                    loss = 0
                    q.requires_grad = True
                    teval.requires_grad = True
                    next_step_pred = model.next_step(q,teval)
                    state_loss = torch.mean((next_step_pred - q_next) ** 2)

                if energy_term:
                    f1 = model.get_H(q)
                    energy_loss = ((f1-energy_)**2).mean()
                    loss += state_loss   + energy_loss
                    print(f'state loss:{state_loss},energy loss:{energy_loss}')
                else:
                    loss = state_loss
                    print(f'state loss:{state_loss}')

                lambda_ = 1e-2

                if reg_grad:
                    derivs = model.time_deriv2(q,teval)
                    regloss = torch.mean(torch.abs(derivs))
                    print(regloss)
                    loss += lambda_* regloss


                    #torch.mean((model.time_deriv(q))**2)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss

            loss_collater[phase].append(running_loss)
            epoch_loss = running_loss
            print('{} Loss: {:.10f}'.format(phase, epoch_loss))

    plt.figure()
    plt.plot(loss_collater['train'], label='train')
    plt.plot(loss_collater['valid'], label='valid')
    plt.yscale('log')
    if energy_term:
        plt.title(f'simple pendulum w energy loss: ntrain_inits:{num_trajectories},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
    else:
        plt.title(f'simple pendulum: ntrain_inits:{num_trajectories},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
    plt.legend()
    plt.show()
    print(model.get_H(torch.cat([q,teval.reshape(-1,1)],1)))
    return model


# model_ft = HNN(2, 200, 1, 0.01)
model_ft = HNN(3,200,1,0.01)
params = list(model_ft.parameters())
optimizer_ft = torch.optim.Adam(params, 1e-3)
model_ft = train_model(model_ft, optimizer_ft, num_epochs=100,energy_term=False,adj_method=False,reg_grad=True)

"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import *
from utils import *
from model import *
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

# %%


num_trajectories = 10
n_test_traj = 1
num_nodes = 2
T_max = 1.01
dt = 0.01
srate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = pendulum(num_trajectories, T_max, dt, srate, noise_std=0, seed=3)
valid_data = pendulum(n_test_traj, T_max, dt, srate, noise_std=0, seed=5)

print(valid_data['energy'])
tnow, tnext,tenergy,_ = nownext(train_data, num_trajectories, T_max, dt, srate)
vnow, vnext,venergy,_ = nownext(valid_data, n_test_traj, T_max, dt, srate)
# print(tnow.shape)
# plt.scatter(tnow[:, 0], tnow[:, 1], s=100)
# plt.show()
traindat = pendpixdata(tnow, tnext,tenergy)
train_dataloader = DataLoader(traindat, batch_size=100, num_workers=2, shuffle=False)
valdat = pendpixdata(vnow, vnext,venergy)
val_dataloader = DataLoader(valdat, batch_size=200, num_workers=2, shuffle=False)

data_dict = {'train': train_dataloader, 'valid': val_dataloader}
running_losses = 0.
loss_collater = {'train': [], 'valid': []}

torch.pi = torch.tensor(np.pi)


# def rebase(q):
#     #     return q
#     qs = torch.fmod((q[:, :2] + torch.pi), (2 * torch.pi)) - torch.pi
#     qdots = q[:, 2:]
#     return torch.cat([qs, qdots], 1)


def train_model(model, optimizer, num_epochs=1,energy_term=False):
    for epoch in range(num_epochs):
        print('epoch:{}'.format(epoch))

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
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
                q.to(device)
                q_next.to(device)
                energy_.to(device)
                q.requires_grad = True
                loss = 0

                next_step_pred = model.next_step(q)
                state_loss = ((next_step_pred - q_next) ** 2).mean()
                if energy_term:
                    f1 = model.forward(q)
                    energy_loss = ((f1-energy_)**2).mean()
                    loss += state_loss   + energy_loss
                    print(f'state loss:{state_loss},energy loss:{energy_loss}')
                else:
                    loss += state_loss
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
    print(model.forward(q))
    return model


model_ft = HNN(2, 200, 1, 0.01)
params = list(model_ft.parameters())
optimizer_ft = torch.optim.Adam(params, 1e-2)
model_ft = train_model(model_ft, optimizer_ft, num_epochs=100)

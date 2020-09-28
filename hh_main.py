"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import *
from utils import *
from hh_model import *
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

num_trajectories = 1
n_test_traj = 1
num_nodes = 2
T_max = 5.01
T_max_t = 10.01
dt = 0.01
srate = 0.01
noise_std = 0.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = dpend_adapted(num_trajectories, T_max, dt, srate, 3)
# train_data = heinon_heiles('name', num_trajectories, num_nodes, T_max, dt, srate, noise_std, 3)#pendulum(num_trajectories, T_max, dt, srate, noise_std=0, seed=3)
valid_data = dpend_adapted(n_test_traj, T_max_t, dt, srate, 4)
# valid_data =heinon_heiles('name', n_test_traj, num_nodes, T_max_t, dt, srate, noise_std, 10)#pendulum(n_test_traj, T_max, dt, srate, noise_std=0, seed=3)
print(train_data['x'].shape, train_data['dx'].shape)
tnow, tnext, tenergy, dtnow, t_time = nownext(train_data, num_trajectories, T_max, dt, srate)
vnow, vnext, venergy, dvnow, v_time = nownext(valid_data, n_test_traj, T_max_t, dt, srate)
traindat = pendpixdata(tnow, tnext, tenergy, dtnow, t_time)
train_dataloader = DataLoader(traindat, batch_size=200, num_workers=2, shuffle=True)
valdat = pendpixdata(vnow, vnext, venergy, dvnow, v_time)
val_dataloader = DataLoader(valdat, batch_size=1000, num_workers=2, shuffle=False)

data_dict = {'train': train_dataloader, 'valid': val_dataloader}
running_losses = 0.
loss_collater = {'train': [], 'valid': []}

torch.pi = torch.tensor(np.pi)

# _, _, xpred, ypred = theta_to_cart(vnow)
# plt.figure()
# plt.scatter(xpred, ypred, label='predicted')
# plt.title(f'initial_traj: ntrain_inits:{num_trajectories},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
# plt.legend()
# plt.show()


def renorm(xnow):
    # firsts = torch.remainder(xnow[:, :2]+torch.pi, 2 * torch.pi) - torch.pi
    # return torch.cat([firsts, xnow[:, 2:]], 1)
    return xnow


def train_model(model, optimizer, scheduler, num_epochs=1, integrator_embedded=True):
    for epoch in range(num_epochs):
        print('epoch:{}'.format(epoch))
        scheduler.step()
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for batch_i, (q, q_next, energy_, dq, tevals) in enumerate(data_dict[phase]):
                if phase == 'train':
                    optimizer.zero_grad()
                q, q_next, dq, tevals = q.float(), q_next.float(), dq.float(), tevals.float()

                q.to(device)
                q_next.to(device)
                energy_.to(device)
                dq.to(device)
                tevals.to(device)
                q.requires_grad = True
                # print(len(q_next))
                if integrator_embedded:
                    next_step_pred = model.next_step(renorm(q))
                    state_loss = torch.mean((renorm(next_step_pred) - renorm(q_next)) ** 2)
                else:
                    curr_deriv = model.time_deriv(renorm(q))
                    state_loss = torch.mean((curr_deriv - dq) ** 2)
                loss = state_loss
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss
                print('{} Loss: {:.10f}'.format(phase, loss))
            loss_collater[phase].append(running_loss)
            epoch_loss = running_loss
            print('{} Epoch Loss: {:.10f}'.format(phase, epoch_loss))

    plt.figure()
    plt.plot(loss_collater['train'], label='train')
    plt.plot(loss_collater['valid'], label='valid')
    plt.yscale('log')
    plt.title(f'pendulum: ntrain_inits:{num_trajectories},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
    plt.legend()
    plt.show()
    print(model.get_H(q))
    preds = []
    qinit = q[0].reshape(1, -1)

    for i in range(len(q_next)):
        next_step_pred = model.next_step(renorm(qinit))
        preds.append(next_step_pred)
        qinit = next_step_pred
    preds = torch.cat(preds)
    print(torch.mean((preds - renorm(q_next)) ** 2))
    preds = preds.detach().numpy()
    q_next = q_next.detach().numpy()

    plt.figure()
    plt.plot(preds[:, 0], preds[:, 1], label='predicted')
    plt.plot(q_next[:, 0], q_next[:, 1], label='true')
    plt.scatter(preds[:, 0], preds[:, 1], label='predicted', s=3)
    plt.scatter(q_next[:, 0], q_next[:, 1], label='true', s=3)

    plt.title(f'dpendulum: ntrain_inits:{num_trajectories},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
    plt.legend()
    plt.show()

    _, _, xpred, ypred = theta_to_cart(preds)
    _, _, xtrue, ytrue = theta_to_cart(q_next)
    plt.figure()
    plt.scatter(xpred, ypred, label='predicted')
    plt.scatter(xtrue, ytrue, label='true')
    plt.title(f' pendulum: ntrain_inits:{num_trajectories},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
    plt.legend()
    plt.show()

    return model


model_ft = HNN(4, 500, 1, srate)
params = list(model_ft.parameters())
optimizer_ft = torch.optim.Adam(params, 1e-3)
scheduler = StepLR(optimizer_ft, step_size=30, gamma=0.1)
model_ft = train_model(model_ft, optimizer_ft, scheduler, num_epochs=50, integrator_embedded=False)
torch.save(model_ft, 'mdl_s1_coord')

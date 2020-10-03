"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import *
from utils import *
from model import *
import numpy as np
import matplotlib.pyplot as plt

# %%

num_trajectories = 100
n_test_traj = 20
num_nodes = 2
T_max = 3.05
T_max_t = 10.05
dt = 0.05
srate = 0.05
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = pendulum(num_trajectories, T_max, dt, srate, noise_std=0, seed=3)
valid_data = pendulum(n_test_traj, T_max_t, dt, srate, noise_std=0, seed=5)

tnow, tnext, tenergy, tdx,tevals = nownext(train_data, num_trajectories, T_max, dt, srate)
vnow, vnext, venergy, vdx,vevals = nownext(valid_data, n_test_traj, T_max_t, dt, srate)


traindat = pendpixdata(tnow, tnext, tenergy, tdx,tevals)
train_dataloader = DataLoader(traindat, batch_size=200, num_workers=2, shuffle=True)
valdat = pendpixdata(vnow, vnext, venergy, vdx,vevals)
val_dataloader = DataLoader(valdat, batch_size=int(T_max_t//srate-1), num_workers=2, shuffle=False)

data_dict = {'train': train_dataloader, 'valid': val_dataloader}
running_losses = 0.
loss_collater = {'train': [], 'valid': []}

torch.pi = torch.tensor(np.pi)


def train_model(model, optimizer, num_epochs=1, energy_term=False, integrator_embedded=False, reg_grad=False):
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
            for batch_i, (q, q_next, energy_, qdx,tevals) in enumerate(data_dict[phase]):
                if phase == 'train':
                    optimizer.zero_grad()
                q, q_next, qdx = q.float(), q_next.float(), qdx.float()
                q.to(device)
                q_next.to(device)
                energy_.to(device)
                qdx.to(device)
                tevals = tevals.float()
                tevals.to(device)
                loss = 0
                q.requires_grad = True
                tevals.requires_grad = True

                if integrator_embedded:
                    next_step_pred = model.next_step(q,tevals)
                    state_loss = torch.mean((next_step_pred - q_next) ** 2)
                else:
                    next_step_pred = model.time_deriv(q,tevals)
                    state_loss = torch.mean((next_step_pred - qdx) ** 2)

                loss = state_loss
                print(f'{phase} state loss {state_loss}')

                beta = 1e-5

                if energy_term:
                    f1 = model.get_H(torch.cat([q,tevals.reshape(-1,1)],1))
                    energy_loss = ((f1 - energy_) ** 2).mean()
                    print(f'{phase} energy loss {energy_loss}')
                    loss += beta*energy_loss

                lambda_ = 1

                if reg_grad:
                    derivs = model.time_deriv2(q, tevals)
                    regloss = torch.mean(torch.abs(derivs))
                    print(f'{phase} reg loss {regloss}')
                    loss += lambda_ * regloss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss

            loss_collater[phase].append(running_loss)
            epoch_loss = running_loss
            print('{} Loss: {:.10f}'.format(phase, epoch_loss))

    # plt.figure()
    # plt.plot(loss_collater['train'], label='train')
    # plt.plot(loss_collater['valid'], label='valid')
    # plt.yscale('log')
    # plt.title(f'simple pendulum: ntrain_inits:{num_trajectories},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
    # plt.legend()
    # plt.show()
    #
    #
    #
    # preds = []
    # pred_ham = []
    # qinit = q[0].reshape(1, -1)
    # for i in range(len(q_next)):
    #     next_step_pred = model.next_step(qinit,tevals[i])
    #     pred_ham.append(model.get_H(torch.cat([qinit,tevals[i].reshape(-1,1)],1)))
    #     preds.append(next_step_pred)
    #     qinit = next_step_pred
    #
    # pred_ham = torch.cat(pred_ham)
    #
    # plt.figure()
    # plt.scatter(range(len(q_next)),pred_ham.detach().numpy())
    # plt.show()
    #
    # preds = torch.cat(preds)
    # preds = preds.detach().numpy()
    # q_next = q_next.detach().numpy()
    #
    # plt.figure()
    # plt.scatter(range(len(preds)),np.mean(np.abs(preds-q_next),1))
    # plt.show()
    #
    #
    # print(np.mean((preds-q_next)**2))
    # plt.figure()
    # plt.plot(preds[:, 0], preds[:, 1], label='predicted')
    # plt.plot(q_next[:, 0], q_next[:, 1], label='true')
    # plt.scatter(preds[:, 0], preds[:, 1], label='predicted', s=3)
    # plt.scatter(q_next[:, 0], q_next[:, 1], label='true', s=3)
    # plt.legend()
    # plt.show()
    return model
# model_ft = HNN(2, 200, 1, 0.01)
model_ft = HNN(3, 200, 1, srate)
params = list(model_ft.parameters())
optimizer_ft = torch.optim.Adam(params, 1e-3,weight_decay=1e-4)
trained_model = train_model(model_ft, optimizer_ft, num_epochs=100, energy_term=False,integrator_embedded=True, reg_grad=False)

torch.save(trained_model,'HNN')
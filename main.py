"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import get_dataset
from utils import *
from model_builder import get_models
from tensorboardX import SummaryWriter
import os
import torch
import argparse
import matplotlib.pyplot as plt
from time import process_time

parser = argparse.ArgumentParser()
parser.add_argument('-ni', '--num_iters', type=int, default=1000)
parser.add_argument("-n_test_traj", '--ntesttraj', type=int, default=25)
parser.add_argument("-n_train_traj", '--ntraintraj', type=int, default=25)
parser.add_argument('-dt', '--dt', type=float, default=0.1)
parser.add_argument('-tmax', '--tmax', type=float, default=3.1)
parser.add_argument('-dname', '--dname', type=str, default='mass_spring')
parser.add_argument('-noise_std', '--noise', type=float, default=0)
parser.add_argument('-type','--type',type=int,default=1)
args = parser.parse_args()
iters = args.num_iters
n_test_traj = args.ntesttraj
n_train_traj = args.ntraintraj
T_max = args.tmax
T_max_t = T_max
dt = args.dt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
type_vec = args.type
num_samples_per_traj = int(np.ceil((T_max / dt))) - 1

if args.noise != 0:
    noisy = True
else:
    noisy = False

dataset_name = args.dname

# dataset preprocessing
train_data = get_dataset(dataset_name, n_train_traj, T_max, dt, noise_std=args.noise, seed=0,type=type_vec)
valid_data = get_dataset(dataset_name, n_test_traj, T_max_t, dt, noise_std=0, seed=1,type=type_vec)
BS = num_samples_per_traj

tnow, tnext, tenergy, tdx, tevals = nownext(train_data, n_train_traj, T_max, dt, dt)
vnow, vnext, venergy, vdx, vevals = nownext(valid_data, n_test_traj, T_max_t, dt, dt)

print_every = 1000

traindat = pendpixdata(tnow, tnext, tenergy, tdx, tevals)
train_dataloader = DataLoader(traindat, batch_size=1500, num_workers=2, shuffle=True)
valdat = pendpixdata(vnow, vnext, venergy, vdx, vevals)
val_dataloader = DataLoader(valdat, batch_size=1500, num_workers=2, shuffle=False)

data_dict = {'train': train_dataloader, 'valid': val_dataloader}
running_losses = 0.
loss_collater = {'train': [], 'valid': []}




def train_model(model_name,model, optimizer, lr_sched, num_epochs=1, integrator_embedded=False):
    for epoch in range(num_epochs):
        print('epoch:{}'.format(epoch))

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # scheduler.step()
                model.train()
            else:
                model.eval()
            lr_sched.step()
            running_loss = 0.0
            # Iterate over data.
            for batch_i, (q, q_next, energy_, qdx, tevals) in enumerate(data_dict[phase]):
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
                    next_step_pred = model.next_step(q, tevals)
                    state_loss = torch.mean((next_step_pred - q_next) ** 2)
                else:
                    next_step_pred = model.time_deriv(q, tevals)
                    state_loss = torch.mean((next_step_pred - qdx) ** 2)

                loss = state_loss
                print(f'{phase} state loss {state_loss}')

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += state_loss

            loss_collater[phase].append(running_loss)
            epoch_loss = running_loss
            print('{} Loss: {:.10f}'.format(phase, epoch_loss))

    plt.figure()
    plt.plot(loss_collater['train'], label='train')
    plt.plot(loss_collater['valid'], label='valid')
    plt.yscale('log')
    plt.title(f'{dataset_name},{model_name}, ntrain_inits:{n_train_traj},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
    plt.legend()
    plt.savefig('test.jpg')

    return model


# model_ft = HNN(2, 200, 1, 0.01)
model_dct = get_models(dt, type=None, hidden_dim=200)
for model_name, model_type in model_dct.items():
    params = list(model_type.parameters())
    optimizer_ft = torch.optim.Adam(params, 1e-3,weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer_ft, 1000, gamma=0.1)
    trained_model = train_model(model_name,model_type, optimizer_ft, lr_sched, num_epochs=iters, integrator_embedded=False)
    parent_dir = os.getcwd()
    path = f"{dataset_name}/{model_name}"
    if not os.path.exists(path):
        os.makedirs(parent_dir+'/'+path)
    torch.save(trained_model, path+'/'+'model')
    del trained_model
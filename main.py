"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import get_dataset
from utils import *
from model_builder import get_models
import pickle
import os
import torch
import argparse
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as rk
solve_ivp = rk

parser = argparse.ArgumentParser()
parser.add_argument('-ni', '--num_iters', type=int, default=1000)
parser.add_argument("-n_test_traj", '--ntesttraj', type=int, default=25)
parser.add_argument("-n_train_traj", '--ntraintraj', type=int, default=25)
parser.add_argument('-dt', '--dt', type=float, default=0.1)
parser.add_argument('-tmax', '--tmax', type=float, default=3.1)
parser.add_argument('-dname', '--dname', type=str, default='mass_spring')
parser.add_argument('-noise_std', '--noise', type=float, default=0)
parser.add_argument('-type','--type',type=int,default=1)
parser.add_argument('-batch_size','--batch_size',type=int,default=2000)
parser.add_argument('-learning_rate','--learning_rate',type=float,default=1e-3)
args = parser.parse_args()
iters = args.num_iters
n_test_traj = args.ntesttraj
n_train_traj = args.ntraintraj
T_max = args.tmax
T_max_t = T_max
dt = args.dt
device = 'cpu'#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
type_vec = args.type
num_samples_per_traj = int(np.ceil((T_max / dt))) - 1


lr_step = iters//2

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
train_dataloader = DataLoader(traindat, batch_size=len(tnow), num_workers=2, shuffle=True)
valdat = pendpixdata(vnow, vnext, venergy, vdx, vevals)
val_dataloader = DataLoader(valdat, batch_size=len(vnow), num_workers=2, shuffle=False)

data_dict = {'train': train_dataloader, 'valid': val_dataloader}
running_losses = 0.
loss_collater = {'train': [], 'valid': []}




def train_model(model_name,model, optimizer, lr_sched, num_epochs=1, integrator_embedded=False):
    phase ='train'

    #load all training data
    for batch_i, (q, q_next, _, qdx, tevals) in enumerate(data_dict[phase]):
        q, q_next, qdx = q.float(), q_next.float(), qdx.float()
        q=q.to(device)
        q_next=q_next.to(device)
        qdx=qdx.to(device)
        tevals = tevals.float()
        tevals =tevals.to(device)
        q.requires_grad = True
        tevals.requires_grad = True

    #iterate over batches - epochs here is proxy for bs
    for epoch in range(num_epochs):
        ixs = torch.randperm(q.shape[0])[:args.batch_size]

        running_loss = 0
        print('epoch:{}'.format(epoch))
        optimizer.zero_grad()

        if integrator_embedded:
            next_step_pred = model.next_step(q[ixs], tevals[ixs])
            state_loss = torch.mean((next_step_pred - q_next[ixs]) ** 2)
        else:
            next_step_pred = model.time_deriv(q[ixs], tevals[ixs])
            state_loss = (next_step_pred - qdx[ixs]).pow(2).mean()

        loss = state_loss

        # if model_name =='TDHNN3':
        #     # print(model.get_weight())
        #     loss += 1e-4*torch.mean(torch.abs(model.get_F(tevals[ixs].reshape(-1,1))))
        #     loss += 1e-4*torch.mean(torch.abs(model.get_D(q[ixs,0].reshape(-1,1))))
        if model_name =='TDHNN4':
            # print(model.get_weight())
            loss += 1e-5*torch.mean(torch.abs(model.get_F(tevals[ixs].reshape(-1,1))))
            loss += 1e-5*torch.mean(torch.abs(model.get_D()))
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        loss_collater[phase].append(running_loss)
        epoch_loss = running_loss
        print('{} Epoch Loss: {:.10f}'.format(phase, epoch_loss))

        lr_sched.step()
    plt.figure()
    plt.plot(loss_collater['train'], label='train')
    plt.plot(loss_collater['valid'], label='valid')
    plt.yscale('log')
    plt.title(f'{dataset_name},{model_name}, ntrain_inits:{n_train_traj},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
    plt.legend()
    plt.savefig(f'{dataset_name}_{type_vec}_training.jpg')

    return model


def integrate_model(model, t_span, y0, t_eval, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 2).to(device)
        t = torch.tensor(t, requires_grad=True, dtype=torch.float32).view(1, 1).to(device)
        dx = model.time_deriv(x, t).data.cpu().numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval, **kwargs)


def test_model(model_name, model):

    # Each epoch has a training and validation phase
    for phase in ['valid']:
        for batch_i, (q, q_next, energy_, qdx, tevals) in enumerate(data_dict[phase]):
            q, q_next, qdx = q.float(), q_next.float(), qdx.float()
            q.to(device)
            q_next.to(device)
            qdx.to(device)
            tevals = tevals.float()
            tevals.to(device)
            q.requires_grad = True
            tevals.requires_grad = True

            qinit = q[0].reshape(1, -1)

            preds = integrate_model(model, [0, T_max_t], qinit.detach().numpy().ravel(),
                                    t_eval=np.arange(0, T_max_t, dt)).y

            main_pred[model_name].append(((preds.T)[:-1], q.detach().numpy()))



model_dct = get_models(dt, type=None, hidden_dim=200)
# main_pred = {'baseline': [], 'HNN': [], 'TDHNN': [],  'TDHNN2': []}

for model_name, model_type in model_dct.items():
    model_type = model_type.to(device)
    params_a = list(model_type.parameters())[:]
    # params_b = list(model_type.parameters())[1:]
    optimizer_ft = torch.optim.Adam([{"params": params_a},
          # {"params": params_b, "lr": 1e-1}
         ],
         args.learning_rate)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer_ft,lr_step, gamma=0.1)
    trained_model = train_model(model_name,model_type, optimizer_ft, lr_sched, num_epochs=iters, integrator_embedded=False)
    parent_dir = os.getcwd()
    path = f"{dataset_name}_{type_vec}/{model_name}"
    if not os.path.exists(path):
        os.makedirs(parent_dir+'/'+path)
    torch.save(trained_model, path+'/'+'model')
    # test_model(model_name,trained_model)
    del trained_model

# f = open(f"main_pred_{dataset_name}.pkl","wb")
# pickle.dump(main_pred,f)
# f.close()
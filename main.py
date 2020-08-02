"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import *
from graph_models_v2 import *
from utils import *
from tensorboardX import SummaryWriter
import argparse
from time import process_time

parser = argparse.ArgumentParser()
parser.add_argument('-ni', '--num_iters', type=int, default=10000)
parser.add_argument("-n_test_traj", '--ntesttraj', type=int, default=20)
parser.add_argument("-n_train_traj", '--ntraintraj', type=int, default=10)
parser.add_argument('-srate', '--srate', type=float, default=0.4)
parser.add_argument('-dt', '--dt', type=float, default=0.4)
parser.add_argument('-tmax', '--tmax', type=float, default=20.4)
parser.add_argument('-integrator', '--integrator', type=str, default='rk4')
parser.add_argument('-save_name', '--save_name', type=str, default='trials_noise')
parser.add_argument('-num_nodes', '--num_nodes', type=int, default=1)
parser.add_argument('-dname', '--dname', type=str, default='n_grav')
parser.add_argument('-noise_std', '--noise', type=float, default=0)
args = parser.parse_args()
num_nodes = args.num_nodes
iters = args.num_iters
n_test_traj = args.ntesttraj
num_trajectories = args.ntraintraj
T_max = args.tmax
dt = args.dt
srate = args.srate
# -1 due to down sampling

num_samples_per_traj = int(np.ceil((T_max / dt) / (srate / dt))) - 1
integ = args.integrator
if args.noise != 0:
    noisy = True
else:
    noisy = False
dataset_name = args.dname
expt_name = args.save_name

# dataset preprocessing
train_data = get_dataset(dataset_name, expt_name, num_trajectories, num_nodes, T_max, dt, srate, args.noise, 0)
valid_data = get_dataset(dataset_name, expt_name, n_test_traj, num_nodes, T_max, dt, srate, 0, 1)
BS = num_samples_per_traj
# dimension of a single particle, if 1D, spdim is 2
spdim = int(train_data['x'][0].shape[0] / num_nodes)
xnow, xnext, dxnow = nownext(train_data, num_trajectories, num_nodes, T_max, dt, srate,
                             spatial_dim=spdim)
newmass = np.repeat(train_data['mass'], num_samples_per_traj, axis=0)
newks = np.repeat(train_data['ks'], num_samples_per_traj, axis=0)
test_xnow, test_xnext, test_dxnow = nownext(valid_data, n_test_traj, num_nodes, T_max, dt, srate,
                                            spatial_dim=spdim)
test_mass = np.repeat(valid_data['mass'], num_samples_per_traj, axis=0)
test_ks = np.repeat(valid_data['ks'], num_samples_per_traj, axis=0)
tot_train_samples = int(xnow.shape[0] / num_nodes)
Tot_iters = int(tot_train_samples / (BS))
num_training_iterations = int(iters / Tot_iters)
print('xnow:{},xnext:{}'.format(xnow.shape, xnext.shape))
print('tot_samples:{},tot_iters:{},ntrain_iters:{}'.format(tot_train_samples, Tot_iters, num_training_iterations))

print_every = 1000

# model loop settings
model_types = ['graphic']
# classic_methods = ['baseline', 'hnn', 'vin']
if integ == 'rk1':
    graph_methods = ['vin_rk1_lr']
if integ == 'rk4':
    graph_methods = ['dgn','hnn','vin_rk4']

lr_stack = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
for model_type in model_types:
    if model_type == 'graphic':
        error_collector = np.zeros((len(lr_stack), len(graph_methods), n_test_traj))
        for lr_index, sublr in enumerate(lr_stack):
            for gm_index, graph_method in enumerate(graph_methods):
                data_dir = 'data/' + dataset_name + '/' + str(sublr) + '/'+ graph_method + '/'
                if not os.path.exists(data_dir):
                    print('non existent path....creating path')
                    os.makedirs(data_dir)
                dirw = graph_method + str(sublr) + integ
                if noisy:
                    writer = SummaryWriter('noisy/' + dataset_name + '/' + str(sublr) + '/' + graph_method +'/' + dirw)
                else:
                    writer = SummaryWriter('noiseless/' + dataset_name + '/' + str(sublr) + '/' + graph_method +'/' + dirw)
                try:
                    sess.close()
                except NameError:
                    pass

                tf.compat.v1.reset_default_graph()
                sess = tf.Session()
                gm = graph_model(sess, graph_method, num_nodes, BS, integ, expt_name, sublr, noisy, spdim, srate, True)
                sess.run(tf.global_variables_initializer())
                saver = tf.compat.v1.train.Saver()
                xvec = np.arange(0, tot_train_samples, 1, dtype=int)

                for iteration in range(num_training_iterations):
                    # if graph_method != 'vin_rk4_lr':
                    #     np.random.shuffle(xvec)
                    for sub_iter in range(Tot_iters):
                        input_batch = np.vstack(
                            [xnow[xvec[i] * num_nodes:xvec[i] * num_nodes + num_nodes] for i in
                             range(sub_iter * BS, (sub_iter + 1) * BS)])
                        true_batch = np.vstack(
                            [xnext[xvec[i] * num_nodes:xvec[i] * num_nodes + num_nodes] for i in
                             range(sub_iter * BS, (sub_iter + 1) * BS)])
                        ks_true = np.vstack([newks[xvec[i]] for i in range(sub_iter * BS, (sub_iter + 1) * BS)])
                        ms_true = np.vstack([newmass[xvec[i]] for i in range(sub_iter * BS, (sub_iter + 1) * BS)])
                        #t1_start = process_time()
                        loss = gm.train_step(input_batch, true_batch, ks_true, ms_true)
                        #t1_end = process_time()
                        writer.add_scalar('train_loss', loss, iteration * Tot_iters + sub_iter)
                        if ((iteration * Tot_iters + sub_iter) % print_every == 0):
                            print('Iteration:{},Training Loss:{:.3g}'.format(iteration * Tot_iters + sub_iter,loss))

                            #print('Time:{}'.format(t1_end - t1_start))
                            # saves model every 1000 iters (I/O slow)
                            if noisy:
                                saver.save(sess, data_dir + graph_method + str(sublr) + integ + 'noisy')
                            else:
                                saver.save(sess, data_dir + graph_method + str(sublr) + integ)

                for t_iters in range(n_test_traj):
                    input_batch = test_xnow[num_nodes * t_iters * BS:num_nodes * t_iters * BS + num_nodes]
                    true_batch = test_xnext[num_nodes * t_iters * BS:num_nodes * (t_iters + 1) * BS]
                    error, _ = gm.test_step(input_batch, true_batch, np.reshape(test_ks[t_iters * BS], [1, -1]),
                                            np.reshape(test_mass[t_iters * BS], [1, -1]), BS)
                    error_collector[lr_index, gm_index, t_iters] = error
                print('mean test error:{}'.format(error_collector[lr_index, :, :].mean(1)))
                print('std test error:{}'.format(error_collector[lr_index, :, :].std(1)))
            if noisy:
                np.save('data/' + dataset_name + '/collater_noisy.npy', error_collector)
            else:
                np.save('data/' + dataset_name + '/collater.npy', error_collector)
    try:
        sess.close()
    except NameError:
        pass

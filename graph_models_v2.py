from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import math as m

class graph_model(object):
    def __init__(self, sess, deriv_method, num_nodes, BS, integ_meth, expt_name, lr,dt,
                 noisy=False, spatial_dim=4):

        self.sess = sess
        self.deriv_method = deriv_method
        self.num_nodes = num_nodes
        self.BS = BS
        self.BS_test = 1
        self.integ_method = integ_meth
        self.expt_name = expt_name
        self.lr = lr
        self.spatial_dim = spatial_dim
        self.dt = dt
        self.hidden_dim = 20
        self.is_noisy = noisy
        self.log_noise_var = None
        self.activate_sub = False
        self._build_net()

    def _build_net(self):


        self.out_to_global = snt.Linear(output_size=1, use_bias=False, name='out_to_global')
        self.out_to_node = snt.Linear(output_size=self.spatial_dim, use_bias=False, name='out_to_node')

        self.graph_network = modules.GraphNetwork(
            edge_model_fn=lambda: snt.nets.MLP([32, 32], activation=tf.nn.softplus, activate_final=True),
            node_model_fn=lambda: snt.nets.MLP([32, 32], activation=tf.nn.softplus, activate_final=True),
            global_model_fn=lambda: snt.nets.MLP([32, 32], activation=tf.nn.softplus, activate_final=True),
        )

        self.pi = tf.constant(m.pi)

        self.base_graph_tr = tf.compat.v1.placeholder(tf.float32,
                                                      shape=[self.num_nodes * self.BS, self.spatial_dim])
        self.ks_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS, self.num_nodes])
        self.ms_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS, self.num_nodes])
        self.true_dq_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, self.spatial_dim])

        self.test_graph_ph = tf.compat.v1.placeholder(tf.float32,
                                                      shape=[self.num_nodes * self.BS_test, self.spatial_dim])
        self.test_ks_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS_test, self.num_nodes])
        self.test_ms_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS_test, self.num_nodes])


    def create_loss_ops(self, true, predicted):
        loss_ops = tf.reduce_mean((true - predicted) ** 2)
        return loss_ops

    def base_graph(self, input_features, ks, ms, num_nodes, extra_flag=True):
        # Node features for graph 0.
        if extra_flag:
            nodes_0 = tf.concat([input_features, tf.reshape(ms, [num_nodes, 1]), tf.reshape(ks, [num_nodes, 1])], 1)
        else:
            nodes_0 = input_features

        globals_0 = [9.81]

        senders_0 = []
        receivers_0 = []
        edges = []
        an = np.arange(0, num_nodes, 1)
        for i in range(len(an)):
            for j in range(i + 1, len(an)):
                senders_0.append(i)
                senders_0.append(j)
                receivers_0.append(j)
                receivers_0.append(i)
        data_dict_0 = {
            "nodes": nodes_0,
            "senders": senders_0,
            "receivers": receivers_0,
            'globals': globals_0
        }

        return data_dict_0

    def rk4_vin_dpend_fixed(self, dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
        vdim = int(self.spatial_dim / 2)

        init_q, init_qdot = x_t[:, :vdim], x_t[:, vdim:]

        p00 = dx_dt_fn(tf.concat([init_q, init_qdot], 1), ks, ms, bs, nodes)
        init_p = p00[:, vdim:]
        k0 = dt * self.new_coupler(tf.concat([init_q, init_p], 1), ks, ms, bs, nodes)
        l0 = dt * p00[:, :vdim]

        nq1 = init_q + 0.5 * k0
        np1 = init_p + 0.5 * l0
        nqd1 = self.new_coupler(tf.concat([nq1, np1], 1), ks, ms, bs, nodes)

        p01 = dx_dt_fn(tf.concat([nq1, nqd1], 1), ks, ms, bs, nodes)
        p1 = p01[:, vdim:]
        k1 = dt * nqd1
        l1 = dt * p01[:, :vdim]

        nq2 = init_q + 0.5 * k1
        np2 = init_p + 0.5 * l1
        nqd2 = self.new_coupler(tf.concat([nq2, np2], 1), ks, ms, bs, nodes)

        p02 = dx_dt_fn(tf.concat([nq2, nqd2], 1), ks, ms, bs, nodes)
        p2 = p01[:, vdim:]
        k2 = dt * nqd2
        l2 = dt * p02[:, :vdim]

        nq3 = init_q + k2
        np3 = init_p + l2
        nqd3 = self.new_coupler(tf.concat([nq3, np3], 1), ks, ms, bs, nodes)

        p03 = dx_dt_fn(tf.concat([nq3, nqd3], 1), ks, ms, bs, nodes)
        #p3 = p03[:, vdim:]
        k3 = dt * nqd3
        l3 = dt * p03[:, :vdim]

        fin_x = init_q + (1 / 6) * (k0 + 2 * k1 + 2 * k2 + k3)
        fin_p = init_p + (1 / 6) * (l0 + 2 * l1 + 2 * l2 + l3)
        fin_v = self.new_coupler(tf.concat([fin_x, fin_p], 1), ks, ms, bs, nodes)
        return tf.concat([fin_x, fin_v], 1)
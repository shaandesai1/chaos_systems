"""
generates double pendulum dataset
"""

import numpy as np
from scipy.integrate import solve_ivp as rk
import pickle 

def doublepend(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, seed, yflag=False):
    def hamiltonian(vec):
        m1, m2, l1, l2 = 1, 1, 1, 1
        g = 9.81
        theta1, theta2, theta1dot, theta2dot = vec
        V = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
        K = 0.5 * m1 * l1 ** 2 * (theta1dot) ** 2 + .5 * m2 * (
                l1 ** 2 * (theta1dot ** 2) + l2 ** 2 * (theta2dot ** 2) + 2 * l1 * l2 * theta1dot * theta2dot * np.cos(
            theta1 - theta2))
        # print(V+K)
        return V + K

    def omega(y):
        """
        Computes the angular velocities of the bobs and returns them
        as a tuple.
        """

        m1 = 1
        t1 = y[0]
        p1 = y[1]
        L1 = 1
        m2 = 1
        t2 = y[2]
        p2 = y[3]
        L2 = 1

        C0 = L1 * L2 * (m1 + m2 * np.sin(t1 - t2) ** 2)

        w1 = (L2 * p1 - L1 * p2 * np.cos(t1 - t2)) / (L1 * C0)
        w2 = (L1 * (m1 + m2) * p2 - L2 *
              m2 * p1 * np.cos(t1 - t2)) / (L2 * m2 * C0)

        return np.array([t1, w1, t2, w2])

    def f_analytical(t, state, m1, m2, l1, l2):
        g = 9.81
        t1, t2, w1, w2 = state
        a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
        a2 = (l1 / l2) * np.cos(t1 - t2)
        f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2 ** 2) * np.sin(t1 - t2) - \
             (g / l1) * np.sin(t1)
        f2 = (l1 / l2) * (w1 ** 2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)
        return np.array([w1, w2, g1, g2])

    collater = {}
    theta = []
    dtheta = []
    energy = []
    mass_arr = []
    ls_arr = []
    np.random.seed(seed)

    for traj in range(num_trajectories):

        t1init = np.random.uniform(-np.pi, np.pi)
        t2init = np.random.uniform(-np.pi, np.pi)
        y0 = np.array([t1init, t2init, 0, 0])
        if yflag:
            y0 = [-0.53202021, -0.38343444, -2.70467816, 0.98074028]
        #             y0 = np.array([3*np.pi/7,3*np.pi/4,0,0])#np.array([t1init, t2init, 0, 0])
        l1, l2, m1, m2 = [1, 1, 1, 1]
        qnrk = rk(lambda t, y: f_analytical(t, y, m1, m2, l1, l2), (0, T_max), y0,
                  t_eval=np.arange(0, T_max, dt),
                  rtol=1e-12, atol=1e-12)

        accum = qnrk.y.T
        ssr = int(sub_sample_rate / dt)
        accum = accum[::ssr]

        daccum = [f_analytical(0, accum[i], m1, m2, l1, l2) for i in range(accum.shape[0])]
        energies = []
        for i in range(accum.shape[0]):
            energies.append(hamiltonian(accum[i]))

        temp_vec = accum
        theta.append(temp_vec)
        dtheta.append(daccum)
        energy.append(energies)
        mass_arr.append([m1, m2])
        ls_arr.append([l1, l2])

    collater['x'] = np.concatenate(theta)
    collater['dx'] = np.concatenate(dtheta)
    collater['energy'] = np.concatenate(energy)
    collater['mass'] = mass_arr
    collater['ks'] = ls_arr
    f = open(name + ".pkl", "wb")
    pickle.dump(collater, f)
    f.close()

    return collater

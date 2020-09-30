"""
generates double pendulum dataset
"""

import numpy as np
from scipy.integrate import solve_ivp as rk
import autograd
from autograd.numpy import cos
import math
def pendulum(num_samples, T_max, dt, srate, noise_std=0, seed=3):
    """simple pendulum"""
    def hamiltonian_fn(coords):
        q, p = np.split(coords, 2)
        H = 9.81 * (1 - cos(q)) + (p ** 2)/2 # pendulum hamiltonian
        return H

    def hamiltonian_eval(coords):
        H = 9.81 * (1 - np.cos(coords[:,0])) + (coords[:,1] ** 2)/2 # pendulum hamiltonian
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt, radius=None, y0=None, **kwargs):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        if y0 is None:
            y0 = np.random.rand(2) * 2. - 1
        if radius is None:
            radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius

        spring_ivp = rk(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval


    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    ssr = int(srate/dt)
    ms = []
    ks = []
    energies = []
    tvalues = []
    for s in range(num_samples):
        x, y, dx, dy, t = get_trajectory()
        x = x[::ssr]
        y = y[::ssr]
        dx = dx[::ssr]
        dy = dy[::ssr]
        ms.append([1.])
        ks.append([9.81])
        xs.append(np.stack([x, y]).T)
        energies.append(hamiltonian_eval(xs[-1]))
        dxs.append(np.stack([dx, dy]).T)
        tvalues.append(t)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['mass'] = np.concatenate(ms)
    data['ks'] = np.concatenate(ks)
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = np.concatenate(tvalues)
    return data


def doublepend(num_trajectories, T_max, dt, sub_sample_rate, gen_coords = False,seed=3, yflag=False):
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

    def tdot_to_p(state):
        pt1 = state[:, 2] + state[:, 3] * np.cos(state[:, 0] - state[:, 1])
        pt2 = state[:, 3] + state[:, 2] * np.cos(state[:, 0] - state[:, 1])
        return np.concatenate([state[:, :2], pt1.reshape(-1, 1), pt2.reshape(-1, 1)], 1)

    def hamilton_rhs(t1, t2, p1, p2):
        """
        Computes the right-hand side of the Hamilton's equations for
        the double pendulum and returns it as an array.
        t1 - The angle of bob #1.
        t2 - The angle of bob #2.
        p1 - The canonical momentum of bob #1.
        p2 - The canonical momentum of bob #2.
        """

        m1 = 1
        L1 = 1
        m2 = 1
        L2 = 1

        g = 9.81

        C0 = L1 * L2 * (m1 + m2 * math.sin(t1 - t2) ** 2)
        C1 = (p1 * p2 * math.sin(t1 - t2)) / C0
        C2 = (m2 * (L2 * p1) ** 2 + (m1 + m2) * (L1 * p2) ** 2 -
              2 * L1 * L2 * m2 * p1 * p2 * math.cos(t1 - t2)) * \
             math.sin(2 * (t1 - t2)) / (2 * C0 ** 2)

        # F is the right-hand side of the Hamilton's equations
        F_t1 = (L2 * p1 - L1 * p2 * math.cos(t1 - t2)) / (L1 * C0)
        F_t2 = (L1 * (m1 + m2) * p2 - L2 *
                m2 * p1 * math.cos(t1 - t2)) / (L2 * m2 * C0)
        F_p1 = -(m1 + m2) * g * L1 * math.sin(t1) - C1 + C2
        F_p2 = -m2 * g * L2 * math.sin(t2) + C1 - C2

        return np.array([F_t1, F_t2, F_p1, F_p2])

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

        t1init = np.random.uniform(-np.pi/10, np.pi/10)
        t2init = np.random.uniform(-np.pi/10, np.pi/10)
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

    if gen_coords is False:
        theta_ps = tdot_to_p(collater['x'])
        theta_ps_dot = [hamilton_rhs(*theta_ps_i) for theta_ps_i in theta_ps]
        collater['x'] = theta_ps
        collater['dx'] = np.array(theta_ps_dot)
    return collater


def heinon_heiles(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, noise_std, seed):
    """heinon heiles data generator"""
    def hamiltonian_fn(coords):
        x, y, px, py = np.split(coords, 4)
        lambda_ = 1
        H = 0.5 * px ** 2 + 0.5 * py ** 2 + 0.5 * (x ** 2 + y ** 2) + lambda_ * (
                    (x ** 2) * y - (y ** 3) / 3)
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dxdt, dydt, dpxdt, dpydt = np.split(dcoords, 4)
        S = np.concatenate([dpxdt, dpydt, -dxdt, -dydt], axis=-1)
        return S

    def get_trajectory(t_span=[0, 3], timescale=0.01, ssr=sub_sample_rate, radius=None, y0=None, noise_std=0.1,
                       **kwargs):

        # get initial state
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        px = np.random.uniform(-.5, .5)
        py = np.random.uniform(-.5, .5)

        y0 = np.array([x, y, px, py])

        spring_ivp = rk(lambda t, y: dynamics_fn(t, y), t_span, y0,
                        t_eval=np.arange(0, t_span[1], timescale),
                        rtol=1e-10)
        accum = spring_ivp.y.T
        ssr = int(ssr / timescale)
        accum = accum[::ssr]
        daccum = [dynamics_fn(None, accum[i]) for i in range(accum.shape[0])]
        energies = []
        for i in range(accum.shape[0]):
            energies.append(np.sum(hamiltonian_fn(accum[i])))

        return accum, np.array(daccum), energies

    def get_dataset(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, seed=seed, test_split=0.5,
                    **kwargs):
        data = {'meta': locals()}

        # randomly sample inputs
        np.random.seed(seed)
        data = {}
        ssr = int(sub_sample_rate / dt)

        xs, dxs, energies, ks, ms = [], [], [], [], []
        for s in range(num_trajectories):
            x, dx, energy = get_trajectory(t_span=[0, T_max], timescale=dt, ssr=sub_sample_rate)

            x += np.random.randn(*x.shape) * noise_std
            dx += np.random.randn(*dx.shape) * noise_std

            xs.append(x)
            dxs.append(dx)
            energies.append(energy)
            ks.append([1])
            ms.append([1])

        data['x'] = np.concatenate(xs)
        data['dx'] = np.concatenate(dxs)
        data['energy'] = np.concatenate(energies)
        data['ks'] = np.concatenate(ks)
        data['mass'] = np.concatenate(ms)

        return data

    return get_dataset(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate)


from autograd.numpy import cos, sin


def dpend_adapted(num_trajectories, T_max, dt, sub_sample_rate, seed,yflag=False):
    """heinon heiles data generator"""

    def hamiltonian_fn(coords):
        t1, t2, pt1, pt2 = np.split(coords, 4)
        numerator = pt1 ** 2 + 2 * pt2 ** 2 - 2 * pt1 * pt2 * cos(t1 - t2)
        denominator = 2 * (1 + sin(t1 - t2) ** 2)
        H = (numerator / denominator) - 2 * 9.81 * cos(t1) - 9.81 * cos(t2)
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dxdt, dydt, dpxdt, dpydt = np.split(dcoords, 4)
        S = np.concatenate([dpxdt, dpydt, -dxdt, -dydt], axis=-1)
        return S

    def get_trajectory(t_span=[0, 3], timescale=0.01, ssr=sub_sample_rate, radius=None, y0=None, noise_std=0.1,
                       **kwargs):

        # get initial state
        t1 = np.random.uniform(-np.pi/2, np.pi/2)
        t2 = np.random.uniform(-np.pi, np.pi)
        pt1 = 0  # np.random.uniform(-np.pi/10, np.pi/10)
        pt2 = 0  # np.random.uniform(-np.pi/10, np.pi/10)

        y0 = np.array([t1, t2, pt1, pt2])

        if yflag:
            y0 = [-0.53202021, -0.38343444, -2.70467816, 0.98074028]

        spring_ivp = rk(lambda t, y: dynamics_fn(t, y), t_span, y0,
                        t_eval=np.arange(0, t_span[1], timescale),
                        rtol=1e-10)
        accum = spring_ivp.y.T
        ssr = int(ssr / timescale)
        accum = accum[::ssr]
        daccum = [dynamics_fn(None, accum[i]) for i in range(accum.shape[0])]
        energies = []
        for i in range(accum.shape[0]):
            energies.append(np.sum(hamiltonian_fn(accum[i])))
        print(energies[-1]-energies[0],len(energies))
        return accum, np.array(daccum), energies,np.arange(0, t_span[1], timescale)

    def get_dataset(num_trajectories, T_max, dt, sub_sample_rate, seed=seed, test_split=0.5,
                    **kwargs):
        data = {'meta': locals()}

        # randomly sample inputs

        data = {}
        ssr = int(sub_sample_rate / dt)

        xs, dxs, energies, ks, ms = [], [], [], [], []
        time = []
        for s in range(num_trajectories):
            x, dx, energy,times = get_trajectory(t_span=[0, T_max], timescale=dt, ssr=sub_sample_rate)

            #             x += np.random.randn(*x.shape) * noise_std
            #             dx += np.random.randn(*dx.shape) * noise_std

            xs.append(x)
            dxs.append(dx)
            energies.append(energy)
            ks.append([1])
            ms.append([1])
            time.append(times)

        data['x'] = np.concatenate(xs)
        data['dx'] = np.concatenate(dxs)
        data['energy'] = np.concatenate(energies)
        data['ks'] = np.concatenate(ks)
        data['mass'] = np.concatenate(ms)
        data['tvalues'] = np.concatenate(time)

        return data

    np.random.seed(seed)
    return get_dataset(num_trajectories, T_max, dt, sub_sample_rate)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train_data = pendulum(1,60.05,0.05,0.05, noise_std=0, seed=44)
    plt.scatter(train_data['x'][:,0],train_data['x'][:,1])
    plt.show()
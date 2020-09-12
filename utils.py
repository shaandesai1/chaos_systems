from torch.utils.data import Dataset, DataLoader
import numpy as np


def nownext(train_data, ntraj, T_max, dt, srate):
    curr_xs = []
    next_xs = []
    ener_xs = []
    curr_dxs = []
    dex = int(np.ceil(T_max / dt) / (srate / dt))
    for i in range(ntraj):
        same_batch = train_data['x'][i * dex:(i + 1) * dex, :]
        curr_x = same_batch[:-1, :]
        next_x = same_batch[1:, :]
        curr_xs.append(curr_x)
        next_xs.append(next_x)

        curr_energ = train_data['energy'][i * dex:(i + 1) * dex]
        ener_xs.append(curr_energ[:-1])
        curr_dx = train_data['dx'][i * dex:(i + 1) * dex, :][:-1, :]
        curr_dxs.append(curr_dx)


    return np.concatenate(curr_xs,0), np.concatenate(next_xs,0), np.concatenate(ener_xs,0),np.concatenate(curr_dxs)


def theta_to_cart(sub_preds):
    x1 = np.sin(sub_preds[:, 0])
    y1 = -np.cos(sub_preds[:, 0])
    x2 = x1 + np.sin(sub_preds[:, 1])
    y2 = y1 - np.cos(sub_preds[:, 1])
    return x1, y1, x2, y2


class pendpixdata(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x, nextx,energy,dx, transform=None):
        """
        Args:
            x: original data 2*28*28 * bs
            next_x: shifted time step of x
        """
        self.x = x
        self.next_x = nextx
        self.energy = energy
        self.dx = dx
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.next_x[idx], self.energy[idx], self.dx[idx]

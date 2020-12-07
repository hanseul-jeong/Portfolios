import numpy as np
import torch

def get_relative_ratio(data):
    T = np.shape(data)[1]
    rr = np.ones_like(data, dtype=np.float64)
    for t in range(T-1):
        rr[:, t+1] = data[:,t+1] / data[:,t]
    return rr

def simplex_projection(v, z=1):
    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - z) / np.arange(1, p+1))[0][-1]
    theta = np.max([0, (sv[rho] - z) / (rho+1)])
    w = (v - theta)
    w[w<0] = 0
    return w


def set_hidden(b, h, n=1, device='cpu'):
    # num_layers, batch, hidden
    hidden = torch.zeros((n, b, h), requires_grad=False, dtype=torch.float32).to(device)
    cell = torch.zeros((n, b, h), requires_grad=False, dtype=torch.float32).to(device)

    return hidden, cell

def normalize(x, columns, type='window'):

    if type == 'window':    # divide with last close of window
        c = columns.index('open')
        T, A, W, C = np.shape(x)
        return x / x[:,:,-1,c].reshape(T,A,1,1)
    elif type == 'past':    # divide with previous window
        return x / np.concatenate((x[:,:,0:1,:], x[:,:,:-1,:]), axis=2)
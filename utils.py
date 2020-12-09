import numpy as np
import torch
import math

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

def normalize(x, columns, type='window'):
    if type == 'window':    # divide with last close of window
        c = columns.index('open')
        A, T, W, C = np.shape(x)
        return x / x[:,:,-1,c].reshape(A,T,1,1)
    elif type == 'past':    # divide with previous window
        return x / np.concatenate((x[:,:,0:1,:], x[:,:,:-1,:]), axis=2)

def set_hidden(b, h, n=1, device='cpu'):
    # num_layers, batch, hidden
    hidden = torch.zeros((n, b, h), requires_grad=False, dtype=torch.float64).to(device)
    cell = torch.zeros((n, b, h), requires_grad=False, dtype=torch.float64).to(device)

    return hidden, cell

def get_logprob(x, mu, sigma):
    '''
    return log-likelihood of x in case of N(mu, sigma^2)
    '''

    constant = 1/(sigma * np.sqrt(2*math.pi))
    z = -(x - mu)**2 / (2 * sigma.pow(2))

    return constant * torch.exp(z)
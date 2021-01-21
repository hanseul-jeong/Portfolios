import numpy as np
import torch
import math

def get_relative_ratio(data):
    T = np.shape(data)[1]
    rr = np.ones_like(data, dtype=np.float32)
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

def mc_simplex(d, points):##########################################################
    """ Sample random points from a simplex with dimension d.
    :param d: Number of dimensions.
    :param points: Total number of points.
    """
    a = np.sort(np.random.random((points, d)))
    a = np.hstack([np.zeros((points,1)), a, np.ones((points,1))])
    return np.diff(a)

def normalize(x, columns, type='window'):
    if type == 'window':    # divide with last close of window
        c = columns.index('open')
        A, T, W, C = np.shape(x)
        return x / x[:,:,-1,c].reshape(A,T,1,1)
    elif type == 'past':    # divide with previous window
        return x / np.concatenate((x[:,:,0:1,:], x[:,:,:-1,:]), axis=2)

def set_hidden(b, h, n=1, device='cpu'):
    # num_layers, batch, hidden
    hidden = torch.zeros((n, b, h), requires_grad=False, dtype=torch.float32).to(device)
    cell = torch.zeros((n, b, h), requires_grad=False, dtype=torch.float32).to(device)

    return hidden, cell

def get_logprob(x, mu, sigma):
    '''
    return log-likelihood of x in case of N(mu, sigma^2)
    '''

    constant = 1/(sigma * np.sqrt(2*math.pi))
    z = -(x - mu)**2 / (2 * sigma.pow(2))

    return constant * torch.exp(z)

def get_SR(x, risk_free):
    '''
    risk-adjusted return of x based on sharpe ratio
    :param x: close prices A, T
    :param risk_free: risk free profit (float)
    :param window: range for standard deviation (int)
    :return: sharpe_ratio
    '''
    dev = x - risk_free
    avg_dev = torch.mean(dev, dim=1)
    volatility = torch.std(dev, dim=1)

    expected_return = torch.where(volatility != 0, avg_dev / volatility, torch.zeros_like(dev))
    return expected_return

def get_diff_SR(x, risk_free):
    '''
    risk-adjusted return of x based on differential sharpe ratio
    J. Moody and M. Saffel, "Reinforcement Learning for Trading", NeurIPS 1999
    :param x:
    :param risk_free:
    :return:
    '''
    A = (1-eta)*A_ + eta*R
    B = (1-eta)*B_ + eta*(R**2)
    dSt = B_ * devA - (1/2) * A_ * devB
    deta = (B_ - (A_**2) ).pow(3/2)
    D = dSt/deta
    return D

def get_pearsonCorr(x, idx, window=-1):
    '''

    :param x: input data. 4-d torch.Tensor (float32) [Assets, Times, Windows, Columns]
    :return:
    '''
    x = x[:,:,:,idx] # Close

    # rolling-window
    if window > 0:
        x = x[:,:,-window:]

    x = x.transpose(1,0)    # T, A, W

    mu = x.mean(2)
    Dev = (x - mu[:,:,None])
    Sigma = torch.sqrt(torch.sum(Dev**2, 2))

    Dev /= Sigma[:,:,None]

    A = torch.bmm(Dev, Dev.transpose(2,1))  # (T, A, W) X (T, W, A) -> (T, A, A)

    return A
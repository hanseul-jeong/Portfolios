import numpy as np

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
from utils import get_relative_ratio, simplex_projection
import numpy as np

def BAH(data):
    '''
    Buy-and-Hold method.
    :param data: M assets stock prices. np.array [MxT] (M: assets, T: times)
    :return: Cumulative return by times. np.array [T]
    '''
    data = get_relative_ratio(data)
    M, T = np.shape(data)
    b_0 = np.ones(M, dtype=np.float32)/M # [1/M, 1/M, ..., 1/M]

    cul_return = np.cumprod(data, axis=1)
    cul_return = np.matmul(b_0.reshape(1,-1), cul_return)

    return cul_return

def Best(data):
    '''
    Best stock strategy (in hindsight).
    :param data: M assets stock prices. np.array [MxT] (M: assets, T: times)
    :return: Cumulative return by times. np.array [T]
    '''
    M, T = np.shape(data)

    Growths = data[:, -1] / data[:, 0]
    idx = np.argmax(Growths)
    b_0 = np.zeros(M, dtype=np.float32)
    b_0[idx] = 1
    cul_return = np.cumprod(get_relative_ratio(data), axis=1)
    cul_return = np.dot(b_0, cul_return)

    return cul_return

def CRP(data):
    '''
    (Uniform) Constant Rebalanced Portfolio.
    :param data: M assets stock prices. np.array [MxT] (M: assets, T: times)
    :return: Cumulative return by times. np.array [T]
    '''
    data = get_relative_ratio(data)
    M, T = np.shape(data)
    b_0 = np.ones((1,M), dtype=np.float32)/M # [1/M, 1/M, ..., 1/M]

    cul_return = np.cumprod(np.dot(b_0,data))

    return cul_return

def EG(data, eta=1.5):
    '''
    Exponential Gradient.
    :param data: M assets stock prices. np.array [MxT] (M: assets, T: times)
    :param eta: learning rate. scalar float
    :return: Cumulative return by times. np.array [T]
    '''
    data = get_relative_ratio(data)
    M, T = np.shape(data)
    b = np.ones_like(data, dtype=np.float32) / M # [1/M, 1/M, ..., 1/M]
    cul_return = np.ones(T, dtype=np.float32)

    for t in range(T-1):
        prev_asset = np.sum(b[:,t-1]*data[:,t-1])
        b[:,t] = b[:,t-1]*np.exp(eta*data[:,t-1]/prev_asset)
        b[:,t] /= np.sum(b[:,t])    # Normalization
        cul_return[t+1] = cul_return[t]*(np.matmul(b[:,t], data[:,t]))

    return cul_return

def Anticor(data, w=5):
    '''
    Anti Correlation.
    :param data: M assets stock prices. np.array [MxT] (M: assets, T: times)
    :param w: window size. scalar int
    :return: Cumulative return by times. np.array [T]
    '''
    data = get_relative_ratio(data)
    M, T = np.shape(data)
    b = np.ones_like(data, dtype=np.float32)/M # initialization

    sample_n = T-(2*w)+1    # # of total - early samples :(T-(2w-1))
    y1 = np.log([data[:,t-(2*w)+1:t-w+1] for t in range(2*w-1, T)])
    y2 = np.log([data[:, t-w+1:t+1] for t in range(2*w-1, T)])

    y1_mu = np.mean(y1, axis=2).reshape(sample_n,M,1)       # expand dim
    y2_mu = np.mean(y2, axis=2).reshape(sample_n, M, 1) # expand dim
    y1_dev = y1 - y1_mu
    y2_dev = y2 - y2_mu
    y1_std = np.std(y1, axis=2, ddof=1)
    y2_std = np.std(y2, axis=2, ddof=1)
    std = np.matmul(y1_std.reshape(sample_n, M, 1), y2_std.reshape(sample_n, 1, M))

    # T-(2w) x M x M
    m_cov = np.matmul(y1_dev,np.transpose(y2_dev,[0,2,1])) / (w-1)
    m_corr = np.where (std != 0, m_cov / std, 0)

    # compare x_i > x_j
    cond = np.zeros([sample_n, M, M], dtype=bool)
    for i in range(M):
        for j in range(M):
            cond_v = np.where(y2_mu[:,i] > y2_mu[:,j], True, False).reshape(-1)
            cond[:,i,j] = cond_v

    # self-correlation
    cond = np.where(cond & (m_corr>0), True, False)
    i_corr = np.empty([sample_n, M, M], dtype=np.float32)
    j_corr = np.empty([sample_n, M, M], dtype=np.float32)
    for m in range(M):
        i_corr[:,m,:] = m_corr[:,m,m].reshape(-1,1)         # expand dim
        j_corr[:, :, m] = m_corr[:, m, m].reshape(-1, 1)    # expand dim

    i_corr = np.where(cond & (i_corr < 0), i_corr, 0)
    j_corr = np.where(cond & (j_corr < 0), j_corr, 0)

    claim = (m_corr * cond) - i_corr - j_corr
    sum_claim = np.sum(claim, axis=-1).reshape(sample_n,M,1)

    transfer = np.where(sum_claim != 0, claim / sum_claim, 0)
    transfer_ = np.zeros([sample_n, M, M], dtype=np.float32)
    # change-needed portfolio vector  (T-2w+1,M,1)
    b_ = b[:,2*w-1:].T.reshape(sample_n,M,1).copy()
    for t in range(sample_n-1):
        transfer_[t] = b_[t]*transfer[t]
        np.fill_diagonal(transfer_[t], 0)  # remove self-transfer
        adjustment = transfer_[t].sum(0) - transfer_[t].sum(1)
        b_[t+1] = b_[t] + adjustment.reshape(M,1)
    # renewal portfolio vector
    b[:, 2*w:] = b_[1:].transpose([1,0,2]).reshape(M, sample_n-1)

    cul_return = np.cumprod((b*data).sum(0))
    return cul_return, b

def OLMAR(data, w=5, eps=10):
    '''
    Online Moving Average Reversion.
    :param data: M assets stock prices. np.array [MxT] (M: assets, T: times)
    :param w: window size. scalar int
    :param eps: epsilon. scalar int
    :return: Cumulative return by times. np.array [T]
    '''
    M, T = np.shape(data)
    b = np.ones_like(data, dtype=np.float32)/M

    # [1/M] for 0 ~ w-2 // [_] for w-1 ~ T
    _ = np.mean([data[:,t-w+1:t+1]for t in range(w-1, T)], axis=-1)
    MA = np.concatenate([data[:,:w-1], _.T], axis=1)
    p_tilda = MA / data

    market_mu = np.mean(p_tilda, axis=0)
    market_dev = p_tilda - market_mu
    var = np.sum(market_dev**2, axis=0)
    for t in range(T-1):
        expected_return = np.dot(b[:, t], p_tilda[:,t])
        step = max(0, np.where(var[t]!=0, (eps - expected_return)/var[t], 0))
        b_ = b[:,t] + step*market_dev[:, t]
        b_norm = simplex_projection(b_)
        b[:,t+1] = b_norm

    cul_return = np.cumprod((b*get_relative_ratio(data)).sum(0))
    return cul_return


def WMAMR(data, w=5, eps=0.5, C=500):
    '''
    Weighted Moving Average Mean Reversion.
    :param data: M assets stock prices. np.array [MxT] (M: assets, T: times)
    :param w: window size. scalar int
    :param eps: epsilon. scalar int
    :param C: variation for step. scalar int
    :return: Cumulative return by times. np.array [T]
    '''

    M, T = np.shape(data)
    b = np.ones_like(data, dtype=np.float32) / M

    p_tilde = np.ones_like(data)
    p_tilde[:, :w-1] = data[:, :w-1]    # early samples

    # moving average
    p_tilde[:, w-1:] = np.mean([data[:, t - w+1:t+1] for t in range(w-1, T)], axis=-1).T
    market_mu = np.mean(p_tilde, axis=0)
    market_dev = p_tilde - market_mu
    var = np.sum(market_dev** 2, axis=0)

    for t in range(T - 1):
        expected_return = max(0, np.dot(b[:, t], p_tilde[:, t]) - eps)
        step = np.where(var[t] != 0, expected_return / var[t], 0)
        b_ = b[:,t] - step * market_dev[:,t]
        b[:, t + 1] = simplex_projection(b_)

    cul_return = np.cumprod((b * get_relative_ratio(data)).sum(0))
    return cul_return

import torch

def norm(reward):
    return (reward - reward.mean().detach() ) / reward.std().detach()

def transaction_cost(actions_with_cash, y, cost=0.0025, device='cuda:0'):
    T, A_ = actions_with_cash.size()

    prev_actions = (actions_with_cash[:-1,:] * y[:-1,:])/(actions_with_cash[:-1,:] * y[:-1,:]).sum(1)[:,None]
    first = torch.zeros([1,A_]).to(device)
    first[0, -1] = 1
    prev_actions = torch.cat((first, prev_actions),dim=0)
    #################################################### Something is weird ################################################
    fee = torch.abs(actions_with_cash[:,-1:] - prev_actions[:,-1:].detach())*cost

    return fee.sum(1)


def naive_logreturn(actions_with_cash, y, cost=False):
    if cost:
        c = calculate_pv_after_commission()
    else:
        c = 0.0
    return actions_with_cash * y
    # cash = actions_with_cash[:,-1:]
    # return torch.cat(((actions_with_cash[:,:]*y[:,:]*(1-c)), cash),dim=1)


def sharpe_ratio(actions, y, cost=True, risk_free=1.0):
    if cost:
        c = calculate_pv_after_commission()
    else:
        c = 0.0
    acc_profit = (actions * y*(1-c))

    excess_return = acc_profit - risk_free
    mu = excess_return.mean()
    volatility = excess_return.std()

    expected_return = torch.where(volatility != 0, mu / volatility, torch.ones_like(mu)*10000)
    return expected_return

def calculate_pv_after_commission(w_now, y, c=0.0025):
    '''

    :param w_now: investment ratio at time t    <Tensor.float32> [T, A]
    :param y: relative ratio of price   <Tensor.float32> [T, A]
    :param c: transaction cost          <float> scalar
    :return: mu. remaining ratio witho
    '''

    # rebalanced w (t0 = [0,0, ..., 0])
    w_prev = (w_now * y) / (w_now* y).sum(1)[:,None]
    w_prev[1:] = w_prev[:-1]
    w_prev[0, :-1] = 0  # No stock
    w_prev[0, -1] = 1   # All cash

    cash_now = w_now[:, -1]
    cash_prev = w_prev[:, -1]

    mu = torch.ones_like(w_now)
    T = w_now.size(0)
    for t in range(T):
        mu0 = 1
        mu1 = 1 - 2*c + c ** 2
        while abs(mu1-mu0) > 1e-10:
            mu0 = mu1
            mu1 = (1 - c * cash_prev[t] -
                   (2 * c - c ** 2) * torch.sum(torch.relu(w_prev[t, :-1] - mu1*w_now[t, :-1]))) / (1 - c * cash_now[t])
        mu[t,:-1] = mu1
    return mu

#
# mu1 = (1 - commission_rate * w0[0] -
#        (2 * commission_rate - commission_rate ** 2) *
#        np.sum(np.maximum(w0[1:] - mu1 * w1[1:], 0))) / \
#       (1 - commission_rate * w1[0])

# def comm_wo_cash(w1, w0, commission_rate=0.0025):
#     """
#     @:param w1: target portfolio vector, first element is btc
#     @:param w0: rebalanced last period portfolio vector, first element is btc
#     @:param commission_rate: rate of commission fee, proportional to the transaction cost
#     """
#     mu0 = 1
#     mu1 = 1 - 2*commission_rate + commission_rate ** 2
#     while abs(mu1-mu0) > 1e-10:
#         mu0 = mu1
#         mu1 = (1 - (2 * commission_rate - commission_rate ** 2) * torch.sum(torch.relu(w0 - mu1*w1)))
#     return mu1
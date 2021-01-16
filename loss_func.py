import torch

def norm(reward):
    return (reward - reward.mean().detach() ) / reward.std().detach()

def transaction_cost(actions, y, cost=0.0025, device='cuda:0'):
    T, A = actions.size()
    prev_actions = (actions[:-1,:] * y[:-1,:]) / (actions[:-1,:] * y[:-1,:]).sum(1)[:,None].detach()
    next_actions = actions[1:, :]

    fee = torch.abs(next_actions - prev_actions)*cost

    return torch.cat((torch.zeros([1,A]).to(device),fee), dim=0).sum(1).mean()


def naive_logreturn(actions, y, cost=False):
    if cost:
        c = calculate_pv_after_commission()
    else:
        c = 0.0
    reward_ = -torch.log((actions*y*(1-c)).sum(1))
    return reward_.mean()


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
    """
    @:param w1: target portfolio vector, first element is btc
    @:param w0: rebalanced last period portfolio vector, first element is btc
    c: rate of commission fee, proportional to the transaction cost
    """
    # rebalanced w (t0 = [0,0, ..., 0])
    w_prev = (w_now * y)
    w_prev[1:] = w_prev[:-1]
    w_prev[0,:] = 0

    cash_prev = 1 - w_prev.sum(1)
    cash_now = 1 - w_now.sum(1)

    mu = torch.ones_like(w_now)
    T = w_now.size(0)
    for t in range(T):
        mu0 = 1
        mu1 = 1 - 2*c + c ** 2
        while abs(mu1-mu0) > 1e-10:
            mu0 = mu1
            mu1 = (1 - c * cash_prev[t] -
                   (2 * c - c ** 2) * torch.sum(torch.relu(w_prev[t] - mu1*w_now[t]))) / (1 - c * cash_now[t])
        mu[t] = mu1
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
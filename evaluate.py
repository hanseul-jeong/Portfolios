import numpy as np

def sharpe_ratio(daily_return, risk_free=1.0):
    '''

    :param daily_return: daily return (e.g., (portfolio * return ratio)_t )
    :param risk_free: risk-free profit
    :return:
    '''
    excess_return = daily_return - risk_free
    mu = np.mean(excess_return)
    volatility = np.std(excess_return)

    expected_return = np.where(volatility != 0, mu / volatility, 10000)
    return expected_return

def MDD(acc_portfolio):
    T = len(acc_portfolio)
    mdd = 1 - min([acc_portfolio[idx] / acc_portfolio[:idx + 1].max() for idx in range(T)])
    return mdd

def calmar_ratio(acc_portfolio):

    return acc_portfolio/MDD(acc_portfolio)

def winning_ratio(acc_portfolio, benchmark):
    '''

    :param acc_portfolio:
    :param benchmark: Buy-and-Hold
    :return:
    '''

    return

def DDR(sequential_profit, MAR=0.0):
    downside_dev = np.mean(np.min(sequential_profit, MAR))**2
    ddr = sequential_profit / downside_dev
    return ddr

# def turn_over(port_after, portfolio_):
#     np.sum()
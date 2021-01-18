import matplotlib.pyplot as plt
from momentum import CRP, BAH, Best, EG, Anticor, OLMAR, WMAMR
import numpy as np
import random

def show_allocation_ratio(actions_real):
    import seaborn as sns
    fig = plt.figure()
    colors = sns.color_palette("Spectral")
    # T = actions_real.size(0)
    T = np.shape(actions_real)[0]
    x = range(T)
    # y = actions_real.detach().cpu().numpy().T   # A, T
    y = actions_real.T
    plt.stackplot(x, y, colors=colors)
    plt.xlabel('Change of asset ratio', fontsize=14)
    plt.show()
    plt.close()

def show_traditional_performance():
    # generate data
    data = np.empty([3, 50], dtype=np.float64)
    base = np.array([100]*50)# + [90]*15 + [80]*10)
    constants = [15.0, 16.0,13.0]

    for m in range(3):
        random.seed(m)
        test_ = np.array([i for i in range(0, 200, 4)])/constants[m]
        noise = np.array([random.randint(-5,5) for i in range(50)])
        data[m,:] = base + test_ + noise

    data /= data[:,0:1]

    # Raw Data
    colors_d = ['Black', 'Gray', 'Lightgray']
    labels_d=['data1', 'data2', 'data3']
    f1 = plt.figure()
    for i in range(3):
        plt.plot(data[i], color=colors_d[i], label=labels_d[i])
    plt.title('Random Data', fontsize=17)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Price', fontsize=15)

    # Culmulative Assets
    funcs = [BAH, Best, EG, CRP, WMAMR, OLMAR, Anticor]
    colors_f = ['Dimgrey','Lightcoral', 'Lightseagreen', 'burlywood','Mediumslateblue','Indianred','Olive',]
    labels_f=['BAH','Best', 'EG','CRP','WMAMR', 'OLMAR', 'Anticor']
    fig = plt.figure()
    for i in range(0,len(funcs)):
        f = funcs[i]
        plt.plot(f(data).reshape(-1), color=colors_f[i], linewidth=2.0, label=labels_f[i])
    plt.legend()
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Cumulative Asset', fontsize=15)
    plt.show()

    plt.close()
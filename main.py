from data_loader import data_loader
from utils import set_hidden
from Policy_gradient import Policy_gradient
import torch
import torch.optim as optim
import numpy as np
import os
import random
from ReplayMemory import ReplayMemory


# set parameters
data_dir = 'Dataset'
data_ = 'KOSPI_daily.db'
data_path = os.path.join(data_dir, data_)
StartDate = 19950101
EndDate = 20070101
# ValidDate = 20150101
model_type = 'CNN'
n_window = 64
n_slide = 1
n_hidden = 12
n_layer = 1
epsilon = 0.9
update = 4
lr = 1e-4
n_batch = 256

n_episode = 80000
type_feature = 'OHLC'   # C or OHLC or OHLCV
gpu = 1

if model_type == 'LSTM':
    from model import LSTM_whole as Model
elif model_type == 'CNN':
    from model import CNN_whole as Model # LSTM_whole as model

device = 'cuda:{gpu}'.format(gpu=gpu) if torch.cuda.is_available() else 'cpu'

loader = data_loader(StartDate=StartDate, EndDate=EndDate, data_dir=data_dir, data_=data_)
train,train_x, valid, valid_x = loader.load_data(type=type_feature, ValidRatio=0.2, n_window=n_window, n_slide=n_slide)

n_asset, n_times, W, n_feature = np.shape(train)
A_, T_, W_, C_ = np.shape(valid)

print('total samples : {times} assets : {assets} window : {window} features : {features}'
      .format(times=n_times, assets=n_asset, window=W, features=n_feature))
from loss_func import *
K = 20

# hidden, cell = set_hidden(n_asset * n_batch, n_hidden, n_layer, device)
# hidden_, cell_ = set_hidden(n_asset * T_, n_hidden, n_layer, device)
train, train_x = torch.from_numpy(train).float().to(device), torch.from_numpy(train_x).float().to(device)
valid, valid_x = torch.from_numpy(valid).float().to(device), torch.from_numpy(valid_x).float().to(device)
for i in [0, 27, 100,1000,10000]:
    torch.manual_seed(i)
    model = Model(n_window=n_window, n_feature=n_feature,
                  n_hidden=n_hidden, n_batch=n_batch, n_asset=n_asset,
                  n_layers=n_layer,device=device).to(device)
    # par = torch.load('ckpt.pt')
    # model.load_state_dict(par.state_dict())
    rb = ReplayMemory(0, n_times, n_batch)

    PG = Policy_gradient(n_batch, n_asset)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(n_episode):
        model.train()
        next_idx = rb.next_batch()
        state = train[:, next_idx:next_idx+n_batch] # A, T, W, C
        y_t = train_x[:, next_idx:next_idx+n_batch].transpose(1,0)  # T, A

        if model_type == 'LSTM':
            state = state.transpose(1,0).contiguous().view(n_batch * n_asset, n_window, n_feature)  # asset_t0 ~ asset_t1 , ... , asset_T
            mu, logvar = model(state, hidden, cell)
            mu = torch.softmax(mu.view(n_batch, n_asset), dim=1)
            logvar = logvar.view(n_batch, n_asset)

        elif model_type == 'CNN':
            state = state.permute(1, 3, 0, 2)  # T, C, A, W
            actions, logvar = model(state)   # T, A, 1
        sigma = torch.exp(logvar/2)
        sigma = torch.ones_like(mu)

        logprob, actions = PG.select_action(mu, sigma, epsilon=epsilon)

        reward = (actions*y_t).sum(1)
        reward = -logprob * torch.log(reward)
        reward = (reward - reward.mean().detach())/reward.std().detach()

        loss = reward.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode % 1000) == 0:
            model.eval()
            y_t = valid_x.transpose(1, 0)  # T, A
            state = valid   # A, T, W, C
            if model_type == 'LSTM':
                state = state.transpose(1, 0).contiguous().view(n_batch * n_asset, n_window,
                                                                n_feature)  # asset_t0 ~ asset_t1 , ... , asset_T
                mu, logvar = model(state, hidden, cell)
                mu = torch.softmax(mu.view(T_, A_),dim=1)
                logvar = logvar.view(T_, A_)
            elif model_type == 'CNN':
                state = state.permute(1, 3, 0, 2)  # T, C, A, W
                actions, logvar = model(state)  # T, A, 1

            sigma = torch.exp(logvar / 2)
            sigma = torch.ones_like(mu)
            logprob, actions = PG.select_action(mu, sigma, epsilon=epsilon)

            reward = (actions * y_t).sum(1)
            print('episode : {ep:5d} train reward : {r:.8f}'
                  .format(ep=episode, r=torch.cumprod(reward.sum(1),dim=0)[-1].item()))





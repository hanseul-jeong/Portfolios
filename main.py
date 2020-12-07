from data_loader import data_loader
from utils import set_hidden
from model import model
from Policy_gradient import Policy_gradient
import torch
import torch.optim as optim
import numpy as np
import os
import random

# set parameters
data_dir = 'Dataset'
data_ = 'KOSPI_daily.db'
data_path = os.path.join(data_dir, data_)
StartDate = 20100101
EndDate = 20200101
ValidDate = 20150101
n_window = 60
n_slide = 1
n_hidden = 32
n_layer = 1
epsilon = 0.9
update = 4
lr = 1e-4
n_batch = 128

n_episode = 10000
type_feature = 'OHLC'   # C or OHLC or OHLCV
gpu = 0

device = 'cuda:{gpu}'.format(gpu=gpu) if torch.cuda.is_available() else 'cpu'

loader = data_loader(StartDate=StartDate, EndDate=EndDate, data_dir=data_dir, data_=data_)
train,train_x, valid, valid_x = loader.load_data(type=type_feature, ValidDate=ValidDate, n_window=n_window, n_slide=n_slide)

n_times, n_asset, W, n_feature = np.shape(train)

print('total samples : {times} assets : {assets} window : {window} features : {features}'
      .format(times=n_times, assets=n_asset, window=W, features=n_feature))

model = model(n_window=n_window, n_feature=n_feature,
              n_hidden=n_hidden, n_batch=n_batch, n_asset=n_asset,
              n_layers=n_layer,device=device).to(device)

hidden, cell = set_hidden(n_asset * n_batch, n_hidden, n_layer, device)
train, train_x = torch.from_numpy(train).float().to(device), torch.from_numpy(train_x).float().to(device)
valid, valid_x = torch.from_numpy(valid).float().to(device), torch.from_numpy(valid_x).float().to(device)

PG = Policy_gradient(n_batch, n_asset)
optimizer = optim.Adam(model.parameters(), lr=lr)
global_rewards = []
for episode in range(n_episode):
    model.train()
    for t in range(0, n_times, n_batch):
        state = train[t:t+n_batch]
        state = state.transpose(1,0).contiguous().view(n_batch * n_asset, n_window, n_feature)
        mu, logvar = model(state, hidden, cell)
        sigma = torch.exp(logvar/2)
        logprob, actions = PG.select_action(mu, sigma, epsilon=epsilon)
        G = train_x * logprob * actions
        # update 40 samples of n_batches
        samples = random.sample(range(len(n_batch)), 4 * 10)
        sampling = [[True if i in samples[u:u+10] else False for i in range(n_batch)] for u in range(0, 40, 10)]

        for u in range(4):
            loss = G[sampling[u]].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    mu, logvar = model(train, hidden, cell)
    logprob, actions = PG.select_action(model, state, epsilon=epsilon)
    G_train = train_x * logprob * actions

    global_rewards.append(G_train.item())
    # mu, logvar = model(valid, hidden, cell)
    # logprob, actions = PG.select_action(model, state, epsilon=epsilon)
    # G_valid = logprob * actions
    print('episode : {ep} epsilon : {eps} train reward : {r}'
          .format(ep=episode, eps=epsilon,r=global_rewards[-1]))

    if episode % update == 0:
        epsilon = 0.01 if episode < 0.01 else epsilon * 0.9


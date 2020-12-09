from data_loader import data_loader
from utils import set_hidden
from model import model
from Policy_gradient import Policy_gradient
import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import get_logprob

# set parameters
data_dir = 'Dataset'
data_ = 'KOSPI_daily.db'
data_path = os.path.join(data_dir, data_)
StartDate = 20100101
EndDate = 20200101
ValidDate = 20150101
n_window = 30
n_slide = 1
n_hidden = 32
n_layer = 1
epsilon = 0.9
update = 4
lr = 1e-4
n_batch = 256

n_episode = 10000
type_feature = 'OHLC'   # C or OHLC or OHLCV
gpu = 1

device = 'cuda:{gpu}'.format(gpu=gpu) if torch.cuda.is_available() else 'cpu'

Assets = ['두산', '기아차','한화', '대림건설','롯데푸드','한진','금호산업','대한항공',
                'LG','신세계','농심','삼성전자','오뚜기','S-Oil','금호타이어','셀트리온']

loader = data_loader(StartDate=StartDate, EndDate=EndDate, data_dir=data_dir, data_=data_, selective=Assets)
train,train_x, valid, valid_x = loader.load_data(type=type_feature, ValidDate=ValidDate, n_window=n_window, n_slide=n_slide)

n_asset, n_times, W, n_feature = np.shape(train)

print('assets : {assets} total samples : {times} window : {window} features : {features}'
      .format(assets=n_asset, times=n_times, window=W, features=n_feature))
n_batch = n_times
model = model(n_window=n_window, n_feature=n_feature,
              n_hidden=n_hidden, n_batch=n_batch, n_asset=n_asset,
              n_layers=n_layer,device=device).to(device)

hidden, cell = set_hidden(n_asset * n_batch, n_hidden, n_layer, device)
train, train_x = torch.from_numpy(train).float().to(device), torch.from_numpy(train_x).float().to(device)
valid, valid_x = torch.from_numpy(valid).float().to(device), torch.from_numpy(valid_x).float().to(device)

window = plt.figure()
ax = window.add_subplot(1,1,1)
plt.xlabel('episode', fontsize=17)
plt.ylabel('Average rewards', fontsize=17)

PG = Policy_gradient(n_batch, n_asset, device=device)
PG_t = Policy_gradient(n_times, n_asset)
optimizer = optim.Adam(model.parameters(), lr=lr)
global_rewards = []
for episode in range(1, n_episode+1):
    model.train()
    rewards = []
    for t in range(0, (n_times//n_batch)*n_batch, n_batch):
        state = train[:, t:t+n_batch]
        state = state.contiguous().view(n_asset*n_batch, n_window, n_feature)
        x_ratio = train_x[:, t:t+n_batch]

        mu, logvar = model(state, hidden, cell)
        mu = mu.view(n_asset, n_batch)
        # sigma = torch.ones_like(mu).detach()
        sigma = torch.exp(logvar/2)
        sigma = sigma.view(n_asset, n_batch)


        ######################################## current var = 1
        logprobs, actions = PG.select_action(mu, sigma, epsilon=epsilon)

        # logprob = logprob.view(n_asset, n_batch)
        # actions = actions.view(n_asset, n_batch)
        # x = torch.normal(mu, torch.ones_like(mu))
        # actions = torch.where(x > 0, x, torch.zeros_like(x))
        # actions = actions / actions.sum(0).unsqueeze(0).detach()
        # actions = torch.where(actions !=0, actions / actions.sum(0).unsqueeze(0).detach(), torch.ones_like(actions)/n_asset)
        # actions = actions / actions.abs().sum(0).unsqueeze(0)
        reward = (x_ratio * actions)
        # logprob = get_logprob(x, mu, torch.ones_like(mu))

        # # update 40 samples of n_batches
        # samples = random.sample(range(n_batch), 4 * 10)
        # sampling = [[True if i in samples[u:u+10] else False for i in range(n_batch)] for u in range(0, 40, 10)]
        #
        # for u in range(4):
        #     loss = G[sampling[u]].sum()
        loss =  -((reward * logprobs).sum(0)).mean()
        # loss = (-torch.log(reward)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(torch.prod(reward.sum(0)).item())
    global_rewards.append(np.mean(rewards))
    # model.eval()
    # valid = valid.view(-1, n_window, n_feature)
    # mu, logvar = model(valid, hidden, cell)
    # sigma = torch.exp(logvar / 2)
    # logprob, actions = PG_t.select_action(mu, sigma, epsilon=epsilon)
    # G_train = -(train_x * logprob * actions)

    # mu, logvar = model(valid, hidden, cell)
    # logprob, actions = PG.select_action(model, state, epsilon=epsilon)
    # G_valid = logprob * actions
    print('episode : {ep} epsilon : {eps:.4f} train reward : {r:.4f}'
          .format(ep=episode, eps=epsilon,r=global_rewards[-1]))

    ax.plot(global_rewards, color='gray', linewidth=0.5)
    plt.pause(0.0000001)

    if episode % 2000 == 0:
        torch.save(model, os.path.join("checkpoint",'ck_v2_{0}.pt'.format(episode//2000)))
        plt.savefig(os.path.join('checkpoint','loss_v2_{0}'.format(episode//2000)))
    ax.lines.pop()

    if episode % update == 0:
        epsilon = 0.01 if episode < 0.01 else epsilon * 0.9


plt.close()


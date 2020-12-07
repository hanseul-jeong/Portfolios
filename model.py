import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self, n_window, n_feature, n_hidden, n_batch, n_asset, n_layers=1, device='cpu'):
        super(model, self).__init__()
        self.n_window = n_window
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        # self.n_batch = n_batch
        # self.n_asset = n_asset

        self.LSTM = nn.LSTM(n_feature, self.n_hidden, n_layers, batch_first=True)

        self.fc1 = nn.Linear(n_hidden, 64)
        self.mu = nn.Linear(64, 1)
        self.logvar = nn.Linear(64, 1)

        self.policy_history = torch.FloatTensor().to(device)

    def forward(self, x, hidden, cell):
        # B(n_asset * n_batch), T, D

        z, (h,c) = self.LSTM(x, (hidden, cell)) # B, T, H
        z = z[:,-1,:].view(-1, self.n_hidden)

        z = torch.tanh(self.fc1(z))
        mu, logvar = self.mu(z), self.logvar(z)

        return mu, logvar

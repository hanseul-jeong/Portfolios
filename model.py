import torch.nn as nn
import torch

class LSTM_whole(nn.Module):
    def __init__(self, n_window, n_feature, n_hidden, n_batch, n_asset, n_layers=1, device='cpu'):
        super(LSTM_whole, self).__init__()
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

class CNN_whole(nn.Module):
    def __init__(self, n_window, n_feature, n_hidden, n_batch, n_asset, n_layers=1, device='cpu'):
        super(CNN_whole, self).__init__()
        self.n_window = n_window
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.__n_batch = n_batch
        self.__n_asset = n_asset

        self.filter1 = nn.Conv2d(n_feature, self.n_hidden, [1,5])
        self.filter2 = nn.Conv2d(self.n_hidden, self.n_hidden, [1,5])
        self.filter3 = nn.Conv2d(self.n_hidden, self.n_hidden, [1,5])
        self.__hidden_dim = self.get_dim(width=n_window, layer=3, filter=5, padding=0, str=1)

        # fully-connected layer
        self.mu = nn.Conv2d(self.n_hidden, 1, [1, self.__hidden_dim])
        self.logvar = nn.Conv2d(self.n_hidden, 1, [1, self.__hidden_dim])

        self.policy_history = torch.FloatTensor().to(device)

    def forward(self, x):
        x = torch.relu(self.filter1(x))
        x = torch.relu(self.filter2(x))
        x = torch.relu(self.filter3(x))

        mu, logvar = self.mu(x).squeeze(), self.logvar(x).squeeze()
        mu = torch.softmax(mu, dim=1)
        return mu, logvar

    def get_dim(self, width, layer=3, filter=3, padding=0, str=1):
        width_ = width + (2*padding)
        for i in range(layer):
            width_ = int((width_ - filter)/str) + 1
        return width_

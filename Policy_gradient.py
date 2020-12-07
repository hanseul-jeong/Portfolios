from torch.distributions import Normal
from torch.autograd import Variable
import numpy as np
import sys
import torch
import random
n_RGB = 3

class Policy_gradient():
    def __init__(self, n_batch, n_asset, discount_factor=0.99, device='cuda'):
        self.input_shape = (1, 4)
        self.n_assets = n_asset
        self.n_batch = n_batch

        self.discount_factor = discount_factor
        self.device = device

    def select_action(self, mu, sigma, epsilon):
        logprob = torch.empty([self.n_assets, self.n_batch], dtype=torch.float32)
        actions = torch.empty([self.n_assets, self.n_batch], dtype=torch.float32)
        for a in range(self.n_assets):
            for b in range(self.n_batch):
                p = random.random()
                if p <= epsilon:
                    mu[a*self.n_batch + b] = 0
                    sigma[a*self.n_batch + b] = 1
                action_dist = Normal(mu[a*self.n_batch + b], sigma[a*self.n_batch + b])
                actions[a, b] = action_dist.sample()
                logprob[a, b] = action_dist.log_prob(actions[a, b])
        #
        # # normalize
        # policies = loglikelihoods

        return logprob, actions

    # Calculate loss and backward
    def get_loss(self, policy):
        reward_lists = []
        n_samples = len(policy.rewards)
        R = 0

        # Discounted reward
        for r in policy.rewards[::-1]:
            R = r + self.discount_factor * R
            reward_lists.insert(0, R)

        reward_list = torch.FloatTensor(reward_lists).to(self.device)

        # Normalization
        reward_list = torch.where(reward_list.std() !=0,
                                  (reward_list - reward_list.mean())/reward_list.std(),
                                  torch.zeros_like(reward_list))

        loss = -torch.sum(policy.policy_history * reward_list) / n_samples

        return loss

    # Backward loss and renewal previous policies and rewards
    def reset_policy(self, policy):
        policy.policy_history = Variable(torch.Tensor()).to(self.device)
        policy.rewards = []
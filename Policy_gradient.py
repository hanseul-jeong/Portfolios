from torch.distributions import Normal
from torch.autograd import Variable
from utils import get_logprob
import numpy as np
import sys
import torch
import random
n_RGB = 3

class Policy_gradient():
    def __init__(self, n_batch, n_asset, discount_factor=0.99, device='cuda'):
        self.input_shape = (1, 4)
        self.__n_asset = n_asset
        self.__n_batch = n_batch

        self.discount_factor = discount_factor
        self.device = device

    def select_action(self, mu, sigma, epsilon):
        # actions = torch.normal(mu, sigma)
        actions = mu
        logprobs = get_logprob(actions, mu, sigma)

        # No 공매도 & normalize
        actions = torch.relu(actions)
        actions = torch.where(actions.sum(1)[:,None].expand(-1,100)==0,
                              torch.ones_like(actions)/self.__n_asset,
                              actions / actions.sum(1)[:,None].detach())
        assert not torch.any(torch.isnan(actions))
        assert not torch.any(torch.isnan(logprobs))
        return logprobs, actions

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
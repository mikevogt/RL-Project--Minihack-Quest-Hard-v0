import torch.nn as nn
import numpy as np
import torch
from collections import deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import collections
import torch.nn.functional as F
from model import combined

class Policy_Gradient(object):

    def __init__(self, crop_state, whole_state, stats_state, env_action_space, gamma, lr, eps, no_frames, decay):
        self.policy = combined(crop_state, whole_state, stats_state, env_action_space.n, lr, eps, no_frames, decay)
        self.log_probs = []
        self.rewards = []
        self.gamma = gamma

    def select_action(self, crop, whole, stats):
        probs = self.policy.forward(crop.to(self.policy.device), whole.to(self.policy.device), stats.to(self.policy.device))
        probs = probs.squeeze()
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def train(self):
        self.policy.optimizer.zero_grad()

        R = np.zeros_like(self.rewards, dtype=np.float64)
        for t in range(len(self.rewards)):
            R_sum = 0
            discount = 1
            for k in range(t, len(self.rewards)):
                R_sum += self.rewards[k]*discount
                discount *= self.gamma
            R[t] = R_sum

        R = torch.tensor(R, dtype=torch.float).to(self.policy.device)
        R = (R - R.mean())/(R.std())#+1e-7) # + eps
        loss = 0

        for lp, r in zip(self.log_probs, R):
            loss += (-lp * r)

        loss.backward()
        self.policy.optimizer.step()

        self.rewards = []
        self.log_probs = []

        return loss

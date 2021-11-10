import torch.nn as nn
import numpy as np
import torch
from collections import deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import collections
import torch.nn.functional as F
from model import combined

class Policy_Gradient_Baseline(object):
    def __init__(self, crop_state, whole_state, stats_state, env_action_space, gamma, policy_lr, critic_lr, adam_eps, weight_decay):
        self.policy_lr = policy_lr
        self.critic_lr = critic_lr
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.policy = combined(crop_state, whole_state, stats_state, env_action_space.n, self.policy_lr, self.adam_eps, no_frames, self.weight_decay)
        self.critic = combined(crop_state, whole_state, stats_state, 1, self.critic_lr, self.adam_eps, no_frames, self.weight_decay)
        self.log_probs = []
        self.rewards = []
        self.states = []

    def select_action(self, crop, whole, stats):
        probs = self.policy.forward(crop.to(self.policy.device), whole.to(self.policy.device), stats.to(self.policy.device))
        probs = probs.squeeze()
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        self.states.append([crop,whole,stats])
        return action.item()

    def save(self, env):
        torch.save(self.policy.state_dict(), "Agent_dicts/PG/PG_state_dict - "+str(env))
        torch.save(self.policy.optimizer.state_dict(), "Agent_dicts/PG/PG_optimizer - "+str(env))

    def load(self, env):
        self.policy.load_state_dict(torch.load("Agent_dicts/PG/PG_state_dict - "+str(env)))
        self.policy.optimizer.load_state_dict(torch.load("Agent_dicts/PG/PG_optimizer - "+str(env)))

    def train(self):
        self.policy.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        R = np.zeros_like(self.rewards, dtype=np.float64)
        V = np.zeros_like(self.rewards, dtype=np.float64)
        V = torch.tensor(V, dtype=torch.float).to(self.policy.device)
        for t in range(len(self.rewards)):
            R_sum = 0
            discount = 1
            for k in range(t, len(self.rewards)):
                R_sum += self.rewards[k]*discount
                discount *= self.gamma
            R[t] = R_sum
            v = self.critic.forward_critic(self.states[t][0].to(self.policy.device), self.states[t][1].to(self.policy.device), self.states[t][2].to(self.policy.device)).squeeze()
            V[t] = v

        R = torch.tensor(R, dtype=torch.float).to(self.policy.device)
        if R.std() > 0:
            std = R.std()
        else:
            std = 1
        R = (R - R.mean())/(std)
        if V.std() > 0:
            std_v = V.std()
        else:
            std_v = 1
        V = (V - V.mean())/(std_v)
        loss = 0

        for lp, r, v in zip(self.log_probs, R, V):
            loss += -(lp * (r-v).detach())

        #loss = loss#/len(self.log_probs)
        loss_critic = (0.5*(R-V)**2).mean()

        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
        self.policy.optimizer.step()

        loss_critic.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic.optimizer.step()


        self.rewards = []
        self.log_probs = []
        self.states = []

        return loss, loss_critic

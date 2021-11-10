import torch.nn as nn
import numpy as np
import torch
from collections import deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import collections
import torch.nn.functional as F
from model import combined
from Replay_Buffer import ReplayBuffer

class PPO(object):
    def __init__(self, crop_state, whole_state, stats_state, env_action_space, gamma, actor_lr, critic_lr, adam_eps, weight_decay):
        self.actor = combined(crop_state, whole_state, stats_state, env_action_space.n, actor_lr, adam_eps, no_frames, weight_decay)
        self.critic = combined(crop_state, whole_state, stats_state, env_action_space.n, critic_lr, adam_eps, no_frames, weight_decay)
        self.gamma = gamma
        self.clip = 0.2
        self.max_gradient_norm = 0.5
        self.gae_lambda = 0.95
        self.replay_buffer = ReplayBuffer(5000)
        self.batch_size = 32
        self.epochs = 5
        self.rewards=[]

    def select_action(self, crop, whole, stats):
        probs = self.actor.forward(crop.to(self.actor.device), whole.to(self.actor.device), stats.to(self.actor.device))
        probs = probs.squeeze()
        prob = Categorical(probs)
        action = prob.sample()
        val = self.critic.forward_critic(crop.to(self.actor.device), whole.to(self.actor.device), stats.to(self.actor.device))
        return action.item(), probs[action.item()].item(), val.item()

    def save(self, env):
        torch.save(self.actor.state_dict(), "Agent_dicts/PPO/PPO_actor_state_dict"+str(env))
        torch.save(self.actor.optimizer.state_dict(), "Agent_dicts/PPO/PPO_actor_optimizer"+str(env))
        torch.save(self.critic.state_dict(), "Agent_dicts/PPO/PPO_critic_state_dict"+str(env))
        torch.save(self.critic.optimizer.state_dict(), "Agent_dicts/PPO/PPO_critic_optimizer"+str(env))

    def load(self, env):
        self.actor.load_state_dict(torch.load("Agent_dicts/PPO/PPO_actor_state_dict"+str(env)))
        self.actor.optimizer.load_state_dict(torch.load("Agent_dicts/PPO/PPO_actor_optimizer"+str(env)))
        self.critic.load_state_dict(torch.load("Agent_dicts/PPO/PPO_critic_state_dict"+str(env)))
        self.critic.optimizer.load_state_dict(torch.load("Agent_dicts/PPO/PPO_critic_optimizer"+str(env)))

    def train(self):
        for _ in range(self.epochs):
            state, actions, reward, log_probs, done, vals = self.replay_buffer.sample(self.batch_size)
            advantage = np.zeros(len(reward), dtype=np.float32)

            for i in range(len(reward)-1):
                discount = 1
                adv = 0
                for j in range(i, len(reward)-1):
                    adv += discount*(reward[j]+self.gamma*vals[j+1]*(1-int(done[j]))-vals[j])
                    discount *= self.gamma*self.gae_lambda
                advantage[i] = adv

            advantage = torch.tensor(advantage, dtype=torch.float).to(self.actor.device)
            vals = torch.tensor(np.array(vals), dtype=torch.float).to(self.actor.device)

            for j in range(self.batch_size):
                crop = stack_crop(state[j])
                whole = stack_whole(state[j])
                stats = stack_stats(state[j]).unsqueeze(0)
                action = torch.tensor(np.array(actions[j])).to(self.actor.device)
                log_prob = torch.tensor(np.array(log_probs[j]), dtype=torch.float).to(self.actor.device)

                V = self.critic.forward_critic(crop.to(self.actor.device), whole.to(self.actor.device), stats.to(self.actor.device))

                new_log_prob = self.actor.forward(crop.to(self.actor.device), whole.to(self.actor.device), stats.to(self.actor.device))

                new_log_prob = new_log_prob.squeeze()
                new_log_prob = Categorical(new_log_prob)
                new_probs = new_log_prob.log_prob(action)
                entropy = new_log_prob.entropy().mean()

                ratio = (new_probs.exp()/log_prob.exp())

                surr1 = ratio*advantage[j]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage[j]

                actor_loss = -torch.min(surr1, surr2).mean()

                returns = advantage[j] + vals[j]
                V = V.squeeze()
                critic_loss = (returns-V)**2
                critic_loss = critic_loss.mean()

                loss = actor_loss + 0.5*critic_loss - 0.0005*entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_gradient_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_gradient_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.replay_buffer.buffer.clear()
        return loss.item(), entropy

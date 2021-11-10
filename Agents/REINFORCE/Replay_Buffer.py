import collections
import numpy as np

Experience = collections.namedtuple('Experience',
                                   field_names=['state', 'probs', 'val', 'action', 'reward', 'done'])
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)

    def size(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        ind = np.random.choice(len(self.buffer), batch_size)
        state_arr = []
        probs_arr, vals_arr, actions_arr, rewards_arr, dones_arr = [], [], [], [], []
        for i in ind:
            states, probs, vals, actions, rewards, dones = self.buffer[i]
            state_arr.append(states)
            probs_arr.append(probs)
            vals_arr.append(vals)
            actions_arr.append(actions)
            rewards_arr.append(rewards)
            dones_arr.append(dones)

        return state_arr, probs_arr, vals_arr, actions_arr, rewards_arr, dones_arr

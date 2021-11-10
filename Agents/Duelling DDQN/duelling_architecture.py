
from gym import spaces
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Duelling_DDQN_Network(nn.Module):
    """
    An implementation of the duelling network architecture, as described in "".
    This architecture uses a single convolution network, followed by 2 fully connected layers,
    after which this output gets split into the value- and advantage function streams.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        super(Duelling_DDQN_Network, self).__init__()
        """
        Initialise the Duelling Architure
        :param observation_space: the state space of the environment, needs to be given in terms of channel x width x height
        :param action_space: the action space of the environment
        """
        self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 6, stride=3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 2, stride=1)

        fc_input_dims = self.calculate_mlp_input_dims(observation_space.shape)

        self.fc1 = nn.Linear(fc_input_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, action_space.n)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def calculate_mlp_input_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def predict(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        flat2 = F.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A

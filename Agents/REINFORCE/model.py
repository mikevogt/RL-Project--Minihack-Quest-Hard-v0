import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class combined(nn.Module):
    def __init__(self, obs_space_crop, obs_space_whole, obs_space_stats, action_space, lr, eps, no_frames, decay):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.action_space = action_space
        self.no_frames = no_frames
        self.decay = decay

        # CNN for cropped images

        obs_space_crop = torch.squeeze(obs_space_crop)
        obs_space_crop = obs_space_crop.unsqueeze(0)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.no_frames, 16, 3, stride=2, padding=1), #channels, outgoing_filters, kernerl_size, stride/no_of_frames
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.ReLU(),
        )

        self.input_dim_crop = self.conv_1(torch.zeros(self.no_frames, *obs_space_crop.squeeze().shape).unsqueeze(0)).view(1, -1).size(1)

        # CNN for whole images

        obs_space_whole = torch.squeeze(obs_space_whole)
        obs_space_whole = obs_space_whole.unsqueeze(0)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(self.no_frames, 16, 5, stride=2, padding=1), #channels, outgoing_filters, kernerl_size, stride/no_of_frames
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(),
        )

        self.input_dim_whole = self.conv_2(torch.zeros(self.no_frames, *obs_space_whole.squeeze().shape).unsqueeze(0)).view(1, -1).size(1)

        # MLP_1

        self.input_dim_mlp = obs_space_stats.shape[0]
        self.output_dim_mlp = 16
        self.fc0 = nn.Sequential(
            nn.Linear(self.input_dim_mlp, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim_mlp)
        )

        # Fully Connected 2nd Layer

        self.fc1_output = 128
        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dim_mlp+self.input_dim_whole+self.input_dim_crop, self.fc1_output),#self.output_dim_mlp+self.input_dim_whole+self.input_dim_crop
            nn.ReLU(),
        )


        # LSTM

        self.lstm_size = self.fc1_output
        self.num_layers = 1
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            batch_first=True,
        )


        # FC3

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1_output, self.action_space),
            nn.Softmax(dim=-1),
        )


        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps= self.eps, weight_decay=self.decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, crop, whole, stats):
        conv_1 = self.conv_1(crop).view(-1)
        conv_2 = self.conv_2(whole).view(-1)
        mlp_1 = self.fc0(stats).view(-1)
        x = torch.cat([conv_1, conv_2, mlp_1])
        x = self.fc1(x)
        h0 = torch.zeros(1,self.fc1_output).unsqueeze(1).requires_grad_().to(self.device)
        c0 = torch.zeros(1,self.fc1_output).unsqueeze(1).requires_grad_().to(self.device)
        x, (hn, cn) = self.lstm(x.unsqueeze(0).unsqueeze(0), (h0, c0))
        x = self.fc2(x)
        return x

    def forward_critic(self, crop, whole, stats):
        conv_1 = self.conv_1(crop).view(-1)
        conv_2 = self.conv_2(whole).view(-1)
        mlp_1 = self.fc0(stats).view(-1)
        x = torch.cat([conv_1, conv_2, mlp_1])
        x = self.fc1(x)
        h0 = torch.zeros(1,self.fc1_output).unsqueeze(1).requires_grad_().to(self.device)
        c0 = torch.zeros(1,self.fc1_output).unsqueeze(1).requires_grad_().to(self.device)
        x, (hn, cn) = self.lstm(x.unsqueeze(0).unsqueeze(0), (h0, c0))
        x = self.fc2_critic(x)
        return x

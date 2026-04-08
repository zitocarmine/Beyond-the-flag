import torch
import torch.nn as nn
import numpy as np

class MarioA2C(nn.Module):
    def __init__(self, n_actions, input_shape=(4, 84, 84)):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            self.feature_size = out.view(1, -1).size(1)
        
        self.lstm = nn.LSTMCell(self.feature_size, 512)
        
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, n_actions)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, hx, cx):
        x = self.features(x / 255.0).view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        return self.actor(hx), self.critic(hx), hx, cx
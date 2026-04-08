import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        self.inverse_net = nn.Sequential(
            nn.Linear(self.feature_size * 2, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(self.feature_size + n_actions, 512), nn.ReLU(),
            nn.Linear(512, self.feature_size)
        )
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

    def get_icm_loss(self, state, next_state, action, n_actions):
        phi_t = self.features(state / 255.0).view(state.size(0), -1)
        phi_next = self.features(next_state / 255.0).view(next_state.size(0), -1)
        
        pred_act = self.inverse_net(torch.cat([phi_t, phi_next], dim=1))
        inv_loss = F.cross_entropy(pred_act, action)
        
        act_oh = F.one_hot(action, n_actions).float()
        pred_phi_next = self.forward_net(torch.cat([phi_t, act_oh], dim=1))
        fwd_loss = 0.5 * F.mse_loss(pred_phi_next, phi_next.detach(), reduction='none').mean(1)
        
        return inv_loss, fwd_loss
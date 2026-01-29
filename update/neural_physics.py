import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class StatePredictor(nn.Module):
    """
    O Forward Model (P(S_t+1 | S_t, A_t)).
    Prevê o futuro estado dado o estado atual e uma ação.
    """

    def __init__(self, state_dim=384, action_dim=384):
        super(StatePredictor, self).__init__()


        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return delta  # Retorna o vetor tangente (direção da mudança)


class ActionGenerator(nn.Module):
    """
    O Inverse Model (P(A_t | S_t, S_t+1)).
    Dado onde estou e onde quero ir, qual a ação necessária?
    """

    def __init__(self, state_dim=384, action_dim=384):
        super(ActionGenerator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, current_state, target_state):
        x = torch.cat([current_state, target_state], dim=-1)
        action = self.net(x)
        return action
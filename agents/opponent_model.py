from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class OpponentModel(nn.Module):

    def __init__(self, obs_dim: int, u_dim: int, n_opponent_actions: int, hidden: int=64):
        super().__init__()
        in_dim = obs_dim + u_dim + n_opponent_actions
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_opponent_actions))
        self.n_opponent_actions = n_opponent_actions

    def forward(self, state, u_onehot, prev_action_onehot):
        x = torch.cat([state, u_onehot, prev_action_onehot], dim=-1)
        logits = self.net(x)
        return logits

    def predict_action(self, state, u_onehot, prev_action_onehot, sample: bool=True):
        with torch.no_grad():
            logits = self.forward(state, u_onehot, prev_action_onehot)
            probs = F.softmax(logits, dim=-1)
            if sample:
                return torch.multinomial(probs, 1).squeeze(-1)
            return probs.argmax(dim=-1)

    def loss(self, state, u_onehot, prev_action_onehot, observed_action):
        logits = self.forward(state, u_onehot, prev_action_onehot)
        return F.cross_entropy(logits, observed_action)

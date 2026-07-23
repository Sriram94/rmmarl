from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(in_dim, out_dim, hidden=1024, n_hidden=6):
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)

class CategoricalActor(nn.Module):

    def __init__(self, obs_dim, u_dim, n_actions, hidden=1024):
        super().__init__()
        self.net = mlp(obs_dim + u_dim, n_actions, hidden=hidden, n_hidden=6)

    def forward(self, obs, u_oh):
        return self.net(torch.cat([obs, u_oh], dim=-1))

    def dist(self, obs, u_oh):
        return torch.distributions.Categorical(logits=self.forward(obs, u_oh))

class CentralizedCritic(nn.Module):

    def __init__(self, n_agents, obs_dim, u_dim, n_actions, hidden=1024):
        super().__init__()
        in_dim = n_agents * (obs_dim + u_dim + n_actions)
        self.net = mlp(in_dim, 1, hidden=hidden, n_hidden=6)

    def forward(self, joint_obs_u, joint_actions_onehot):
        x = torch.cat([joint_obs_u, joint_actions_onehot], dim=-1)
        return self.net(x).squeeze(-1)

class CentralizedValue(nn.Module):

    def __init__(self, n_agents, obs_dim, u_dim, hidden=1024):
        super().__init__()
        in_dim = n_agents * (obs_dim + u_dim)
        self.net = mlp(in_dim, 1, hidden=hidden, n_hidden=6)

    def forward(self, joint_obs_u):
        return self.net(joint_obs_u).squeeze(-1)

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    return F.gumbel_softmax(logits, tau=temperature, hard=hard)

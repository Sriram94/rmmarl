from __future__ import annotations
import torch
import torch.nn as nn

def _mlp(in_dim, out_dim, hidden, n_hidden=3, out_activation=None):
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)

class DDPGActor(nn.Module):

    def __init__(self, obs_dim, u_dim, action_dim, hidden=1024):
        super().__init__()
        self.net = _mlp(obs_dim + u_dim, action_dim, hidden, n_hidden=6, out_activation=nn.Tanh())

    def forward(self, obs, u_oh):
        return self.net(torch.cat([obs, u_oh], dim=-1))

class DDPGCritic(nn.Module):

    def __init__(self, obs_dim, u_dim, action_dim, hidden=1024, opp_action_dim=0):
        super().__init__()
        self.net = _mlp(obs_dim + u_dim + action_dim + opp_action_dim, 1, hidden, n_hidden=6)

    def forward(self, obs, u_oh, action, opp_action=None):
        parts = [obs, u_oh, action]
        if opp_action is not None:
            parts.append(opp_action)
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)

class PerRMStateActor(nn.Module):

    def __init__(self, obs_dim, n_rm_states, action_dim, hidden=512):
        super().__init__()
        self.heads = nn.ModuleList([_mlp(obs_dim, action_dim, hidden, n_hidden=3, out_activation=nn.Tanh()) for _ in range(n_rm_states)])

    def forward(self, obs, u_index: int):
        return self.heads[u_index](obs)

    def forward_batched(self, obs, u_indices: torch.Tensor):
        out = torch.zeros(obs.shape[0], self.heads[0][-2].out_features, device=obs.device)
        for u in range(len(self.heads)):
            mask = u_indices == u
            if mask.any():
                out[mask] = self.heads[u](obs[mask])
        return out

class PerRMStateCritic(nn.Module):

    def __init__(self, obs_dim, n_rm_states, action_dim, hidden=512, opp_action_dim=0):
        super().__init__()
        self.heads = nn.ModuleList([_mlp(obs_dim + action_dim + opp_action_dim, 1, hidden, n_hidden=3) for _ in range(n_rm_states)])

    def forward(self, obs, u_index: int, action, opp_action=None):
        parts = [obs, action] if opp_action is None else [obs, action, opp_action]
        return self.heads[u_index](torch.cat(parts, dim=-1)).squeeze(-1)

    def forward_batched(self, obs, u_indices: torch.Tensor, action, opp_action=None):
        parts = [obs, action] if opp_action is None else [obs, action, opp_action]
        x = torch.cat(parts, dim=-1)
        out = torch.zeros(obs.shape[0], device=obs.device)
        for u in range(len(self.heads)):
            mask = u_indices == u
            if mask.any():
                out[mask] = self.heads[u](x[mask]).squeeze(-1)
        return out

class ContinuousOpponentModel(nn.Module):

    def __init__(self, obs_dim, u_dim, opponent_action_dim, hidden=64):
        super().__init__()
        in_dim = obs_dim + u_dim + opponent_action_dim
        self.net = _mlp(in_dim, opponent_action_dim, hidden, n_hidden=4, out_activation=nn.Tanh())

    def forward(self, state, u_onehot, prev_action):
        return self.net(torch.cat([state, u_onehot, prev_action], dim=-1))

    def predict_action(self, state, u_onehot, prev_action):
        with torch.no_grad():
            return self.forward(state, u_onehot, prev_action)

    def loss(self, state, u_onehot, prev_action, observed_action):
        pred = self.forward(state, u_onehot, prev_action)
        return torch.nn.functional.mse_loss(pred, observed_action)

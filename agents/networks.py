from __future__ import annotations
import torch
import torch.nn as nn

class CrossProductQNet(nn.Module):

    def __init__(self, obs_dim: int, u_dim: int, n_actions: int, hidden: int=1024, opp_action_dim: int=0):
        super().__init__()
        in_dim = obs_dim + u_dim + opp_action_dim
        self.opp_action_dim = opp_action_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, state, u_onehot, opp_action_onehot=None):
        parts = [state, u_onehot]
        if opp_action_onehot is not None:
            parts.append(opp_action_onehot)
        x = torch.cat(parts, dim=-1)
        return self.net(x)

class ShallowCrossProductQNet(nn.Module):

    def __init__(self, obs_dim: int, u_dim: int, n_options: int, hidden: int=256):
        super().__init__()
        in_dim = obs_dim + u_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_options))

    def forward(self, state, u_onehot):
        x = torch.cat([state, u_onehot], dim=-1)
        return self.net(x)

class PerRMStateQNet(nn.Module):

    def __init__(self, obs_dim: int, n_rm_states: int, n_actions: int, hidden: int=512, opp_action_dim: int=0):
        super().__init__()
        self.n_rm_states = n_rm_states
        self.opp_action_dim = opp_action_dim
        in_dim = obs_dim + opp_action_dim
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)) for _ in range(n_rm_states)])

    def forward(self, state, u_index: int, opp_action_onehot=None):
        x = state if opp_action_onehot is None else torch.cat([state, opp_action_onehot], dim=-1)
        return self.heads[u_index](x)

    def forward_batched(self, state, u_indices: torch.Tensor, opp_action_onehot=None):
        x = state if opp_action_onehot is None else torch.cat([state, opp_action_onehot], dim=-1)
        out = torch.zeros(state.shape[0], self.heads[0][-1].out_features, device=state.device)
        for u in range(self.n_rm_states):
            mask = u_indices == u
            if mask.any():
                out[mask] = self.heads[u](x[mask])
        return out

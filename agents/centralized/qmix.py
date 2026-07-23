from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import CentralizedController
from .buffers import JointReplayBuffer, JointExperience
from ..networks import CrossProductQNet
from ..common import one_hot_batch, epsilon_greedy

class MixingNetwork(nn.Module):

    def __init__(self, n_agents: int, state_dim: int, hidden: int=64, mixing_hidden: int=32):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden = mixing_hidden
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_agents * mixing_hidden))
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, mixing_hidden))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, mixing_hidden), nn.ReLU(), nn.Linear(mixing_hidden, 1))

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.mixing_hidden)
        b1 = self.hyper_b1(state).view(B, 1, self.mixing_hidden)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.mixing_hidden, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(B)

class QMIXTeam(CentralizedController):
    variant = 'QMIX'

    def __init__(self, agent_ids: List[str], obs_dim: int, n_rm_states: int, n_actions: int, device: str='cpu', lr: float=0.01, gamma: float=0.9, buffer_size: int=200000, target_update_every: int=100, seed: int=0):
        super().__init__(agent_ids, obs_dim, n_rm_states, n_actions)
        self.device, self.gamma = (device, gamma)
        self.target_update_every = target_update_every
        self.rng = np.random.default_rng(seed)
        self._train_steps = 0
        self.buffer = JointReplayBuffer(capacity=buffer_size, seed=seed)
        self.q_nets = {aid: CrossProductQNet(obs_dim, n_rm_states, n_actions).to(device) for aid in agent_ids}
        self.q_targets = {aid: CrossProductQNet(obs_dim, n_rm_states, n_actions).to(device) for aid in agent_ids}
        for aid in agent_ids:
            self.q_targets[aid].load_state_dict(self.q_nets[aid].state_dict())
        state_dim = self.n_agents * (obs_dim + n_rm_states)
        self.mixer = MixingNetwork(self.n_agents, state_dim).to(device)
        self.mixer_target = MixingNetwork(self.n_agents, state_dim).to(device)
        self.mixer_target.load_state_dict(self.mixer.state_dict())
        params = list(self.mixer.parameters())
        for aid in agent_ids:
            params += list(self.q_nets[aid].parameters())
        self.optim = torch.optim.Adam(params, lr=lr)

    def act(self, obs, u_idx, epsilon) -> Dict[str, int]:
        actions = {}
        for aid in self.agent_ids:
            o = torch.tensor(obs[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
            u_oh = one_hot_batch(torch.tensor([u_idx[aid]]), self.n_rm_states).to(self.device)
            with torch.no_grad():
                q = self.q_nets[aid](o, u_oh).squeeze(0)
            actions[aid] = epsilon_greedy(q, epsilon, self.rng)
        return actions

    def store(self, obs, u_idx, actions, rewards, next_obs, next_u_idx, dones) -> None:
        self.buffer.push(JointExperience(obs, u_idx, actions, rewards, next_obs, next_u_idx, dones))

    def _global_state(self, obs_key: str, u_key: str, batch: List[JointExperience]) -> torch.Tensor:
        parts = []
        for aid in self.agent_ids:
            obs_arr = np.stack([getattr(e, obs_key)[aid] for e in batch])
            u_arr = [getattr(e, u_key)[aid] for e in batch]
            o = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
            u_oh = one_hot_batch(torch.tensor(u_arr, dtype=torch.long), self.n_rm_states).to(self.device)
            parts.append(torch.cat([o, u_oh], dim=-1))
        return torch.cat(parts, dim=-1)

    def train(self, batch_size: int=64) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        B = len(batch)
        state = self._global_state('obs', 'u_idx', batch)
        next_state = self._global_state('next_obs', 'next_u_idx', batch)
        agent_qs, agent_qs_next = ([], [])
        team_reward = torch.zeros(B, device=self.device)
        team_done = torch.zeros(B, device=self.device)
        for aid in self.agent_ids:
            o = torch.tensor(np.stack([e.obs[aid] for e in batch]), dtype=torch.float32, device=self.device)
            u_oh = one_hot_batch(torch.tensor([e.u_idx[aid] for e in batch]), self.n_rm_states).to(self.device)
            a = torch.tensor([e.actions[aid] for e in batch], dtype=torch.long, device=self.device)
            o_next = torch.tensor(np.stack([e.next_obs[aid] for e in batch]), dtype=torch.float32, device=self.device)
            u_next_oh = one_hot_batch(torch.tensor([e.next_u_idx[aid] for e in batch]), self.n_rm_states).to(self.device)
            q_all = self.q_nets[aid](o, u_oh)
            agent_qs.append(q_all.gather(1, a.unsqueeze(1)).squeeze(1))
            with torch.no_grad():
                q_next_all = self.q_targets[aid](o_next, u_next_oh)
                agent_qs_next.append(q_next_all.max(dim=1).values)
            team_reward += torch.tensor([e.rewards[aid] for e in batch], dtype=torch.float32, device=self.device)
            team_done += torch.tensor([float(e.dones[aid]) for e in batch], dtype=torch.float32, device=self.device)
        team_reward /= self.n_agents
        team_done = (team_done > 0).float()
        agent_qs = torch.stack(agent_qs, dim=1)
        agent_qs_next = torch.stack(agent_qs_next, dim=1)
        q_tot = self.mixer(agent_qs, state)
        with torch.no_grad():
            q_tot_next = self.mixer_target(agent_qs_next, next_state)
            y = team_reward + self.gamma * (1 - team_done) * q_tot_next
        loss = F.mse_loss(q_tot, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            for aid in self.agent_ids:
                self.q_targets[aid].load_state_dict(self.q_nets[aid].state_dict())
            self.mixer_target.load_state_dict(self.mixer.state_dict())
        return loss.item()

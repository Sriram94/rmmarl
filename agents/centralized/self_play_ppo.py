from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from .mappo import MAPPOTeam, _Rollout
from .networks import CategoricalActor
from ..common import one_hot_batch

class _DecentralizedValue(torch.nn.Module):

    def __init__(self, obs_dim, u_dim, hidden=1024):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(obs_dim + u_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, obs_u):
        return self.net(obs_u).squeeze(-1)

class SelfPlayPPOTeam(MAPPOTeam):
    variant = 'self-play PPO'
    on_policy = True

    def __init__(self, agent_ids: List[str], obs_dim: int, n_rm_states: int, n_actions: int, device: str='cpu', lr: float=0.0003, gamma: float=0.9, gae_lambda: float=0.95, clip_eps: float=0.2, rollout_len: int=128, epochs: int=4, minibatch_size: int=64, seed: int=0):
        from .base import CentralizedController
        CentralizedController.__init__(self, agent_ids, obs_dim, n_rm_states, n_actions)
        self.device, self.gamma, self.lam = (device, gamma, gae_lambda)
        self.clip_eps, self.rollout_len, self.epochs, self.minibatch_size = (clip_eps, rollout_len, epochs, minibatch_size)
        self.rng = np.random.default_rng(seed)
        self.actors = {aid: CategoricalActor(obs_dim, n_rm_states, n_actions).to(device) for aid in agent_ids}
        self.critics = {aid: _DecentralizedValue(obs_dim, n_rm_states).to(device) for aid in agent_ids}
        self.actor_optims = {aid: torch.optim.Adam(a.parameters(), lr=lr) for aid, a in self.actors.items()}
        self.critic_optims = {aid: torch.optim.Adam(c.parameters(), lr=lr) for aid, c in self.critics.items()}
        self.rollout = _Rollout(agent_ids)

    def act(self, obs, u_idx, epsilon) -> Dict[str, int]:
        actions, logps, values = ({}, {}, {})
        for aid in self.agent_ids:
            o = torch.tensor(obs[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
            u_oh = one_hot_batch(torch.tensor([u_idx[aid]]), self.n_rm_states).to(self.device)
            with torch.no_grad():
                dist = self.actors[aid].dist(o, u_oh)
                a = dist.sample()
                logp = dist.log_prob(a)
                v = self.critics[aid](torch.cat([o, u_oh], dim=-1))
            actions[aid] = int(a.item())
            logps[aid] = logp.item()
            values[aid] = v.item()
        self._last_actions_logp_value = (actions, logps, values, obs, u_idx)
        return actions

    def _ppo_update(self) -> float:
        T = len(self.rollout)
        total_loss = 0.0
        for aid in self.agent_ids:
            rewards = np.array(self.rollout.rewards[aid], dtype=np.float32)
            values = np.array(self.rollout.values[aid] + [self.rollout.values[aid][-1]], dtype=np.float32)
            dones = np.array(self.rollout.dones[aid], dtype=np.float32)
            adv = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
                adv[t] = gae
            returns = adv + values[:T]
            adv = (adv - adv.mean()) / (adv.std() + 1e-08)
            obs = torch.tensor(np.stack(self.rollout.obs[aid]), dtype=torch.float32, device=self.device)
            u_idx_t = torch.tensor(self.rollout.u_idx[aid], dtype=torch.long, device=self.device)
            u_oh = one_hot_batch(u_idx_t, self.n_rm_states).to(self.device)
            obs_u = torch.cat([obs, u_oh], dim=-1)
            actions_t = torch.tensor(self.rollout.actions[aid], dtype=torch.long, device=self.device)
            old_logp = torch.tensor(self.rollout.logp[aid], dtype=torch.float32, device=self.device)
            adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            idx = np.arange(T)
            for _ in range(self.epochs):
                self.rng.shuffle(idx)
                for start in range(0, T, self.minibatch_size):
                    mb = idx[start:start + self.minibatch_size]
                    mb_t = torch.tensor(mb, dtype=torch.long, device=self.device)
                    dist = self.actors[aid].dist(obs[mb_t], u_oh[mb_t])
                    new_logp = dist.log_prob(actions_t[mb_t])
                    ratio = torch.exp(new_logp - old_logp[mb_t])
                    surr1 = ratio * adv_t[mb_t]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb_t]
                    actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()
                    v_pred = self.critics[aid](obs_u[mb_t])
                    critic_loss = F.mse_loss(v_pred, returns_t[mb_t])
                    self.actor_optims[aid].zero_grad()
                    actor_loss.backward()
                    self.actor_optims[aid].step()
                    self.critic_optims[aid].zero_grad()
                    critic_loss.backward()
                    self.critic_optims[aid].step()
                    total_loss += actor_loss.item() + critic_loss.item()
        self.rollout.reset()
        return total_loss / (self.n_agents * self.epochs)

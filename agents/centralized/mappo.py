from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from .base import CentralizedController
from .networks import CategoricalActor, CentralizedValue
from ..common import one_hot_batch

class _Rollout:

    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.reset()

    def reset(self):
        self.obs = {aid: [] for aid in self.agent_ids}
        self.u_idx = {aid: [] for aid in self.agent_ids}
        self.actions = {aid: [] for aid in self.agent_ids}
        self.logp = {aid: [] for aid in self.agent_ids}
        self.rewards = {aid: [] for aid in self.agent_ids}
        self.dones = {aid: [] for aid in self.agent_ids}
        self.values = {aid: [] for aid in self.agent_ids}

    def __len__(self):
        return len(self.obs[self.agent_ids[0]])

class MAPPOTeam(CentralizedController):
    variant = 'MAPPO'
    on_policy = True

    def __init__(self, agent_ids: List[str], obs_dim: int, n_rm_states: int, n_actions: int, device: str='cpu', lr: float=0.0003, gamma: float=0.9, gae_lambda: float=0.95, clip_eps: float=0.2, rollout_len: int=128, epochs: int=4, minibatch_size: int=64, seed: int=0):
        super().__init__(agent_ids, obs_dim, n_rm_states, n_actions)
        self.device, self.gamma, self.lam = (device, gamma, gae_lambda)
        self.clip_eps, self.rollout_len, self.epochs, self.minibatch_size = (clip_eps, rollout_len, epochs, minibatch_size)
        self.rng = np.random.default_rng(seed)
        self.actors = {aid: CategoricalActor(obs_dim, n_rm_states, n_actions).to(device) for aid in agent_ids}
        self.critics = {aid: CentralizedValue(self.n_agents, obs_dim, n_rm_states).to(device) for aid in agent_ids}
        self.actor_optims = {aid: torch.optim.Adam(a.parameters(), lr=lr) for aid, a in self.actors.items()}
        self.critic_optims = {aid: torch.optim.Adam(c.parameters(), lr=lr) for aid, c in self.critics.items()}
        self.rollout = _Rollout(agent_ids)

    def _joint_obs_u(self, obs, u_idx):
        parts = []
        for aid in self.agent_ids:
            u_oh = one_hot_batch(torch.tensor([u_idx[aid]]), self.n_rm_states).to(self.device).squeeze(0)
            o = torch.tensor(obs[aid], dtype=torch.float32, device=self.device)
            parts.append(torch.cat([o, u_oh]))
        return torch.cat(parts)

    def act(self, obs, u_idx, epsilon) -> Dict[str, int]:
        actions, logps, values = ({}, {}, {})
        joint = self._joint_obs_u(obs, u_idx).unsqueeze(0)
        for aid in self.agent_ids:
            o = torch.tensor(obs[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
            u_oh = one_hot_batch(torch.tensor([u_idx[aid]]), self.n_rm_states).to(self.device)
            with torch.no_grad():
                dist = self.actors[aid].dist(o, u_oh)
                a = dist.sample()
                logp = dist.log_prob(a)
                v = self.critics[aid](joint)
            actions[aid] = int(a.item())
            logps[aid] = logp.item()
            values[aid] = v.item()
        self._last_actions_logp_value = (actions, logps, values, obs, u_idx)
        return actions

    def store(self, obs, u_idx, actions, rewards, next_obs, next_u_idx, dones) -> None:
        _, logps, values, cached_obs, cached_u = self._last_actions_logp_value
        for aid in self.agent_ids:
            self.rollout.obs[aid].append(cached_obs[aid])
            self.rollout.u_idx[aid].append(cached_u[aid])
            self.rollout.actions[aid].append(actions[aid])
            self.rollout.logp[aid].append(logps[aid])
            self.rollout.rewards[aid].append(rewards[aid])
            self.rollout.dones[aid].append(float(dones[aid]))
            self.rollout.values[aid].append(values[aid])

    def train(self, batch_size: int=64) -> Optional[float]:
        if len(self.rollout) < self.rollout_len:
            return None
        return self._ppo_update()

    def _ppo_update(self) -> float:
        T = len(self.rollout)
        total_loss = 0.0
        joint_obs_u = self._build_joint_tensor()
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
                    v_pred = self.critics[aid](joint_obs_u[mb_t])
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

    def _build_joint_tensor(self):
        parts = []
        for aid in self.agent_ids:
            obs = torch.tensor(np.stack(self.rollout.obs[aid]), dtype=torch.float32, device=self.device)
            u_oh = one_hot_batch(torch.tensor(self.rollout.u_idx[aid], dtype=torch.long), self.n_rm_states).to(self.device)
            parts.append(torch.cat([obs, u_oh], dim=-1))
        return torch.cat(parts, dim=-1)

from __future__ import annotations
from typing import Dict, List, Optional
import copy
import numpy as np
import torch
import torch.nn.functional as F
from .base import CentralizedController
from .buffers import JointReplayBuffer, JointExperience
from .networks import CategoricalActor, CentralizedCritic, gumbel_softmax_sample
from ..common import one_hot, one_hot_batch, epsilon_greedy

class MADDPGTeam(CentralizedController):
    variant = 'MADDPG'

    def __init__(self, agent_ids: List[str], obs_dim: int, n_rm_states: int, n_actions: int, device: str='cpu', lr: float=0.01, gamma: float=0.9, buffer_size: int=200000, target_update_tau: float=0.01, seed: int=0):
        super().__init__(agent_ids, obs_dim, n_rm_states, n_actions)
        self.device, self.gamma, self.tau = (device, gamma, target_update_tau)
        self.rng = np.random.default_rng(seed)
        self.buffer = JointReplayBuffer(capacity=buffer_size, seed=seed)
        self.actors = {aid: CategoricalActor(obs_dim, n_rm_states, n_actions).to(device) for aid in agent_ids}
        self.actor_targets = {aid: copy.deepcopy(a).to(device) for aid, a in self.actors.items()}
        self.critics = {aid: CentralizedCritic(self.n_agents, obs_dim, n_rm_states, n_actions).to(device) for aid in agent_ids}
        self.critic_targets = {aid: copy.deepcopy(c).to(device) for aid, c in self.critics.items()}
        self.actor_optims = {aid: torch.optim.Adam(a.parameters(), lr=lr) for aid, a in self.actors.items()}
        self.critic_optims = {aid: torch.optim.Adam(c.parameters(), lr=lr) for aid, c in self.critics.items()}

    def act(self, obs, u_idx, epsilon) -> Dict[str, int]:
        actions = {}
        for aid in self.agent_ids:
            o = torch.tensor(obs[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
            u_oh = one_hot_batch(torch.tensor([u_idx[aid]]), self.n_rm_states).to(self.device)
            with torch.no_grad():
                logits = self.actors[aid](o, u_oh).squeeze(0)
            actions[aid] = epsilon_greedy(logits, epsilon, self.rng)
        return actions

    def store(self, obs, u_idx, actions, rewards, next_obs, next_u_idx, dones) -> None:
        self.buffer.push(JointExperience(obs, u_idx, actions, rewards, next_obs, next_u_idx, dones))

    def _stack_joint(self, batch: List[JointExperience], key: str, is_next: bool=False):
        per_agent = {}
        for aid in self.agent_ids:
            if key == 'obs':
                arr = [e.next_obs[aid] if is_next else e.obs[aid] for e in batch]
                per_agent[aid] = torch.tensor(np.stack(arr), dtype=torch.float32, device=self.device)
            elif key == 'u':
                arr = [e.next_u_idx[aid] if is_next else e.u_idx[aid] for e in batch]
                per_agent[aid] = one_hot_batch(torch.tensor(arr), self.n_rm_states).to(self.device)
        return per_agent

    def train(self, batch_size: int=64) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        obs = self._stack_joint(batch, 'obs', is_next=False)
        u_oh = self._stack_joint(batch, 'u', is_next=False)
        next_obs = self._stack_joint(batch, 'obs', is_next=True)
        next_u_oh = self._stack_joint(batch, 'u', is_next=True)
        joint_obs_u = torch.cat([torch.cat([obs[aid], u_oh[aid]], dim=-1) for aid in self.agent_ids], dim=-1)
        joint_next_obs_u = torch.cat([torch.cat([next_obs[aid], next_u_oh[aid]], dim=-1) for aid in self.agent_ids], dim=-1)
        actions_oh = {aid: one_hot_batch(torch.tensor([e.actions[aid] for e in batch]), self.n_actions).to(self.device) for aid in self.agent_ids}
        with torch.no_grad():
            target_actions_oh = {}
            for aid in self.agent_ids:
                logits = self.actor_targets[aid](next_obs[aid], next_u_oh[aid])
                target_actions_oh[aid] = F.one_hot(logits.argmax(dim=-1), self.n_actions).float()
            joint_target_actions = torch.cat([target_actions_oh[aid] for aid in self.agent_ids], dim=-1)
        total_loss = 0.0
        for aid in self.agent_ids:
            r = torch.tensor([e.rewards[aid] for e in batch], dtype=torch.float32, device=self.device)
            done = torch.tensor([float(e.dones[aid]) for e in batch], dtype=torch.float32, device=self.device)
            joint_actions = torch.cat([actions_oh[a] for a in self.agent_ids], dim=-1)
            q = self.critics[aid](joint_obs_u, joint_actions)
            with torch.no_grad():
                q_next = self.critic_targets[aid](joint_next_obs_u, joint_target_actions)
                y = r + self.gamma * (1 - done) * q_next
            critic_loss = F.mse_loss(q, y)
            self.critic_optims[aid].zero_grad()
            critic_loss.backward()
            self.critic_optims[aid].step()
            own_logits = self.actors[aid](obs[aid], u_oh[aid])
            own_soft = gumbel_softmax_sample(own_logits, hard=False)
            joint_actions_for_actor = torch.cat([own_soft if a == aid else actions_oh[a].detach() for a in self.agent_ids], dim=-1)
            actor_loss = -self.critics[aid](joint_obs_u, joint_actions_for_actor).mean()
            self.actor_optims[aid].zero_grad()
            actor_loss.backward()
            self.actor_optims[aid].step()
            total_loss += critic_loss.item() + actor_loss.item()
        self._soft_update_targets()
        return total_loss / self.n_agents

    def _soft_update_targets(self):
        for aid in self.agent_ids:
            for p, tp in zip(self.actors[aid].parameters(), self.actor_targets[aid].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.critics[aid].parameters(), self.critic_targets[aid].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

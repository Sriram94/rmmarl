from __future__ import annotations
from typing import Dict, List, Optional
import copy
import numpy as np
import torch
import torch.nn.functional as F
from ..base import Agent
from .networks import DDPGActor, DDPGCritic, PerRMStateActor, PerRMStateCritic, ContinuousOpponentModel
from .replay_buffer import ContReplayBuffer, ContExperience
from ..common import one_hot, one_hot_batch

class ContinuousRMAgent(Agent):
    variant = 'base'
    uses_counterfactual = True
    has_opponent_model = True

    def __init__(self, obs_dim: int, n_rm_states: int, action_dim: int, opponent_ids: List[str], n_opponent_actions: int, device: str='cpu', actor_lr: float=0.01, critic_lr: float=0.01, gamma: float=0.9, tau: float=0.01, buffer_size: int=200000, noise_std: float=0.2, seed: int=0):
        self.obs_dim, self.n_rm_states, self.action_dim = (obs_dim, n_rm_states, action_dim)
        self.gamma, self.tau, self.device = (gamma, tau, device)
        self.noise_std = noise_std
        self._actor_lr, self._critic_lr = (actor_lr, critic_lr)
        self.rng = np.random.default_rng(seed)
        self.buffer = ContReplayBuffer(capacity=buffer_size, seed=seed)
        self.opponent_ids = opponent_ids
        self.n_opponent_actions = n_opponent_actions
        self.opp_action_dim = len(opponent_ids) * n_opponent_actions
        self.opponent_models = {oid: ContinuousOpponentModel(obs_dim, n_rm_states, n_opponent_actions).to(device) for oid in opponent_ids}
        self.opponent_optims = {oid: torch.optim.Adam(m.parameters(), lr=critic_lr) for oid, m in self.opponent_models.items()}
        self._prev_opponent_actions = {oid: np.zeros(n_opponent_actions, dtype=np.float32) for oid in opponent_ids}

    def _opp_actions_to_vec(self, opp_actions: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.opponent_ids:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([opp_actions.get(oid, np.zeros(self.n_opponent_actions, dtype=np.float32)) for oid in self.opponent_ids]).astype(np.float32)

    def _actor_online(self, obs_t, u_idx):
        raise NotImplementedError

    def _train_step(self, batch):
        raise NotImplementedError

    def act(self, obs_vec: np.ndarray, u_index: int, epsilon: float) -> np.ndarray:
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self._actor_online(obs_t, u_index).squeeze(0).cpu().numpy()
        noise = self.rng.normal(0, self.noise_std * max(epsilon, 0.05), size=self.action_dim).astype(np.float32)
        return np.clip(a + noise, -1.0, 1.0)

    def predict_opponent_actions(self, obs_vec, u_index) -> Dict[str, np.ndarray]:
        state_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_oh = torch.tensor(one_hot(u_index, self.n_rm_states), device=self.device).unsqueeze(0)
        preds = {}
        for oid, model in self.opponent_models.items():
            prev_a = torch.tensor(self._prev_opponent_actions[oid], device=self.device).unsqueeze(0)
            preds[oid] = model.predict_action(state_t, u_oh, prev_a).squeeze(0).cpu().numpy()
        return preds

    def observe_opponents(self, obs_vec, u_index, joint_actions, prev_opponent_actions) -> None:
        state_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_oh = torch.tensor(one_hot(u_index, self.n_rm_states), device=self.device).unsqueeze(0)
        for oid, model in self.opponent_models.items():
            if oid not in joint_actions:
                continue
            prev_a = torch.tensor(prev_opponent_actions.get(oid, np.zeros(self.n_opponent_actions, dtype=np.float32)), dtype=torch.float32, device=self.device).unsqueeze(0)
            target = torch.tensor(joint_actions[oid], dtype=torch.float32, device=self.device).unsqueeze(0)
            loss = model.loss(state_t, u_oh, prev_a, target)
            opt = self.opponent_optims[oid]
            opt.zero_grad()
            loss.backward()
            opt.step()
        self._prev_opponent_actions.update({oid: joint_actions[oid] for oid in self.opponent_ids if oid in joint_actions})

    def _predict_next_opp_vec_batch(self, s_next: torch.Tensor, u_next_oh: torch.Tensor, opp_action_batch: torch.Tensor):
        if not self.opponent_ids or self.opp_action_dim == 0:
            return None
        parts = []
        for i, oid in enumerate(self.opponent_ids):
            model = self.opponent_models[oid]
            prev_a = opp_action_batch[:, i * self.n_opponent_actions:(i + 1) * self.n_opponent_actions]
            parts.append(model.forward(s_next, u_next_oh, prev_a))
        return torch.cat(parts, dim=-1)

    def remember(self, s, u_idx, a, r, s_next, u_next_idx, done, cf_targets=None, u_to_index=None, opp_action=None) -> None:
        assert cf_targets is not None and u_to_index is not None
        if self.opp_action_dim > 0:
            assert opp_action is not None, f"{self.variant} has an opponent model and requires the ACTUAL opponent action(s) taken at this step (Line 8's 'a', from the rollout) -- pass opp_action."
            opp_vec = self._opp_actions_to_vec(opp_action)
        else:
            opp_vec = np.zeros(0, dtype=np.float32)
        exps = []
        for u, (u_next, reward) in cf_targets.items():
            exps.append(ContExperience(state=s, u_index=u_to_index(u), action=a, reward=reward, next_state=s_next, next_u_index=u_to_index(u_next), done=False, opp_action=opp_vec))
        self.buffer.push_many(exps)

    def train(self, batch_size: int=64) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        return self._train_step(batch)

    def _batch_tensors(self, batch):
        s = torch.tensor(np.stack([e.state for e in batch]), dtype=torch.float32, device=self.device)
        u_idx = torch.tensor([e.u_index for e in batch], dtype=torch.long, device=self.device)
        a = torch.tensor(np.stack([e.action for e in batch]), dtype=torch.float32, device=self.device)
        r = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        s_next = torch.tensor(np.stack([e.next_state for e in batch]), dtype=torch.float32, device=self.device)
        u_next_idx = torch.tensor([e.next_u_index for e in batch], dtype=torch.long, device=self.device)
        done = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        if self.opp_action_dim > 0:
            opp_action = torch.tensor(np.stack([e.opp_action for e in batch]), dtype=torch.float32, device=self.device)
        else:
            opp_action = None
        return (s, u_idx, a, r, s_next, u_next_idx, done, opp_action)

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

class DCROMContinuousAgent(ContinuousRMAgent):
    variant = 'DCROM-continuous'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = DDPGActor(self.obs_dim, self.n_rm_states, self.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = DDPGCritic(self.obs_dim, self.n_rm_states, self.action_dim, opp_action_dim=self.opp_action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self._critic_lr)

    def _actor_online(self, obs_t, u_idx):
        u_oh = one_hot_batch(torch.tensor([u_idx]), self.n_rm_states).to(self.device)
        return self.actor(obs_t, u_oh)

    def _train_step(self, batch):
        s, u_idx, a, r, s_next, u_next_idx, done, opp_action = self._batch_tensors(batch)
        u_oh = one_hot_batch(u_idx, self.n_rm_states)
        u_next_oh = one_hot_batch(u_next_idx, self.n_rm_states)
        with torch.no_grad():
            a_next = self.actor_target(s_next, u_next_oh)
            next_opp_action = self._predict_next_opp_vec_batch(s_next, u_next_oh, opp_action)
            q_next = self.critic_target(s_next, u_next_oh, a_next, next_opp_action)
            y = r + self.gamma * (1 - done) * q_next
        q = self.critic(s, u_oh, a, opp_action)
        critic_loss = F.mse_loss(q, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        actor_loss = -self.critic(s, u_oh, self.actor(s, u_oh), opp_action).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return critic_loss.item() + actor_loss.item()

class DQROMContinuousAgent(ContinuousRMAgent):
    variant = 'DQROM-continuous'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = PerRMStateActor(self.obs_dim, self.n_rm_states, self.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = PerRMStateCritic(self.obs_dim, self.n_rm_states, self.action_dim, opp_action_dim=self.opp_action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self._critic_lr)

    def _actor_online(self, obs_t, u_idx):
        return self.actor(obs_t, u_idx)

    def _train_step(self, batch):
        s, u_idx, a, r, s_next, u_next_idx, done, opp_action = self._batch_tensors(batch)
        with torch.no_grad():
            a_next = self.actor_target.forward_batched(s_next, u_next_idx)
            u_next_oh = one_hot_batch(u_next_idx, self.n_rm_states)
            next_opp_action = self._predict_next_opp_vec_batch(s_next, u_next_oh, opp_action)
            q_next = self.critic_target.forward_batched(s_next, u_next_idx, a_next, next_opp_action)
            y = r + self.gamma * (1 - done) * q_next
        q = self.critic.forward_batched(s, u_idx, a, opp_action)
        critic_loss = F.mse_loss(q, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        actor_actions = self.actor.forward_batched(s, u_idx)
        actor_loss = -self.critic.forward_batched(s, u_idx, actor_actions, opp_action).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return critic_loss.item() + actor_loss.item()

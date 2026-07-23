from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from .base import Agent
from .networks import ShallowCrossProductQNet
from .replay_buffer import ReplayBuffer, Experience
from .common import one_hot_batch, epsilon_greedy
from .baselines import DQNBaseline, DQRMBaseline

class _HighLevelController:

    def __init__(self, obs_dim, n_rm_states, n_options, device='cpu', lr=0.01, gamma=0.9, buffer_size=200000, target_update_every=100, seed=0, hidden=256):
        self.n_options = n_options
        self.n_rm_states = n_rm_states
        self.gamma = gamma
        self.device = device
        self.target_update_every = target_update_every
        self.rng = np.random.default_rng(seed)
        self._steps = 0
        self.buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        self.q_eval = ShallowCrossProductQNet(obs_dim, n_rm_states, n_options, hidden=hidden).to(device)
        self.q_target = ShallowCrossProductQNet(obs_dim, n_rm_states, n_options, hidden=hidden).to(device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.optim = torch.optim.Adam(self.q_eval.parameters(), lr=lr)

    def select_option(self, obs_vec, u_index, epsilon) -> int:
        s = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_oh = one_hot_batch(torch.tensor([u_index]), self.n_rm_states).to(self.device)
        with torch.no_grad():
            q = self.q_eval(s, u_oh).squeeze(0)
        return epsilon_greedy(q, epsilon, self.rng)

    def push_macro_transition(self, start_obs, start_u_idx, option, discounted_return, end_obs, end_u_idx, effective_gamma, done):
        self.buffer.push(Experience(state=start_obs, u_index=start_u_idx, action=option, reward=discounted_return, next_state=end_obs, next_u_index=end_u_idx, done=done))
        self._last_effective_gamma = effective_gamma

    def train(self, batch_size: int=64) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        s = torch.tensor(np.stack([e.state for e in batch]), dtype=torch.float32, device=self.device)
        u_idx = torch.tensor([e.u_index for e in batch], dtype=torch.long, device=self.device)
        opt = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        ret = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        s_next = torch.tensor(np.stack([e.next_state for e in batch]), dtype=torch.float32, device=self.device)
        u_next_idx = torch.tensor([e.next_u_index for e in batch], dtype=torch.long, device=self.device)
        done = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        u_oh = one_hot_batch(u_idx, self.n_rm_states)
        u_next_oh = one_hot_batch(u_next_idx, self.n_rm_states)
        q_so = self.q_eval(s, u_oh).gather(1, opt.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s_next, u_next_oh).max(dim=1).values
            y = ret + self.gamma ** getattr(self, '_last_effective_gamma', 1) * (1 - done) * q_next
        loss = F.mse_loss(q_so, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._steps += 1
        if self._steps % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
        return loss.item()

class _HierarchicalBase(Agent):
    n_options = 4
    option_duration = 8

    def __init__(self, obs_dim: int, n_rm_states: int, n_actions: int, device: str='cpu', gamma: float=0.9, seed: int=0, **low_kwargs):
        self.obs_dim, self.n_rm_states, self.n_actions = (obs_dim, n_rm_states, n_actions)
        self.gamma, self.device = (gamma, device)
        self.high = _HighLevelController(obs_dim, n_rm_states, self.n_options, device=device, gamma=gamma, seed=seed)
        self.low_controllers = [self._make_low_controller(obs_dim, n_rm_states, n_actions, device, gamma, seed + 1000 + k, low_kwargs) for k in range(self.n_options)]
        self._reset_option_state()
        self._last_option = 0

    def _make_low_controller(self, obs_dim, n_rm_states, n_actions, device, gamma, seed, low_kwargs):
        raise NotImplementedError

    def _reset_option_state(self):
        self._option = None
        self._start_obs = None
        self._start_u_idx = None
        self._acc_return = 0.0
        self._gamma_pow = 1.0
        self._k = 0

    def act(self, obs_vec, u_index, epsilon) -> int:
        if self._option is None:
            self._option = self.high.select_option(obs_vec, u_index, epsilon)
            self._start_obs, self._start_u_idx = (obs_vec, u_index)
            self._acc_return, self._gamma_pow, self._k = (0.0, 1.0, 0)
        return self.low_controllers[self._option].act(obs_vec, u_index, epsilon)

    def observe_opponents(self, obs_vec, u_index, joint_actions, prev_opponent_actions) -> None:
        self.low_controllers[self._option].observe_opponents(obs_vec, u_index, joint_actions, prev_opponent_actions)

    def remember(self, s, u_idx, a, r, s_next, u_next_idx, done, cf_targets=None, u_to_index=None) -> None:
        option = self._option
        self._last_option = option
        self.low_controllers[option].remember(s, u_idx, a, r, s_next, u_next_idx, done, cf_targets=cf_targets, u_to_index=u_to_index)
        self._acc_return += self._gamma_pow * r
        self._gamma_pow *= self.gamma
        self._k += 1
        if self._k >= self.option_duration or done:
            self.high.push_macro_transition(self._start_obs, self._start_u_idx, option, self._acc_return, s_next, u_next_idx, effective_gamma=self._k, done=done)
            self._reset_option_state()

    def train(self, batch_size: int=64) -> Optional[float]:
        low_loss = self.low_controllers[self._last_option].train(batch_size=batch_size)
        high_loss = self.high.train(batch_size=batch_size)
        return low_loss if low_loss is not None else high_loss

    def predict_opponent_actions(self, obs_vec, u_index):
        return self.low_controllers[self._last_option].predict_opponent_actions(obs_vec, u_index)

class DHRLBaseline(_HierarchicalBase):
    variant = 'DHRL'
    uses_counterfactual = False
    has_opponent_model = False

    def _make_low_controller(self, obs_dim, n_rm_states, n_actions, device, gamma, seed, low_kwargs):
        return DQNBaseline(obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed, **low_kwargs)

class DHRLRMBaseline(_HierarchicalBase):
    variant = 'DHRL-RM'
    uses_counterfactual = True
    has_opponent_model = False

    def _make_low_controller(self, obs_dim, n_rm_states, n_actions, device, gamma, seed, low_kwargs):
        return DQRMBaseline(obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed, **low_kwargs)

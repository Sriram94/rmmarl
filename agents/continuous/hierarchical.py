from __future__ import annotations
from typing import Optional
import numpy as np
from ..base import Agent
from ..hierarchical import _HighLevelController
from .baselines import DDPGBaseline, DQRMContinuousBaseline

class _ContinuousHierarchicalBase(Agent):
    n_options = 4
    option_duration = 8

    def __init__(self, obs_dim: int, n_rm_states: int, action_dim: int, device: str='cpu', gamma: float=0.9, seed: int=0, **low_kwargs):
        self.obs_dim, self.n_rm_states, self.action_dim = (obs_dim, n_rm_states, action_dim)
        self.gamma, self.device = (gamma, device)
        self.high = _HighLevelController(obs_dim, n_rm_states, self.n_options, device=device, gamma=gamma, seed=seed)
        self.low_controllers = [self._make_low_controller(obs_dim, n_rm_states, action_dim, device, gamma, seed + 1000 + k, low_kwargs) for k in range(self.n_options)]
        self._reset_option_state()
        self._last_option = 0

    def _make_low_controller(self, obs_dim, n_rm_states, action_dim, device, gamma, seed, low_kwargs):
        raise NotImplementedError

    def _reset_option_state(self):
        self._option = None
        self._start_obs = None
        self._start_u_idx = None
        self._acc_return = 0.0
        self._gamma_pow = 1.0
        self._k = 0

    def act(self, obs_vec, u_index, epsilon) -> np.ndarray:
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

class DHRLContinuousBaseline(_ContinuousHierarchicalBase):
    variant = 'DHRL-continuous'
    uses_counterfactual = False
    has_opponent_model = False

    def _make_low_controller(self, obs_dim, n_rm_states, action_dim, device, gamma, seed, low_kwargs):
        return DDPGBaseline(obs_dim, n_rm_states, action_dim, device=device, gamma=gamma, seed=seed, **low_kwargs)

class DHRLRMContinuousBaseline(_ContinuousHierarchicalBase):
    variant = 'DHRL-RM-continuous'
    uses_counterfactual = True
    has_opponent_model = False

    def _make_low_controller(self, obs_dim, n_rm_states, action_dim, device, gamma, seed, low_kwargs):
        return DQRMContinuousBaseline(obs_dim, n_rm_states, action_dim, device=device, gamma=gamma, seed=seed, **low_kwargs)

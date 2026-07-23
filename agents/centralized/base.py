from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

class CentralizedController:
    variant = 'base'
    on_policy = False

    def __init__(self, agent_ids: List[str], obs_dim: int, n_rm_states: int, n_actions: int):
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.obs_dim = obs_dim
        self.n_rm_states = n_rm_states
        self.n_actions = n_actions

    def act(self, obs: Dict[str, np.ndarray], u_idx: Dict[str, int], epsilon: float) -> Dict[str, int]:
        raise NotImplementedError

    def store(self, obs, u_idx, actions, rewards, next_obs, next_u_idx, dones) -> None:
        raise NotImplementedError

    def train(self, batch_size: int=64) -> Optional[float]:
        raise NotImplementedError

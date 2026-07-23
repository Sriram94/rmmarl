from __future__ import annotations
from typing import Dict, Optional
import numpy as np

class Agent:
    variant = 'base'
    uses_counterfactual = False
    has_opponent_model = False

    def act(self, obs_vec: np.ndarray, u_index: int, epsilon: float) -> int:
        raise NotImplementedError

    def observe_opponents(self, obs_vec: np.ndarray, u_index: int, joint_actions: Dict[str, int], prev_opponent_actions: Dict[str, int]) -> None:
        pass

    def remember(self, s, u_idx, a, r, s_next, u_next_idx, done, cf_targets=None, u_to_index=None) -> None:
        raise NotImplementedError

    def train(self, batch_size: int=64) -> Optional[float]:
        raise NotImplementedError

    def predict_opponent_actions(self, obs_vec: np.ndarray, u_index: int) -> Dict[str, int]:
        raise NotImplementedError(f'{self.variant} has no opponent model')

from __future__ import annotations
import random
from collections import deque
from typing import Dict, List, NamedTuple
import numpy as np

class JointExperience(NamedTuple):
    obs: Dict[str, np.ndarray]
    u_idx: Dict[str, int]
    actions: Dict[str, int]
    rewards: Dict[str, float]
    next_obs: Dict[str, np.ndarray]
    next_u_idx: Dict[str, int]
    dones: Dict[str, bool]

class JointReplayBuffer:

    def __init__(self, capacity: int=100000, seed: int=None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, exp: JointExperience):
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> List[JointExperience]:
        batch_size = min(batch_size, len(self.buffer))
        return self.rng.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

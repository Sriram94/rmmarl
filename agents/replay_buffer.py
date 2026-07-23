from __future__ import annotations
import random
from collections import deque
from typing import NamedTuple, List
import numpy as np

class Experience(NamedTuple):
    state: np.ndarray
    u_index: int
    action: int
    reward: float
    next_state: np.ndarray
    next_u_index: int
    done: bool
    opp_action: np.ndarray = None

class ReplayBuffer:

    def __init__(self, capacity: int=200000, seed: int=None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, exp: Experience):
        self.buffer.append(exp)

    def push_many(self, exps: List[Experience]):
        self.buffer.extend(exps)

    def sample(self, batch_size: int) -> List[Experience]:
        batch_size = min(batch_size, len(self.buffer))
        return self.rng.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

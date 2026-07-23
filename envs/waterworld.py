from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
COLORS = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta']

@dataclass
class WaterworldState:
    pursuer_pos: np.ndarray
    pursuer_vel: np.ndarray
    poison_pos: np.ndarray
    poison_vel: np.ndarray
    food_pos: np.ndarray
    food_vel: np.ndarray
    t: int
    touched_food: str = None
    collision: bool = False
_ACTIONS = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float32)
N_ACTIONS = len(_ACTIONS)

class WaterworldEnv:

    def __init__(self, max_steps: int=1500, capture_radius: float=0.04, accel: float=0.006, damping: float=0.98, seed: int=None, n_pursuers: int=3, n_poison: int=2):
        self.n_pursuers = n_pursuers
        self.n_poison = n_poison
        self.n_food = len(COLORS)
        self.max_steps = max_steps
        self.capture_radius = capture_radius
        self.accel = accel
        self.damping = damping
        self.rng = np.random.default_rng(seed)
        self.agent_ids = [f'pursuer_{i}' for i in range(self.n_pursuers)] + [f'poison_{i}' for i in range(self.n_poison)]
        self.state: WaterworldState = None

    def _rand_pos(self, n):
        return self.rng.uniform(0.05, 0.95, size=(n, 2)).astype(np.float32)

    def _rand_vel(self, n, scale=0.004):
        return self.rng.uniform(-scale, scale, size=(n, 2)).astype(np.float32)

    def reset(self) -> WaterworldState:
        self.state = WaterworldState(pursuer_pos=self._rand_pos(self.n_pursuers), pursuer_vel=np.zeros((self.n_pursuers, 2), dtype=np.float32), poison_pos=self._rand_pos(self.n_poison), poison_vel=np.zeros((self.n_poison, 2), dtype=np.float32), food_pos=self._rand_pos(self.n_food), food_vel=self._rand_vel(self.n_food), t=0)
        return self.state

    def _move_and_bounce(self, pos, vel):
        pos = pos + vel
        for d in (0, 1):
            over = pos[:, d] > 1.0
            under = pos[:, d] < 0.0
            vel[over | under, d] *= -1
            pos[:, d] = np.clip(pos[:, d], 0.0, 1.0)
        return (pos, vel)

    def observe(self, agent_id: str, state: WaterworldState) -> np.ndarray:
        if agent_id.startswith('pursuer'):
            idx = int(agent_id.split('_')[1])
            own_pos, own_vel = (state.pursuer_pos[idx], state.pursuer_vel[idx])
        else:
            idx = int(agent_id.split('_')[1])
            own_pos, own_vel = (state.poison_pos[idx], state.poison_vel[idx])
        others = np.concatenate([(state.pursuer_pos - own_pos).flatten(), (state.poison_pos - own_pos).flatten()])
        food_rel = (state.food_pos - own_pos).flatten()
        return np.concatenate([own_pos, own_vel, others, food_rel]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return 2 + 2 + (self.n_pursuers + self.n_poison) * 2 + self.n_food * 2

    def step(self, actions: Dict[str, int]):
        s = self.state
        p_acc = np.array([_ACTIONS[actions[f'pursuer_{i}']] for i in range(self.n_pursuers)], dtype=np.float32) * self.accel
        z_acc = np.array([_ACTIONS[actions[f'poison_{i}']] for i in range(self.n_poison)], dtype=np.float32) * self.accel
        s.pursuer_vel = s.pursuer_vel * self.damping + p_acc
        s.poison_vel = s.poison_vel * self.damping + z_acc
        s.pursuer_pos, s.pursuer_vel = self._move_and_bounce(s.pursuer_pos, s.pursuer_vel)
        s.poison_pos, s.poison_vel = self._move_and_bounce(s.poison_pos, s.poison_vel)
        s.food_pos, s.food_vel = self._move_and_bounce(s.food_pos, s.food_vel)
        touched = None
        for f_idx in range(self.n_food):
            d = np.linalg.norm(s.pursuer_pos - s.food_pos[f_idx], axis=1)
            if np.any(d < self.capture_radius):
                touched = COLORS[f_idx]
                s.food_pos[f_idx] = self._rand_pos(1)[0]
                break
        s.touched_food = touched
        collision = False
        for pi in range(self.n_pursuers):
            d = np.linalg.norm(s.poison_pos - s.pursuer_pos[pi], axis=1)
            if np.any(d < self.capture_radius):
                collision = True
                break
        s.collision = collision
        s.t += 1
        env_done = collision or s.t >= self.max_steps
        info = {'touched_food': touched, 'collision': collision}
        return (s, env_done, info)

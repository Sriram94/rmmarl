from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
INGREDIENTS = ('onion', 'tomato', 'carrot')
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0)]
INTERACT, NOOP = (4, 5)
N_ACTIONS = 6

@dataclass
class OvercookedState:
    agent_pos: np.ndarray
    agent_carrying: list
    pot_ingredients: set
    cook_timer: int
    soup_ready: bool
    t: int
    placed_ingredient: Optional[str] = None
    soup_just_ready: bool = False
    delivered: bool = False

class OvercookedEnv:

    def __init__(self, grid_size: int=15, max_steps: int=12000, cook_time: int=10, seed: int=None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.cook_time = cook_time
        self.rng = np.random.default_rng(seed)
        self.agent_ids = ['cook_0', 'cook_1']
        self.pot_pos = np.array([0, grid_size // 2])
        self.delivery_pos = np.array([grid_size - 1, grid_size // 2])
        self.ingredient_counters = {'onion': np.array([grid_size // 2, 0]), 'tomato': np.array([1, grid_size - 1]), 'carrot': np.array([grid_size - 2, grid_size - 1])}
        self._static_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._static_map[self.pot_pos[0], self.pot_pos[1]] = 1.0
        self._static_map[self.delivery_pos[0], self.delivery_pos[1]] = 2.0
        for i, ing in enumerate(INGREDIENTS):
            cpos = self.ingredient_counters[ing]
            self._static_map[cpos[0], cpos[1]] = 3.0 + i
        self.state: OvercookedState = None

    def reset(self) -> OvercookedState:
        starts = np.array([[self.grid_size // 2, self.grid_size // 2 - 1], [self.grid_size // 2, self.grid_size // 2 + 1]])
        self.state = OvercookedState(agent_pos=starts.copy(), agent_carrying=[None, None], pot_ingredients=set(), cook_timer=0, soup_ready=False, t=0)
        return self.state

    def observe(self, agent_id: str, state: OvercookedState) -> np.ndarray:
        idx = self.agent_ids.index(agent_id)
        other = 1 - idx
        own_pos = state.agent_pos[idx].astype(np.float32)
        other_rel = (state.agent_pos[other] - state.agent_pos[idx]).astype(np.float32)
        carrying_oh = self._carry_onehot(state.agent_carrying[idx])
        other_carrying_oh = self._carry_onehot(state.agent_carrying[other])
        rel_pot = (self.pot_pos - state.agent_pos[idx]).astype(np.float32)
        rel_delivery = (self.delivery_pos - state.agent_pos[idx]).astype(np.float32)
        rel_ingredients = np.concatenate([(self.ingredient_counters[i] - state.agent_pos[idx]).astype(np.float32) for i in INGREDIENTS])
        full_map = self._static_map.copy()
        full_map[state.agent_pos[idx][0], state.agent_pos[idx][1]] += 10.0
        full_map[state.agent_pos[other][0], state.agent_pos[other][1]] += 20.0
        return np.concatenate([own_pos, other_rel, carrying_oh, other_carrying_oh, rel_pot, rel_delivery, rel_ingredients, full_map.flatten()]).astype(np.float32)

    @staticmethod
    def _carry_onehot(item) -> np.ndarray:
        options = [None, 'onion', 'tomato', 'carrot', 'soup']
        v = np.zeros(len(options), dtype=np.float32)
        v[options.index(item)] = 1.0
        return v

    @property
    def obs_dim(self) -> int:
        return 2 + 2 + 5 + 5 + 2 + 2 + 6 + self.grid_size * self.grid_size

    def step(self, actions: Dict[str, int]):
        s = self.state
        s.placed_ingredient = None
        s.soup_just_ready = False
        s.delivered = False
        for idx, aid in enumerate(self.agent_ids):
            a = actions[aid]
            if a == INTERACT:
                self._interact(idx)
            elif a != NOOP:
                dx, dy = ACTION_DELTAS[a]
                new_pos = s.agent_pos[idx] + np.array([dx, dy])
                new_pos = np.clip(new_pos, 0, self.grid_size - 1)
                s.agent_pos[idx] = new_pos
        if s.cook_timer > 0:
            s.cook_timer -= 1
            if s.cook_timer == 0:
                s.soup_ready = True
                s.soup_just_ready = True
        s.t += 1
        env_done = s.t >= self.max_steps
        info = {'placed_ingredient': s.placed_ingredient, 'soup_just_ready': s.soup_just_ready, 'delivered': s.delivered}
        return (s, env_done, info)

    def _interact(self, idx: int):
        s = self.state
        pos = s.agent_pos[idx]
        carrying = s.agent_carrying[idx]
        if carrying is None:
            for ing, cpos in self.ingredient_counters.items():
                if np.array_equal(pos, cpos):
                    s.agent_carrying[idx] = ing
                    return
            if s.soup_ready and np.array_equal(pos, self.pot_pos):
                s.agent_carrying[idx] = 'soup'
                s.soup_ready = False
                s.pot_ingredients = set()
                return
        else:
            if carrying in INGREDIENTS and np.array_equal(pos, self.pot_pos) and (carrying not in s.pot_ingredients) and (s.cook_timer == 0) and (not s.soup_ready):
                s.pot_ingredients.add(carrying)
                s.agent_carrying[idx] = None
                s.placed_ingredient = carrying
                if s.pot_ingredients == set(INGREDIENTS):
                    s.cook_timer = self.cook_time
                return
            if carrying == 'soup' and np.array_equal(pos, self.delivery_pos):
                s.agent_carrying[idx] = None
                s.delivered = True
                return

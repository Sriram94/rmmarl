from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Optional, Tuple
NO_EVENT = None

@dataclass
class RewardMachine:
    states: Tuple[Hashable, ...]
    initial_state: Hashable
    terminal_states: Tuple[Hashable, ...]
    events: Tuple[Hashable, ...]
    transitions: Dict[Tuple[Hashable, Hashable], Hashable] = field(default_factory=dict)
    rewards: Dict[Tuple[Hashable, Hashable], float] = field(default_factory=dict)
    labelling_fn: Optional[Callable[[object], Hashable]] = None

    def __post_init__(self):
        assert self.initial_state in self.states
        for t in self.terminal_states:
            assert t in self.states

    def is_terminal(self, u: Hashable) -> bool:
        return u in self.terminal_states

    def label(self, env_state) -> Hashable:
        if self.labelling_fn is None:
            raise ValueError('No labelling function attached to this RM.')
        return self.labelling_fn(env_state)

    def delta(self, u: Hashable, event: Hashable) -> Hashable:
        return self.transitions.get((u, event), u)

    def sigma(self, u: Hashable, u_next: Hashable) -> float:
        return self.rewards.get((u, u_next), 0.0)

    def step(self, u: Hashable, env_state) -> Tuple[Hashable, float]:
        event = self.label(env_state)
        u_next = self.delta(u, event)
        r = self.sigma(u, u_next)
        return (u_next, r)

    def counterfactual_step(self, u: Hashable, env_state) -> Tuple[Hashable, float]:
        return self.step(u, env_state)

    @property
    def num_states(self) -> int:
        return len(self.states)

    def state_index(self, u: Hashable) -> int:
        return self.states.index(u)

from __future__ import annotations
from itertools import chain, combinations
from .reward_machine import RewardMachine
from .overcooked import INGREDIENTS
READY = 'READY'

def _powerset_states():
    return [frozenset(c) for c in chain.from_iterable((combinations(INGREDIENTS, r) for r in range(len(INGREDIENTS) + 1)))]

def build_overcooked_rm() -> RewardMachine:
    progress_states = _powerset_states()
    states = progress_states + [READY]
    initial = frozenset()
    full = frozenset(INGREDIENTS)
    transitions = {}
    rewards = {}
    for prog in progress_states:
        if prog == full:
            transitions[prog, 'soup_ready'] = READY
            rewards[prog, READY] = 0.0
            continue
        for ing in INGREDIENTS:
            if ing in prog:
                continue
            new_prog = prog | {ing}
            transitions[prog, f'placed_{ing}'] = new_prog
            rewards[prog, new_prog] = 0.0
    transitions[READY, 'delivered'] = initial
    rewards[READY, initial] = 100.0

    def labelling_fn(env_state):
        if env_state.delivered:
            return 'delivered'
        if env_state.soup_just_ready:
            return 'soup_ready'
        if env_state.placed_ingredient is not None:
            return f'placed_{env_state.placed_ingredient}'
        return None
    events = tuple([f'placed_{i}' for i in INGREDIENTS] + ['soup_ready', 'delivered'])
    return RewardMachine(states=tuple(states), initial_state=initial, terminal_states=(), events=events, transitions=transitions, rewards=rewards, labelling_fn=labelling_fn)

def build_overcooked_rms(agent_ids):
    rm = build_overcooked_rm()
    return {aid: rm for aid in agent_ids}

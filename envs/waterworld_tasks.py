from __future__ import annotations
from itertools import product
from typing import List, Tuple
from .reward_machine import RewardMachine
POISONED = 'POISONED'
TASKS = {1: [['red', 'green']], 2: [['blue', 'cyan']], 3: [['magenta', 'yellow']], 4: [['red', 'green'], ['blue', 'cyan']], 5: [['red', 'green'], ['blue', 'cyan'], ['magenta', 'yellow']], 6: [['red', 'green', 'blue'], ['cyan', 'magenta', 'yellow']]}

def _progress_states(seq_lens: List[int]):
    return list(product(*[range(n + 1) for n in seq_lens]))

def build_pursuer_rm(task_id: int) -> RewardMachine:
    sequences = TASKS[task_id]
    seq_lens = [len(s) for s in sequences]
    prog_states = _progress_states(seq_lens)
    initial = tuple((0 for _ in sequences))
    states = list(prog_states) + [POISONED]
    terminal_states = (POISONED,)
    transitions = {}
    rewards = {}

    def is_complete(prog):
        return all((p == n for p, n in zip(prog, seq_lens)))
    for prog in prog_states:
        if is_complete(prog):
            continue
        transitions[prog, 'collision'] = POISONED
        rewards[prog, POISONED] = -10.0
        for i, (seq, p) in enumerate(zip(sequences, prog)):
            if p >= len(seq):
                continue
            next_color = seq[p]
            new_prog = list(prog)
            new_prog[i] = p + 1
            new_prog = tuple(new_prog)
            if is_complete(new_prog):
                target = initial
                reward = 10.0
            else:
                target = new_prog
                reward = 0.0
            transitions[prog, next_color] = target
            rewards[prog, target] = reward

    def labelling_fn(env_state):
        if env_state.collision:
            return 'collision'
        return env_state.touched_food
    events = tuple(set((c for seq in sequences for c in seq)) | {'collision'})
    return RewardMachine(states=tuple(states), initial_state=initial, terminal_states=terminal_states, events=events, transitions=transitions, rewards=rewards, labelling_fn=labelling_fn)

def build_poison_rm() -> RewardMachine:
    states = ('ACTIVE', 'DONE')
    transitions = {('ACTIVE', 'collision'): 'DONE'}
    rewards = {('ACTIVE', 'DONE'): 10.0}

    def labelling_fn(env_state):
        return 'collision' if env_state.collision else None
    return RewardMachine(states=states, initial_state='ACTIVE', terminal_states=('DONE',), events=('collision',), transitions=transitions, rewards=rewards, labelling_fn=labelling_fn)

def build_waterworld_rms(task_id: int, n_pursuers: int, n_poison: int):
    rms = {}
    for i in range(n_pursuers):
        rms[f'pursuer_{i}'] = build_pursuer_rm(task_id)
    for i in range(n_poison):
        rms[f'poison_{i}'] = build_poison_rm()
    return rms
MULTITASK_ORDER = (1, 2, 3, 4, 5, 6)

def build_multitask_pursuer_rm() -> RewardMachine:
    states = [POISONED]
    initial = None
    transitions = {}
    rewards = {}

    def is_complete(task_id, prog):
        seq_lens = [len(s) for s in TASKS[task_id]]
        return all((p == n for p, n in zip(prog, seq_lens)))
    for task_id in MULTITASK_ORDER:
        sequences = TASKS[task_id]
        seq_lens = [len(s) for s in sequences]
        for prog in _progress_states(seq_lens):
            if is_complete(task_id, prog):
                continue
            state_key = (task_id, prog)
            states.append(state_key)
            if initial is None and task_id == MULTITASK_ORDER[0] and all((p == 0 for p in prog)):
                initial = state_key
            transitions[state_key, 'collision'] = POISONED
            rewards[state_key, POISONED] = -10.0
            for i, (seq, p) in enumerate(zip(sequences, prog)):
                if p >= len(seq):
                    continue
                next_color = seq[p]
                new_prog = list(prog)
                new_prog[i] = p + 1
                new_prog = tuple(new_prog)
                if is_complete(task_id, new_prog):
                    next_task_id = MULTITASK_ORDER[(MULTITASK_ORDER.index(task_id) + 1) % len(MULTITASK_ORDER)]
                    target = (next_task_id, tuple((0 for _ in TASKS[next_task_id])))
                    reward = 10.0
                else:
                    target = (task_id, new_prog)
                    reward = 0.0
                transitions[state_key, next_color] = target
                rewards[state_key, target] = reward

    def labelling_fn(env_state):
        if env_state.collision:
            return 'collision'
        return env_state.touched_food
    events = tuple(set((c for t in MULTITASK_ORDER for seq in TASKS[t] for c in seq)) | {'collision'})
    return RewardMachine(states=tuple(states), initial_state=initial, terminal_states=(POISONED,), events=events, transitions=transitions, rewards=rewards, labelling_fn=labelling_fn)

def build_multitask_waterworld_rms(n_pursuers: int, n_poison: int):
    rms = {}
    for i in range(n_pursuers):
        rms[f'pursuer_{i}'] = build_multitask_pursuer_rm()
    for i in range(n_poison):
        rms[f'poison_{i}'] = build_poison_rm()
    return rms

from __future__ import annotations
from .reward_machine import RewardMachine
NEED_B, NEED_A = ('NEED_B', 'NEED_A')

def build_ant_rm() -> RewardMachine:
    states = (NEED_B, NEED_A)
    transitions = {(NEED_B, 'reached_B'): NEED_A, (NEED_A, 'reached_A'): NEED_B}
    rewards = {(NEED_B, NEED_A): 0.0, (NEED_A, NEED_B): 1000.0}

    def labelling_fn(env_state):
        if env_state.reached_point == 'B':
            return 'reached_B'
        if env_state.reached_point == 'A':
            return 'reached_A'
        return None
    return RewardMachine(states=states, initial_state=NEED_B, terminal_states=(), events=('reached_A', 'reached_B'), transitions=transitions, rewards=rewards, labelling_fn=labelling_fn)

def build_ant_rms(agent_ids):
    rm = build_ant_rm()
    return {aid: rm for aid in agent_ids}

def build_ant_state_reward_fns(env):
    return {aid: lambda next_state, action, aid=aid: env.torque_penalty(aid, action) for aid in env.agent_ids}

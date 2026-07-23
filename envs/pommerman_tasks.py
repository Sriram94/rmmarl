from __future__ import annotations
from .reward_machine import RewardMachine
INIT, SELF_COIN, OPP_COIN = ('INIT', 'SELF_COIN', 'OPP_COIN')
WIN, LOSE, DRAW = ('WIN', 'LOSE', 'DRAW')

def build_pommerman_rm(self_team: int, env) -> RewardMachine:
    agent_ids, team_of = (env.agent_ids, env.team_of)
    other_team = 1 - self_team
    states = (INIT, SELF_COIN, OPP_COIN, WIN, LOSE, DRAW)
    transitions = {(INIT, 'self_captured_coin'): SELF_COIN, (INIT, 'opp_captured_coin'): OPP_COIN, (INIT, 'self_team_died'): DRAW, (INIT, 'opp_team_died'): DRAW, (INIT, 'mutual_death'): DRAW, (INIT, 'timeout'): DRAW, (SELF_COIN, 'self_team_died'): DRAW, (SELF_COIN, 'opp_team_died'): WIN, (SELF_COIN, 'mutual_death'): DRAW, (SELF_COIN, 'timeout'): DRAW, (OPP_COIN, 'self_team_died'): LOSE, (OPP_COIN, 'opp_team_died'): DRAW, (OPP_COIN, 'mutual_death'): DRAW, (OPP_COIN, 'timeout'): DRAW}
    rewards = {(SELF_COIN, WIN): 1.0, (OPP_COIN, LOSE): -1.0}

    def labelling_fn(state):
        self_alive = any((state.agent_alive[i] for i, aid in enumerate(agent_ids) if team_of[aid] == self_team))
        opp_alive = any((state.agent_alive[i] for i, aid in enumerate(agent_ids) if team_of[aid] == other_team))
        if not self_alive and (not opp_alive):
            return 'mutual_death'
        if not self_alive:
            return 'self_team_died'
        if not opp_alive:
            return 'opp_team_died'
        if state.t >= env.max_steps:
            return 'timeout'
        if state.captured_by_team == self_team:
            return 'self_captured_coin'
        if state.captured_by_team == other_team:
            return 'opp_captured_coin'
        return None
    events = ('self_captured_coin', 'opp_captured_coin', 'self_team_died', 'opp_team_died', 'mutual_death', 'timeout')
    return RewardMachine(states=states, initial_state=INIT, terminal_states=(WIN, LOSE, DRAW), events=events, transitions=transitions, rewards=rewards, labelling_fn=labelling_fn)

def build_pommerman_rms(env):
    rm_by_team = {0: build_pommerman_rm(0, env), 1: build_pommerman_rm(1, env)}
    return {aid: rm_by_team[env.team_of[aid]] for aid in env.agent_ids}

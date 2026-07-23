from __future__ import annotations
from typing import List
from .maddpg import MADDPGTeam
from .mappo import MAPPOTeam
from .qmix import QMIXTeam
from .self_play_ppo import SelfPlayPPOTeam
CENTRALIZED_ALGORITHMS = ('maddpg', 'mappo', 'self_play_ppo', 'qmix')

def build_centralized_agent(algo: str, *, agent_ids: List[str], obs_dim: int, n_rm_states: int, n_actions: int, device: str='cpu', gamma: float=0.9, seed: int=0):
    algo = algo.lower()
    if algo == 'maddpg':
        return MADDPGTeam(agent_ids, obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed)
    if algo == 'mappo':
        return MAPPOTeam(agent_ids, obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed)
    if algo == 'self_play_ppo':
        return SelfPlayPPOTeam(agent_ids, obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed)
    if algo == 'qmix':
        return QMIXTeam(agent_ids, obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed)
    raise ValueError(f"Unknown centralized algorithm '{algo}'. Choose from {CENTRALIZED_ALGORITHMS}")

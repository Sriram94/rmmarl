from __future__ import annotations
from typing import List
from .ddpg_agent import DCROMContinuousAgent, DQROMContinuousAgent
from .baselines import DDPGBaseline, DDPGOMBaseline, MADDPGProxyBaseline, DQRMContinuousBaseline
from .hierarchical import DHRLContinuousBaseline, DHRLRMContinuousBaseline
CONTINUOUS_ALGORITHMS = ('dcrom', 'dqrom', 'ddpg', 'ddpg_om', 'ma_ddpg', 'dqrm', 'dhrl', 'dhrl_rm')

def build_continuous_agent(algo: str, *, obs_dim: int, n_rm_states: int, action_dim: int, opponent_ids: List[str]=None, n_opponent_actions: int=None, device: str='cpu', seed: int=0, gamma: float=0.9):
    algo = algo.lower()
    opponent_ids = opponent_ids or []
    n_opponent_actions = n_opponent_actions or action_dim
    if algo == 'dcrom':
        return DCROMContinuousAgent(obs_dim, n_rm_states, action_dim, opponent_ids=opponent_ids, n_opponent_actions=n_opponent_actions, device=device, gamma=gamma, seed=seed)
    if algo == 'dqrom':
        return DQROMContinuousAgent(obs_dim, n_rm_states, action_dim, opponent_ids=opponent_ids, n_opponent_actions=n_opponent_actions, device=device, gamma=gamma, seed=seed)
    if algo == 'ddpg':
        return DDPGBaseline(obs_dim, n_rm_states, action_dim, device=device, gamma=gamma, seed=seed)
    if algo == 'ddpg_om':
        return DDPGOMBaseline(obs_dim, n_rm_states, action_dim, opponent_ids=opponent_ids, n_opponent_actions=n_opponent_actions, device=device, gamma=gamma, seed=seed)
    if algo == 'ma_ddpg':
        return MADDPGProxyBaseline(obs_dim, n_rm_states, action_dim, opponent_ids=opponent_ids, n_opponent_actions=n_opponent_actions, device=device, gamma=gamma, seed=seed)
    if algo == 'dqrm':
        return DQRMContinuousBaseline(obs_dim, n_rm_states, action_dim, device=device, gamma=gamma, seed=seed)
    if algo == 'dhrl':
        return DHRLContinuousBaseline(obs_dim, n_rm_states, action_dim, device=device, gamma=gamma, seed=seed)
    if algo == 'dhrl_rm':
        return DHRLRMContinuousBaseline(obs_dim, n_rm_states, action_dim, device=device, gamma=gamma, seed=seed)
    raise ValueError(f"Unknown continuous algorithm '{algo}'. Choose from {CONTINUOUS_ALGORITHMS}")

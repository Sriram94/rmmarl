from __future__ import annotations
from typing import List
from .crom_agent import DCROMAgent, DQROMAgent
from .baselines import DQNBaseline, DQNOMBaseline, MADQNBaseline, DQRMBaseline
from .hierarchical import DHRLBaseline, DHRLRMBaseline
ALGORITHMS = ('dcrom', 'dqrom', 'dqn', 'dqn_om', 'ma_dqn', 'dqrm', 'dhrl', 'dhrl_rm')

def build_agent(algo: str, *, obs_dim: int, n_rm_states: int, n_actions: int, opponent_ids: List[str]=None, n_opponent_actions: int=None, device: str='cpu', seed: int=0, gamma: float=0.9, buffer_size: int=200000):
    algo = algo.lower()
    opponent_ids = opponent_ids or []
    n_opponent_actions = n_opponent_actions or n_actions
    if algo == 'dcrom':
        return DCROMAgent(agent_id=None, obs_dim=obs_dim, n_rm_states=n_rm_states, n_actions=n_actions, opponent_ids=opponent_ids, n_opponent_actions=n_opponent_actions, device=device, gamma=gamma, seed=seed, buffer_size=buffer_size)
    if algo == 'dqrom':
        return DQROMAgent(agent_id=None, obs_dim=obs_dim, n_rm_states=n_rm_states, n_actions=n_actions, opponent_ids=opponent_ids, n_opponent_actions=n_opponent_actions, device=device, gamma=gamma, seed=seed, buffer_size=buffer_size)
    if algo == 'dqn':
        return DQNBaseline(obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed, buffer_size=buffer_size)
    if algo == 'dqn_om':
        return DQNOMBaseline(obs_dim, n_rm_states, n_actions, opponent_ids=opponent_ids, n_opponent_actions=n_opponent_actions, device=device, gamma=gamma, seed=seed, buffer_size=buffer_size)
    if algo == 'ma_dqn':
        return MADQNBaseline(obs_dim, n_rm_states, n_actions, opponent_ids=opponent_ids, n_opponent_actions=n_opponent_actions, device=device, gamma=gamma, seed=seed, buffer_size=buffer_size)
    if algo == 'dqrm':
        return DQRMBaseline(obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed, buffer_size=buffer_size)
    if algo == 'dhrl':
        return DHRLBaseline(obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed, buffer_size=buffer_size)
    if algo == 'dhrl_rm':
        return DHRLRMBaseline(obs_dim, n_rm_states, n_actions, device=device, gamma=gamma, seed=seed, buffer_size=buffer_size)
    raise ValueError(f"Unknown algorithm '{algo}'. Choose from {ALGORITHMS}")

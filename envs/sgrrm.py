from __future__ import annotations
from typing import Dict, Hashable, List, Tuple
from .reward_machine import RewardMachine

class SGRRM:

    def __init__(self, base_env, reward_machines: Dict[str, RewardMachine], state_reward_fns: Dict[str, callable]=None):
        self.env = base_env
        self.rms = reward_machines
        self.state_reward_fns = state_reward_fns or {}
        self.agent_ids = base_env.agent_ids
        self.env_state = None
        self.u = {aid: rm.initial_state for aid, rm in self.rms.items()}

    def reset(self):
        self.env_state = self.env.reset()
        self.u = {aid: rm.initial_state for aid, rm in self.rms.items()}
        return self._obs()

    def _obs(self) -> Dict[str, Tuple]:
        return {aid: (self.env.observe(aid, self.env_state), self.u[aid]) for aid in self.agent_ids}

    def step(self, actions: Dict[str, Hashable]):
        next_env_state, env_done, info = self.env.step(actions)
        rewards, next_u, dones = ({}, {}, {})
        for aid, rm in self.rms.items():
            u_next, r = rm.step(self.u[aid], next_env_state)
            if aid in self.state_reward_fns:
                r += self.state_reward_fns[aid](next_env_state, actions[aid])
            rewards[aid] = r
            next_u[aid] = u_next
            dones[aid] = env_done or rm.is_terminal(u_next)
        self.env_state = next_env_state
        self.u = next_u
        return (self._obs(), rewards, dones, info)

    def counterfactual_targets(self, agent_id: str, next_env_state) -> Dict[Hashable, Tuple[Hashable, float]]:
        rm = self.rms[agent_id]
        return {u: rm.step(u, next_env_state) for u in rm.states}

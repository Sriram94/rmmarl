# waterworld_rm_env.py

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces
import random
from typing import Dict, List, Tuple, Any

# -----------------------------
# Helper / constants
# -----------------------------
DEFAULT_AGENT_RADIUS = 0.03
DEFAULT_FOOD_RADIUS = 0.06
DEFAULT_POISON_RADIUS = 0.045
DEFAULT_WORLD_SIZE = 1.0

# Discrete action grid (e.g. 8 directions + stay)
_DISCRETE_ACTIONS = [
    np.array([0.0, 0.0]),
    np.array([1.0, 0.0]),
    np.array([-1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([0.0, -1.0]),
    np.array([1.0, 1.0]),
    np.array([1.0, -1.0]),
    np.array([-1.0, 1.0]),
    np.array([-1.0, -1.0]),
]

# -----------------------------
# Waterworld environment (PettingZoo AEC)
# -----------------------------
class WaterworldRMEnv(AECEnv):
    metadata = {"render_modes": [], "name": "waterworld_rm_env"}

    def __init__(self,
                 n_agents: int = 3,
                 n_food: int = 5,
                 n_poison: int = 10,
                 world_size: float = DEFAULT_WORLD_SIZE,
                 agent_speed: float = 0.03,
                 sensor_range: float = 0.2,
                 radius_agent: float = DEFAULT_AGENT_RADIUS,
                 radius_food: float = DEFAULT_FOOD_RADIUS,
                 radius_poison: float = DEFAULT_POISON_RADIUS,
                 discrete_actions: List[np.ndarray] = None,
                 seed: int = None):
        super().__init__()
        self.n_agents = n_agents
        self.n_food = n_food
        self.n_poison = n_poison
        self.world_size = world_size
        self.agent_speed = agent_speed
        self.sensor_range = sensor_range
        self.radius_agent = radius_agent
        self.radius_food = radius_food
        self.radius_poison = radius_poison

        self.discrete_actions = discrete_actions or _DISCRETE_ACTIONS
        self.n_actions = len(self.discrete_actions)

        self.rng = np.random.RandomState(seed)

        # Agent positions and velocities
        self.agent_pos = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.agent_vel = np.zeros((self.n_agents, 2), dtype=np.float32)
        # Food & poison positions
        self.food_pos = np.zeros((self.n_food, 2), dtype=np.float32)
        self.poison_pos = np.zeros((self.n_poison, 2), dtype=np.float32)

        # Set up agent names in AEC
        self.agents = ["agent_" + str(i) for i in range(self.n_agents)]
        self.possible_agents = self.agents[:]

        # Observations / spaces
        # Each agent sees sensors: distances to food, poison, other agents within sensor_range
        # We'll build a fixed-size vector. Also a flag for collision.
        obs_dim = self._obs_vector_length()
        self.observation_spaces = {agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
                                   for agent in self.agents}
        self.action_spaces = {agent: spaces.Discrete(self.n_actions) for agent in self.agents}

        self._agent_selector = None
        self.current_agent = None
        self._cumulative_rewards = None
        self._dones = None
        self._terminated = None

    def _obs_vector_length(self) -> int:
        return 2 * (self.n_food + self.n_poison + (self.n_agents - 1)) + 2

    def reset(self, seed: int = None, return_info: bool = False, **kwargs):
        if seed is not None:
            self.rng.seed(seed)
        # reset positions
        self.agent_pos = self.rng.rand(self.n_agents, 2).astype(np.float32) * self.world_size
        self.agent_vel = np.zeros_like(self.agent_pos)
        self.food_pos = self.rng.rand(self.n_food, 2).astype(np.float32) * self.world_size
        self.poison_pos = self.rng.rand(self.n_poison, 2).astype(np.float32) * self.world_size

        self._agent_selector = self._agent_iterator()
        self.current_agent = next(self._agent_selector)
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self._dones = {agent: False for agent in self.agents}
        self._terminated = {agent: False for agent in self.agents}

        obs = {agent: self._make_obs(i) for i, agent in enumerate(self.agents)}
        if return_info:
            return obs, {}
        return obs

    def _agent_iterator(self):
        # simple round robin
        while True:
            for ag in self.agents:
                yield ag

    def observe(self, agent: str) -> np.ndarray:
        idx = self.agents.index(agent)
        return self._make_obs(idx)

    def step(self, action):
        # Store the action temporarily
        if not hasattr(self, "_action_buffer"):
            self._action_buffer = {}

        # record
        self._action_buffer[self.current_agent] = action

        # advance current_agent
        self.current_agent = next(self._agent_selector)

        # If we've collected actions from all agents, simulate environment step
        if len(self._action_buffer) == len(self.agents):
            # build full action array
            act_arr = []
            for i, ag in enumerate(self.agents):
                act_idx = self._action_buffer[ag]
                act_vec = self.discrete_actions[act_idx]
                act_arr.append(act_vec)
            act_arr = np.stack(act_arr, axis=0)  # shape (n_agents, 2)

            # perform environment dynamics
            obs_all, rewards_all, dones_all, _info = self._joint_step(act_arr)

            # distribute observations / rewards
            for i, ag in enumerate(self.agents):
                self._cumulative_rewards[ag] = rewards_all[i]
                self._dones[ag] = dones_all
            # clear buffer
            self._action_buffer = {}

        # report for current agent
        agent = self.current_agent
        reward = self._cumulative_rewards.get(agent, 0.0)
        done = self._dones.get(agent, False)
        termination = self._terminated.get(agent, False)

        info = {}
        return None, reward, termination, done, info

    def _joint_step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray],
                                                     np.ndarray, bool, Dict]:
        """
        Simulate one time step for all agents simultaneously.
        :param actions: shape (n_agents, 2) thrust vectors (discrete-> continuous vector)
        :returns: observations dict, rewards array, done flag, info dict
        """
        # normalize / limit speed
        norms = np.linalg.norm(actions, axis=1, keepdims=True)
        scale = np.where(norms > self.agent_speed, self.agent_speed / (norms + 1e-8), 1.0)
        actions = actions * scale

        # update velocity / position
        self.agent_vel = actions
        self.agent_pos += actions
        # boundary clip
        self.agent_pos = np.clip(self.agent_pos, 0.0, self.world_size)

        # compute rewards
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        dones = False

        # detect collisions with food/poison
        for i in range(self.n_agents):
            pi = self.agent_pos[i]
            # food
            for fi, fp in enumerate(self.food_pos):
                if np.linalg.norm(pi - fp) <= (self.radius_agent + self.radius_food):
                    rewards[i] += 1.0
                    # respawn food
                    self.food_pos[fi] = self.rng.rand(2).astype(np.float32) * self.world_size
            # poison
            for pj, pp in enumerate(self.poison_pos):
                if np.linalg.norm(pi - pp) <= (self.radius_agent + self.radius_poison):
                    rewards[i] -= 1.0
                    # respawn poison
                    self.poison_pos[pj] = self.rng.rand(2).astype(np.float32) * self.world_size

        # observations
        obs = {ag: self._make_obs(i) for i, ag in enumerate(self.agents)}

        return obs, rewards, dones, {}

    def _make_obs(self, agent_idx: int) -> np.ndarray:
        # Build vector: relative positions of food, poison, other agents
        parts = []
        mypos = self.agent_pos[agent_idx]
        # food
        for fp in self.food_pos:
            dv = fp - mypos
            # normalize by sensor range
            if np.linalg.norm(dv) <= self.sensor_range:
                parts.append(dv / (self.sensor_range + 1e-8))
            else:
                parts.append(np.zeros(2))
        # poison
        for pp in self.poison_pos:
            dv = pp - mypos
            if np.linalg.norm(dv) <= self.sensor_range:
                parts.append(dv / (self.sensor_range + 1e-8))
            else:
                parts.append(np.zeros(2))
        # other agents
        for j in range(self.n_agents):
            if j == agent_idx:
                continue
            dv = self.agent_pos[j] - mypos
            if np.linalg.norm(dv) <= self.sensor_range:
                parts.append(dv / (self.sensor_range + 1e-8))
            else:
                parts.append(np.zeros(2))
        flat = np.concatenate(parts, axis=0).astype(np.float32)
        # flags: collided with food? collided with poison? (for last step)
        # As a simple placeholder, we just put 0,0
        flags = np.array([0.0, 0.0], dtype=np.float32)
        obs_vec = np.concatenate([flat, flags], axis=0)
        return obs_vec

    def close(self):
        pass

    def agent_selection(self):
        return self.current_agent

    def num_agents(self):
        return len(self.agents)

    def is_done(self):
        # if all agents terminated
        return all(self._terminated.values())

# wrap default AEC env
def env():
    env = WaterworldRMEnv()
    # optional wrappers (e.g. to enforce agent iterable, etc.)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

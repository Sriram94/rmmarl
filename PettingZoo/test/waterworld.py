"""
Agents:
  Pursuers: DCROM agents (RM-driven)
  Poisons:  DQN agents (env reward driven)
"""

import numpy as np, random, sys
from typing import Dict, List, Tuple, Any
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import random
from gymnasium import spaces

# Ensure access to agent files
sys.path.append("/mnt/data")
try:
    from DCROM import DCROMAgent
except Exception as e:
    DCROMAgent = None
    print("Warning: could not import DCROMAgent from dcrom.py:", e)
try:
    from DQN import DQNCrossProductAgent
except Exception as e:
    DQNCrossProductAgent = None
    print("Warning: could not import DQNCrossProductAgent:", e)

# ----------------------------------------------------------
# Basic RM classes
# ----------------------------------------------------------
class SimpleSequenceRM:
    """A reward machine for a simple ordered color sequence."""
    def __init__(self, colors: List[str]):
        self.colors = colors
        self.n_states = len(colors) + 1
        self.terminal_states = {self.n_states - 1}
    def next_state(self, u: int, events: List[str]):
        if u in self.terminal_states:
            return u
        next_color = self.colors[u]
        if next_color in events:
            return u + 1
        return u
    def reward(self, u: int, u_next: int):
        return 1.0 if (u_next in self.terminal_states and u_next != u) else 0.0
    def is_terminal(self, u: int):
        return u in self.terminal_states

class ConjunctiveRM:
    """A conjunction (AND) of multiple RMs."""
    def __init__(self, rms: List[SimpleSequenceRM]):
        self.rms = rms
        self.n_states = np.prod([rm.n_states for rm in rms])
        self.terminal_states = None
    def unpack_state(self, u: int):
        """Decode combined index to component states."""
        s = []
        base = 1
        for rm in self.rms:
            idx = (u // base) % rm.n_states
            s.append(idx)
            base *= rm.n_states
        return s
    def pack_state(self, states: List[int]):
        base, val = 1, 0
        for i, rm in enumerate(self.rms):
            val += states[i] * base
            base *= rm.n_states
        return val
    def next_state(self, u: int, events: List[str]):
        subs = self.unpack_state(u)
        next_subs = [rm.next_state(subs[i], events) for i, rm in enumerate(self.rms)]
        return self.pack_state(next_subs)
    def reward(self, u: int, u_next: int):
        subs, next_subs = self.unpack_state(u), self.unpack_state(u_next)
        return 1.0 if all(self.rms[i].is_terminal(next_subs[i]) for i in range(len(self.rms))) and not all(self.rms[i].is_terminal(subs[i]) for i in range(len(self.rms))) else 0.0

# ----------------------------------------------------------
# Waterworld Environment
# ----------------------------------------------------------
COLOR_NAMES = ["red", "green", "blue", "cyan", "magenta", "yellow"]
_DISCRETE_ACTIONS = [
    np.array([0.0, 0.0]), np.array([1,0]), np.array([-1,0]),
    np.array([0,1]), np.array([0,-1]), np.array([1,1]),
    np.array([1,-1]), np.array([-1,1]), np.array([-1,-1])
]

class WaterworldColorRMCoop(AECEnv):
    metadata = {"render_modes": [], "name": "waterworld_color_rm_multi_coop"}

    def __init__(self, n_pursuers=3, n_poison=2, n_food_each=1, seed=None):
        super().__init__()
        self.n_pursuers, self.n_poison = n_pursuers, n_poison
        self.n_agents = n_pursuers + n_poison
        self.world_size = 1.0
        self.speed = 0.03
        self.sensor_range = 0.2
        self.radius_agent = 0.03
        self.radius_food = 0.06
        self.radius_poison = 0.045
        self.discrete_actions = _DISCRETE_ACTIONS
        self.n_actions = len(self.discrete_actions)
        self.pursuer_names = [f"pursuer_{i}" for i in range(n_pursuers)]
        self.poison_names = [f"poison_{i}" for i in range(n_poison)]
        self.agents = self.pursuer_names + self.poison_names
        self.possible_agents = list(self.agents)
        self.rng = np.random.RandomState(seed)

        self.food_items = [{'color': c, 'pos': self.rng.rand(2)} for c in COLOR_NAMES]
        obs_dim = 2 * (len(self.food_items) + (self.n_agents - 1)) + 2
        self.observation_spaces = {a: spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32) for a in self.agents}
        self.action_spaces = {a: spaces.Discrete(self.n_actions) for a in self.agents}

        self.agent_pos = np.zeros((self.n_agents, 2), np.float32)
        self.agent_vel = np.zeros((self.n_agents, 2), np.float32)
        self._action_buffer = {}

        self.shared_rms = self.build_rms()
        self.rm_states = [0 for _ in self.shared_rms]
        self.last_event = {a: None for a in self.pursuer_names}

    def build_rms(self):
        """Define all six RMs."""
        rm0 = SimpleSequenceRM(["red", "green"])
        rm1 = SimpleSequenceRM(["blue", "cyan"])
        rm2 = SimpleSequenceRM(["magenta", "yellow"])
        rm3 = ConjunctiveRM([rm0, rm1])
        rm4 = ConjunctiveRM([rm0, rm1, rm2])
        rm5 = ConjunctiveRM([SimpleSequenceRM(["red","green","blue"]), SimpleSequenceRM(["cyan","magenta","yellow"])])
        return [rm0, rm1, rm2, rm3, rm4, rm5]

    def reset(self, seed=None, return_info=False, **kwargs):
        if seed is not None:
            self.rng.seed(seed)
        self.agent_pos = self.rng.rand(self.n_agents, 2)
        for f in self.food_items:
            f['pos'] = self.rng.rand(2)
        self._action_buffer.clear()
        self.rm_states = [0 for _ in self.shared_rms]
        self.last_event = {a: None for a in self.pursuer_names}
        self._agent_selector = self._agent_iterator()
        self.current_agent = next(self._agent_selector)
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        obs = {a: self._make_obs(i) for i,a in enumerate(self.agents)}
        return obs

    def _agent_iterator(self):
        while True:
            for a in self.agents:
                yield a

    def step(self, action):
        self._action_buffer[self.current_agent] = action
        self.current_agent = next(self._agent_selector)
        if len(self._action_buffer) == len(self.agents):
            act_arr = np.stack([self.discrete_actions[self._action_buffer[a]] for a in self.agents])
            self._joint_step(act_arr)
            self._action_buffer.clear()
        reward = self._cumulative_rewards[self.current_agent]
        return None, reward, False, False, {}

    def _joint_step(self, acts):
        # move agents
        norms = np.linalg.norm(acts, axis=1, keepdims=True)+1e-8
        acts = acts / norms * self.speed
        self.agent_pos = np.clip(self.agent_pos + acts, 0, self.world_size)

        self.last_event = {a: None for a in self.pursuer_names}
        team_events = []

        # Pursuer-food collisions
        for i,p in enumerate(self.pursuer_names):
            pos = self.agent_pos[i]
            for food in self.food_items:
                if np.linalg.norm(pos - food['pos']) < (self.radius_agent + self.radius_food):
                    self.last_event[p] = food['color']
                    team_events.append(food['color'])
                    food['pos'] = self.rng.rand(2)

        # Poison-pursuer collisions (with coupled rewards)
        for j,pz in enumerate(self.poison_names):
            pz_idx = self.n_pursuers + j
            pz_pos = self.agent_pos[pz_idx]
            for i,p in enumerate(self.pursuer_names):
                if np.linalg.norm(pz_pos - self.agent_pos[i]) < (self.radius_agent*2):
                    self._cumulative_rewards[pz] += 1.0
                    self._cumulative_rewards[p] -= 1.0

        # Joint event triggers (if >=2 pursuers got events)
        if len(team_events) >= 2:
            for i,rm in enumerate(self.shared_rms):
                u_prev = self.rm_states[i]
                u_next = rm.next_state(u_prev, team_events)
                r = rm.reward(u_prev, u_next)
                self.rm_states[i] = u_next
                if r > 0:
                    # reward shared equally among pursuers
                    share = r / len(self.pursuer_names)
                    for p in self.pursuer_names:
                        self._cumulative_rewards[p] += share

        # refresh observations
        return {a:self._make_obs(i) for i,a in enumerate(self.agents)}

    def _make_obs(self, idx):
        parts = []
        mypos = self.agent_pos[idx]
        for f in self.food_items:
            dv = f['pos'] - mypos
            parts.append(np.clip(dv/self.sensor_range, -1, 1))
        for j in range(self.n_agents):
            if j==idx: continue
            dv = self.agent_pos[j]-mypos
            parts.append(np.clip(dv/self.sensor_range, -1, 1))
        obs = np.concatenate(parts+[np.zeros(2)],0).astype(np.float32)
        return obs

# Wrapper
def env():
    e = WaterworldColorRMCoop()
    return wrappers.OrderEnforcingWrapper(e)

# Training hook placeholder (needs to be modified for each head-to-head comparison with DQN)

def train_demo(n_episodes=3, max_steps=100):
    e = env()
    obs_dim = e.obs()
    n_actions = e.n_actions()
    n_op_actions = e.n_actions()
    prev_op_action_dim = n_op_actions
    e.build_rms()
    prev_opp_action = [0,0,0,0]
    opp_action = [0,0,0,0]

    obs = e.reset()
    print("Env reset with agents:", e.agents)
    for ep in range(n_episodes):
        e.reset()
        random_number = random.randint(0, 5)
	if random_number == 0: 
            rm = e.rm0
	elif random_number == 1: 
            rm = e.rm1
	elif random_number == 2: 
            rm = e.rm2
	elif random_number == 3: 
            rm = e.rm3
	elif random_number == 4: 
            rm = e.rm4
	else: 
            rm = e.rm5

    	agent1 = DCROMAgent(obs_dim, n_actions, rm, n_op_actions, prev_op_action_dim)
    	agent2 = DQNCrossProductAgent(obs_dim, n_actions, rm)
    	s = 0
		
      
        for t in range(max_steps):
            for a in e.agents:
                if e.agents == "pursuer":
                    action_new = DCROMAgent.select_action(e.s, e.u, prev_op_action)
                    opp_action[s] = action_new
                    s = s + 1
                    

                else: 
                    aciton_new = DQNCrossProductAgent.select_action(e.s, e.u)

                    opp_action[s] = action_new
                    s = s + 1



                DCROMAgent.store_transition(e.s, e.u, e.a, e.r, e.s_next, e.u_next, e.done, prev_opp_action, opp_action)
                DQNCrossProductAgent.store_transition(e.s, e.u, e.a, e.r, e.s_next, e.u_next, e.done)
		prev_opp_action = opp_action

        print(f"Episode {ep} done, sample rewards:", e._cumulative_rewards)
        DCROMAgent.train_step()
        DQNCrossProductAgent.train_step()
    print("Demo run complete.")

if __name__ == "__main__":
    train_demo()

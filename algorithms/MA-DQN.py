
"""ma_dqn_crossprod.py

MA-DQN on Cross-Product State Space


Components:
- RewardMachine: small RM API
- QNetwork: consumes obs, rm_onehot, prev_op_onehot as separate inputs (but internally concatenates them)
- ReplayBuffer
- MADQNCrossProductAgent: DQN agent using previous opponent action as input component

"""

import random
from typing import Callable, Any, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Reward Machine (simple API)
# -----------------------------
class RewardMachine:
    def __init__(self, n_states:int, delta:Dict[Tuple[int, Any], int], sigma:Dict[Tuple[int,int], float], label_fn:Callable[[Any], Any], terminal_states:List[int]=None):
        self.n_states = n_states
        self.delta = delta
        self.sigma = sigma
        self.label_fn = label_fn
        self.terminal_states = set(terminal_states or [])

    def next_state(self, u:int, env_state:Any):
        event = self.label_fn(env_state)
        return self.delta.get((u, event), u)

    def reward(self, u:int, u_next:int) -> float:
        return self.sigma.get((u, u_next), 0.0)

    def is_terminal(self, u:int) -> bool:
        return u in self.terminal_states

# -----------------------------
# Q-network
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim:int, rm_state_dim:int, prev_op_dim:int, n_actions:int, hidden_sizes=[256,256]):
        super().__init__()
        # We'll treat each input as its own component; internally we'll concat them before MLP
        in_dim = obs_dim + rm_state_dim + prev_op_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs:torch.Tensor, rm_onehot:torch.Tensor, prev_op_onehot:torch.Tensor):
        # obs: [B, obs_dim], rm_onehot: [B, rm_state_dim], prev_op_onehot: [B, prev_op_dim]
        x = torch.cat([obs, rm_onehot, prev_op_onehot], dim=-1)
        return self.net(x)

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition:Tuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size:int):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# MA-DQN Agent (Cross-Product + Prev Opp Action)
# -----------------------------
class MADQNCrossProductAgent:
    def __init__(self,
                 obs_dim:int,
                 n_actions:int,
                 rm:RewardMachine,
                 n_prev_op_actions:int,
                 lr:float=1e-3,
                 gamma:float=0.99,
                 buffer_size:int=100000,
                 batch_size:int=64,
                 target_update_freq:int=1000,
                 device:torch.device=device):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rm = rm
        self.n_rm_states = rm.n_states if rm is not None else 1
        self.n_prev_op_actions = n_prev_op_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        # Q-network and target: takes obs, rm_onehot, prev_op_onehot as separate inputs
        self.q_net = QNetwork(obs_dim, self.n_rm_states, self.n_prev_op_actions, n_actions).to(device)
        self.q_target = QNetwork(obs_dim, self.n_rm_states, self.n_prev_op_actions, n_actions).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.update_count = 0
        self.target_update_freq = target_update_freq

    def one_hot(self, indices, depth:int):
        arr = np.zeros((len(indices), depth), dtype=np.float32)
        for i, idx in enumerate(indices):
            arr[i, int(idx)] = 1.0
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def select_action(self, obs:np.ndarray, rm_state:int, prev_op_action_idx:int, eps:float=0.0):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        rm_oh = torch.zeros((1, self.n_rm_states), dtype=torch.float32, device=self.device)
        rm_oh[0, rm_state] = 1.0
        prev_op_oh = torch.zeros((1, self.n_prev_op_actions), dtype=torch.float32, device=self.device)
        prev_op_oh[0, prev_op_action_idx] = 1.0
        with torch.no_grad():
            qvals = self.q_net(obs_t, rm_oh, prev_op_oh)
            if random.random() < eps:
                return random.randrange(self.n_actions)
            else:
                return int(torch.argmax(qvals, dim=-1).item())

    def store_transition(self, s, u, a, r, s_next, u_next, done, prev_op_action_idx, prev_op_action_next_idx):
        # store: s, u, a, r, s_next, u_next, done, prev_op_action_idx, prev_op_action_next_idx
        # prev_op_action_idx: previous opponent joint action at time t (used as input at time t)
        # prev_op_action_next_idx: previous opponent joint action at time t+1 (used as input at next state)
        self.replay.push((s, u, a, r, s_next, u_next, float(done), prev_op_action_idx, prev_op_action_next_idx))

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None
        s_batch, u_batch, a_batch, r_batch, s_next_batch, u_next_batch, done_batch, prev_op_batch, prev_op_next_batch = self.replay.sample(self.batch_size)
        s_batch = torch.tensor(np.vstack(s_batch), dtype=torch.float32, device=self.device)
        s_next_batch = torch.tensor(np.vstack(s_next_batch), dtype=torch.float32, device=self.device)
        u_batch = torch.tensor(u_batch, dtype=torch.long, device=self.device)
        u_next_batch = torch.tensor(u_next_batch, dtype=torch.long, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.long, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)
        prev_op_batch = torch.tensor(prev_op_batch, dtype=torch.long, device=self.device)
        prev_op_next_batch = torch.tensor(prev_op_next_batch, dtype=torch.long, device=self.device)

        # one-hot encodings for rm states and prev op actions
        rm_onehot = F.one_hot(u_batch, num_classes=self.n_rm_states).float()
        rm_next_onehot = F.one_hot(u_next_batch, num_classes=self.n_rm_states).float()
        prev_op_onehot = F.one_hot(prev_op_batch, num_classes=self.n_prev_op_actions).float()
        prev_op_next_onehot = F.one_hot(prev_op_next_batch, num_classes=self.n_prev_op_actions).float()

        # compute target: r + gamma * max_a' Q_target(s_next, u_next, prev_op_next)
        with torch.no_grad():
            q_next = self.q_target(s_next_batch, rm_next_onehot, prev_op_next_onehot)
            max_q_next, _ = q_next.max(dim=1)
            target_q = r_batch + (1.0 - done_batch) * (self.gamma * max_q_next)

        # current q for taken actions using current network on (s, u, prev_op)
        q_vals = self.q_net(s_batch, rm_onehot, prev_op_onehot)
        q_taken = q_vals.gather(1, a_batch.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_taken, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return {'q_loss': loss.item()}

# -----------------------------
# Dummy label and RM for testing
# -----------------------------
def _dummy_label_fn(env_state):
    s = np.array(env_state)
    ssum = float(s.sum())
    if ssum < 0.0:
        return 0
    elif ssum < 1.0:
        return 1
    else:
        return 2

def build_dummy_rm():
    n_states = 3
    delta = {
        (0,0): 0, (0,1): 1, (0,2): 2,
        (1,0): 1, (1,1): 1, (1,2): 2,
        (2,0): 2, (2,1): 2, (2,2): 2,
    }
    sigma = {(1,2): 1.0}
    return RewardMachine(n_states, delta, sigma, _dummy_label_fn, terminal_states=[2])

def dummy_env_step(action):
    obs = np.random.randn(4).astype(np.float32)
    done = random.random() < 0.05
    return obs, done

if __name__ == '__main__':
    import torch.nn.functional as F
    obs_dim = 4
    n_actions = 5
    n_prev_op_actions = 6  # size of previous opponent action encoding
    rm = build_dummy_rm()
    agent = MADQNCrossProductAgent(obs_dim, n_actions, rm, n_prev_op_actions,
                                   lr=1e-3, gamma=0.99, buffer_size=2000, batch_size=32, target_update_freq=50)

    # populate replay buffer with random transitions (including prev-op values)
    for _ in range(600):
        s = np.random.randn(obs_dim).astype(np.float32)
        u = random.randrange(rm.n_states)
        a = random.randrange(n_actions)
        s_next, done = dummy_env_step(a)
        u_next = rm.next_state(u, s_next)
        r = rm.reward(u, u_next)
        prev_op = random.randrange(n_prev_op_actions)
        prev_op_next = random.randrange(n_prev_op_actions)
        agent.store_transition(s, u, a, r, s_next, u_next, done, prev_op, prev_op_next)

    # run some training steps
    for step in range(300):
        metrics = agent.train_step()
        if metrics and step % 25 == 0:
            print(f"Step {step}: q_loss={metrics['q_loss']:.4f}")

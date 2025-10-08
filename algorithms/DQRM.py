
"""dqrm.py

Deep QRM (DQRM) implementation.

Key components:
- RewardMachine: small RM API
- QNetwork: maps observation -> Q-values (per RM state networks)
- ReplayBuffer
- DQRMAgent: performs counterfactual updates across RM states

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
    def __init__(self, n_states: int, delta: Dict[Tuple[int, Any], int],
                 sigma: Dict[Tuple[int, int], float],
                 label_fn: Callable[[Any], Any],
                 terminal_states: List[int] = None):
        self.n_states = n_states
        self.delta = delta
        self.sigma = sigma
        self.label_fn = label_fn
        self.terminal_states = set(terminal_states or [])

    def next_state(self, u: int, env_state: Any):
        event = self.label_fn(env_state)
        return self.delta.get((u, event), u)

    def reward(self, u: int, u_next: int) -> float:
        return self.sigma.get((u, u_next), 0.0)

    def is_terminal(self, u: int) -> bool:
        return u in self.terminal_states

# -----------------------------
# Q-network (per RM state)
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=[128,128]):
        super().__init__()
        in_dim = obs_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor):
        return self.net(obs)

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition: Tuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# DQRM Agent
# -----------------------------
class DQRMAgent:
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 rm: RewardMachine,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 device: torch.device = device):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rm = rm
        self.n_rm_states = rm.n_states
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        # per-RM-state Q-networks and targets
        self.q_nets = nn.ModuleList([QNetwork(obs_dim, n_actions).to(device) for _ in range(self.n_rm_states)])
        self.q_targets = nn.ModuleList([QNetwork(obs_dim, n_actions).to(device) for _ in range(self.n_rm_states)])
        for tgt, src in zip(self.q_targets, self.q_nets):
            tgt.load_state_dict(src.state_dict())

        # optimizer over all Q parameters
        q_params = []
        for q in self.q_nets:
            q_params += list(q.parameters())
        self.q_optimizer = torch.optim.Adam(q_params, lr=lr)

        # replay
        self.replay = ReplayBuffer(buffer_size)

        self.update_count = 0
        self.target_update_freq = target_update_freq

    def select_action(self, obs: np.ndarray, rm_state: int, eps: float = 0.0):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q_nets[rm_state](obs_t)  # select from network corresponding to current RM state
            if random.random() < eps:
                return random.randrange(self.n_actions)
            else:
                return int(torch.argmax(qvals, dim=-1).item())

    def store_transition(self, s, u, a, r, s_next, u_next, done):
        # store: s, u, a, r, s_next, u_next, done
        self.replay.push((s, u, a, r, s_next, u_next, float(done)))

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None
        s_batch, u_batch, a_batch, r_batch, s_next_batch, u_next_batch, done_batch = self.replay.sample(self.batch_size)

        # convert to tensors
        s_batch = torch.tensor(np.vstack(s_batch), dtype=torch.float32, device=self.device)
        s_next_batch = torch.tensor(np.vstack(s_next_batch), dtype=torch.float32, device=self.device)
        u_batch = torch.tensor(u_batch, dtype=torch.long, device=self.device)
        u_next_batch = torch.tensor(u_next_batch, dtype=torch.long, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.long, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)

        batch_size = s_batch.shape[0]
        n_rm = self.n_rm_states

        # Compute counterfactual targets for each possible RM state u_cf
        # targets[batch, u_cf] = r_cf + gamma * max_a' Q_target(s_next, u_cf_next)
        targets = torch.zeros((batch_size, n_rm), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for u_cf in range(n_rm):
                u_cf_next_list = []
                r_cf_list = []
                for i in range(batch_size):
                    s_next_np = s_next_batch[i].cpu().numpy()
                    u_cf_next = self.rm.next_state(u_cf, s_next_np)
                    r_cf = self.rm.reward(u_cf, u_cf_next)
                    u_cf_next_list.append(u_cf_next)
                    r_cf_list.append(r_cf)
                u_cf_next_arr = torch.tensor(u_cf_next_list, dtype=torch.long, device=self.device)
                r_cf_tensor = torch.tensor(r_cf_list, dtype=torch.float32, device=self.device)

                # compute q_target(s_next, u_cf_next)
                q_next_vals = torch.zeros((batch_size, self.n_actions), dtype=torch.float32, device=self.device)
                for i in range(batch_size):
                    u_next_i = int(u_cf_next_arr[i].item())
                    q_vals_next = self.q_targets[u_next_i](s_next_batch[i:i+1])  # [1, n_actions]
                    q_next_vals[i] = q_vals_next
                max_q_next, _ = q_next_vals.max(dim=1)
                target_cf = r_cf_tensor + (1.0 - done_batch) * (self.gamma * max_q_next)
                targets[:, u_cf] = target_cf

        # Now compute losses: for each u_cf, compute q_net[u_cf](s_batch), gather taken actions, and compare to targets[:, u_cf]
        q_losses = []
        for u_cf in range(n_rm):
            q_net = self.q_nets[u_cf]
            q_vals = q_net(s_batch)  # [B, n_actions]
            q_taken = q_vals.gather(1, a_batch.unsqueeze(1)).squeeze(1)
            target_for_u = targets[:, u_cf].detach()
            loss_u = F.mse_loss(q_taken, target_for_u)
            q_losses.append(loss_u)

        loss_q = torch.stack(q_losses).mean()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_([p for q in self.q_nets for p in q.parameters()], 10.0)
        self.q_optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            for tgt, src in zip(self.q_targets, self.q_nets):
                tgt.load_state_dict(src.state_dict())

        return {'q_loss': loss_q.item()}

# -----------------------------
# Dummy label and RM build for testing
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
    obs_dim = 4
    n_actions = 5
    rm = build_dummy_rm()
    agent = DQRMAgent(obs_dim, n_actions, rm,
                       lr=1e-3, gamma=0.99, buffer_size=2000, batch_size=32, target_update_freq=50)

    # populate replay buffer with random transitions
    for _ in range(600):
        s = np.random.randn(obs_dim).astype(np.float32)
        u = random.randrange(rm.n_states)
        a = random.randrange(n_actions)
        s_next, done = dummy_env_step(a)
        u_next = rm.next_state(u, s_next)
        r = rm.reward(u, u_next)
        agent.store_transition(s, u, a, r, s_next, u_next, done)

    # run some training steps
    for step in range(300):
        metrics = agent.train_step()
        if metrics and step % 25 == 0:
            print(f"Step {step}: q_loss={metrics['q_loss']:.4f}")

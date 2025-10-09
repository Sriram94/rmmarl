import random
from typing import Callable, Any, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class QNetwork(nn.Module):
    def __init__(self, obs_dim:int, rm_state_dim:int, n_actions:int):
        super().__init__()
        in_dim = obs_dim + rm_state_dim
        hidden_size = 1024
        layers = []
        for _ in range(6):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs:torch.Tensor, rm_onehot:torch.Tensor):
        x = torch.cat([obs, rm_onehot], dim=-1)
        return self.net(x)

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

class DQNCrossProductAgent:
    def __init__(self,
                 obs_dim:int,
                 n_actions:int,
                 rm:RewardMachine,
                 lr:float=0.01,
                 gamma:float=0.9,
                 buffer_size:int=int(2e7),
                 batch_size:int=64,
                 target_update_freq:int=100,
                 fixed_eps:float=0.1,
                 device:torch.device=device):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rm = rm
        self.n_rm_states = rm.n_states if rm is not None else 1
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.fixed_eps = fixed_eps

        # Q-network and target
        self.q_net = QNetwork(obs_dim, self.n_rm_states, n_actions).to(device)
        self.q_target = QNetwork(obs_dim, self.n_rm_states, n_actions).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.update_count = 0
        self.target_update_freq = target_update_freq

    def select_action(self, obs:np.ndarray, rm_state:int):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        rm_oh = torch.zeros((1, self.n_rm_states), dtype=torch.float32, device=self.device)
        rm_oh[0, rm_state] = 1.0
        with torch.no_grad():
            qvals = self.q_net(obs_t, rm_oh)
            if random.random() < self.fixed_eps:
                return random.randrange(self.n_actions)
            else:
                return int(torch.argmax(qvals, dim=-1).item())

    def store_transition(self, s, u, a, r, s_next, u_next, done):
        self.replay.push((s, u, a, r, s_next, u_next, float(done)))

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None
        s_batch, u_batch, a_batch, r_batch, s_next_batch, u_next_batch, done_batch = self.replay.sample(self.batch_size)
        s_batch = torch.tensor(np.vstack(s_batch), dtype=torch.float32, device=self.device)
        s_next_batch = torch.tensor(np.vstack(s_next_batch), dtype=torch.float32, device=self.device)
        u_batch = torch.tensor(u_batch, dtype=torch.long, device=self.device)
        u_next_batch = torch.tensor(u_next_batch, dtype=torch.long, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.long, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)

        rm_onehot = F.one_hot(u_batch, num_classes=self.n_rm_states).float()
        rm_next_onehot = F.one_hot(u_next_batch, num_classes=self.n_rm_states).float()

        # Compute targets
        with torch.no_grad():
            q_next = self.q_target(s_next_batch, rm_next_onehot)
            max_q_next, _ = q_next.max(dim=1)
            target_q = r_batch + (1.0 - done_batch) * (self.gamma * max_q_next)

        q_vals = self.q_net(s_batch, rm_onehot)
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
    agent = DQNCrossProductAgent(obs_dim, n_actions, rm)

    for _ in range(600):
        s = np.random.randn(obs_dim).astype(np.float32)
        u = random.randrange(rm.n_states)
        a = random.randrange(n_actions)
        s_next, done = dummy_env_step(a)
        u_next = rm.next_state(u, s_next)
        r = rm.reward(u, u_next)
        agent.store_transition(s, u, a, r, s_next, u_next, done)

    for step in range(300):
        metrics = agent.train_step()
        if metrics and step % 25 == 0:
            print(f"Step {step}: q_loss={metrics['q_loss']:.4f}")

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class QNet(nn.Module):

    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    variant = 'DQN'

    def __init__(self, obs_dim, n_actions, device='cpu', lr=0.001, gamma=0.9, buffer_size=200000, target_update_every=100, seed=0):
        self.obs_dim, self.n_actions, self.gamma = (obs_dim, n_actions, gamma)
        self.device = device
        self.q_eval = QNet(obs_dim, n_actions).to(device)
        self.q_target = QNet(obs_dim, n_actions).to(device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.optim = torch.optim.Adam(self.q_eval.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.target_update_every = target_update_every
        self._steps = 0

    def act(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_eval(s).argmax(dim=1).item())

    def store(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def train(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None
        batch = self.pyrng.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        s = torch.tensor(np.stack(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s_next = torch.tensor(np.stack(s_next), dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        q_sa = self.q_eval(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s_next).max(dim=1).values
            y = r + self.gamma * (1 - done) * q_next
        loss = F.mse_loss(q_sa, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._steps += 1
        if self._steps % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
        return loss.item()

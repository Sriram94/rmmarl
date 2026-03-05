import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(s).to(device),
            torch.FloatTensor(a).to(device),
            torch.FloatTensor(r).unsqueeze(1).to(device),
            torch.FloatTensor(ns).to(device),
            torch.FloatTensor(d).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)



class Actor(nn.Module):
    """
    7-layer network
    6 x 1024 ReLU layers
    """

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        layers = []
        input_dim = state_dim

        for _ in range(6):
            layers.append(nn.Linear(input_dim, 1024))
            layers.append(nn.ReLU())
            input_dim = 1024

        layers.append(nn.Linear(1024, action_dim))
        self.net = nn.Sequential(*layers)

        self.max_action = max_action

    def forward(self, x):
        return self.max_action * torch.tanh(self.net(x))


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        layers = []
        input_dim = state_dim + action_dim

        for _ in range(6):
            layers.append(nn.Linear(input_dim, 1024))
            layers.append(nn.ReLU())
            input_dim = 1024

        layers.append(nn.Linear(1024, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))


class Manager(nn.Module):
    """
    High-level option selector
    2 layers with 256 units
    """

    def __init__(self, state_dim, num_options):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_options)
        )

    def forward(self, x):
        return self.net(x)



class OptionPolicy:

    def __init__(self, state_dim, action_dim, max_action, lr):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.total_updates = 0


class DHRLContinuousAgent:

    def __init__(self, state_dim, action_dim, max_action, num_options):

        self.gamma = 0.9
        self.batch_size = 64
        self.epsilon = 0.9
        self.target_update_freq = 100

        self.num_options = num_options
        self.max_action = max_action

        self.manager = Manager(state_dim, num_options).to(device)
        self.manager_target = Manager(state_dim, num_options).to(device)

        self.manager_target.load_state_dict(self.manager.state_dict())

        self.manager_opt = optim.Adam(self.manager.parameters(), lr=0.01)

        self.options = [
            OptionPolicy(state_dim, action_dim, max_action, 0.01)
            for _ in range(num_options)
        ]

        self.low_replay = ReplayBuffer(int(2e7))
        self.high_replay = ReplayBuffer(int(2e7))

        self.learn_steps = 0


    def select_option(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.num_options - 1)

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q = self.manager(state)
        return q.argmax().item()


    def select_action(self, state, option):

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.options[option].actor(state)
        return action.cpu().detach().numpy()[0]


    def store_low(self, s, a, r, ns, d):
        self.low_replay.push(s, a, r, ns, d)

    def store_high(self, s, o, r, ns, d):
        self.high_replay.push(s, o, r, ns, d)


    def train(self):

        if len(self.low_replay) < self.batch_size:
            return

        self.learn_steps += 1

        for opt in self.options:

            s, a, r, ns, d = self.low_replay.sample(self.batch_size)

            with torch.no_grad():

                next_a = opt.actor_target(ns)
                target_q = opt.critic_target(ns, next_a)
                y = r + (1 - d) * self.gamma * target_q

            q = opt.critic(s, a)
            critic_loss = nn.MSELoss()(q, y)

            opt.critic_opt.zero_grad()
            critic_loss.backward()
            opt.critic_opt.step()

            actor_loss = -opt.critic(s, opt.actor(s)).mean()

            opt.actor_opt.zero_grad()
            actor_loss.backward()
            opt.actor_opt.step()


        if len(self.high_replay) >= self.batch_size:

            s, o, r, ns, d = self.high_replay.sample(self.batch_size)

            o = o.long()

            with torch.no_grad():

                q_next = self.manager_target(ns).max(1, keepdim=True)[0]
                target = r + (1 - d) * self.gamma * q_next

            q = self.manager(s).gather(1, o)

            loss = nn.MSELoss()(q, target)

            self.manager_opt.zero_grad()
            loss.backward()
            self.manager_opt.step()


        if self.learn_steps % self.target_update_freq == 0:

            self.manager_target.load_state_dict(self.manager.state_dict())

            for opt in self.options:

                opt.actor_target.load_state_dict(opt.actor.state_dict())
                opt.critic_target.load_state_dict(opt.critic.state_dict())

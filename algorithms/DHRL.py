
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.net(x)



class HighLevelNetwork(nn.Module):
    def __init__(self, input_dim, num_options):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_options)
        )

    def forward(self, x):
        return self.net(x)



class ReplayBuffer:
    def __init__(self, capacity=int(2e7)):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)



class OptionAgent:

    def __init__(self, state_dim, action_dim,
                 lr=0.01,
                 gamma=0.9,
                 batch_size=64,
                 target_update=100):

        self.q_net = DeepQNetwork(state_dim, action_dim)
        self.target_net = DeepQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.replay_buffer = ReplayBuffer()
        self.learn_steps = 0

    def select_action(self, state, epsilon=0.9):

        if random.random() < epsilon:
            return random.randrange(self.q_net.net[-1].out_features)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def update(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        next_q = self.target_net(next_states).max(1)[0]

        target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1

        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())



class HighLevelAgent:

    def __init__(self, state_dim, num_options,
                 lr=0.01,
                 gamma=0.9,
                 batch_size=64,
                 target_update=100):

        self.q_net = HighLevelNetwork(state_dim, num_options)
        self.target_net = HighLevelNetwork(state_dim, num_options)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.replay_buffer = ReplayBuffer()
        self.learn_steps = 0

    def select_option(self, state, epsilon=0.9):

        if random.random() < epsilon:
            return random.randrange(self.q_net.net[-1].out_features)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)

        return q_values.argmax().item()

    def update(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        states, options, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        options = torch.LongTensor(options)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, options.unsqueeze(1)).squeeze()

        next_q = self.target_net(next_states).max(1)[0]

        target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1

        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())



class DHRL:

    def __init__(self,
                 env,
                 state_dim,
                 action_dim,
                 num_options,
                 option_length=10):

        self.env = env
        self.option_length = option_length
        self.num_options = num_options

        self.manager = HighLevelAgent(state_dim, num_options)

        self.options = [
            OptionAgent(state_dim, action_dim)
            for _ in range(num_options)
        ]

    def train(self,
              episodes=1000,
              epsilon=0.9,
              max_steps=500):

        for episode in range(episodes):

            state = self.env.reset()
            total_reward = 0

            for step in range(max_steps):

                option = self.manager.select_option(state, epsilon)

                option_reward = 0

                for _ in range(self.option_length):

                    action = self.options[option].select_action(state, epsilon)

                    next_state, reward, done, _ = self.env.step(action)

                    self.options[option].replay_buffer.push(
                        state, action, reward, next_state, done
                    )

                    self.options[option].update()

                    option_reward += reward

                    state = next_state
                    total_reward += reward

                    if done:
                        break

                self.manager.replay_buffer.push(
                    state, option, option_reward, next_state, done
                )

                self.manager.update()

                if done:
                    break

            print(f"Episode {episode}  Reward: {total_reward}")



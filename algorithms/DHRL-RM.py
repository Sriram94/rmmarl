import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# ---------------------- Deep Low-level Network ----------------------
class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[1024]*6):
        super(DeepQNetwork, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ---------------------- Smaller High-level Network ----------------------
class HighLevelNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HighLevelNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# ---------------------- Replay Buffer ----------------------
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

# ---------------------- Low-level Agent ----------------------
class LowLevelAgent:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.9, buffer_size=int(2e7), batch_size=64):
        self.q_network = DeepQNetwork(state_dim, action_dim)
        self.target_network = DeepQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.target_update_freq = 100

    def select_action(self, state, epsilon=0.9):
        if random.random() < epsilon:
            return random.randint(0, self.q_network.model[-1].out_features - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_state).max(1)[0]
        expected_q = reward + self.gamma * next_q_values * (1 - done)

        loss = nn.MSELoss()(q_values, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# ---------------------- High-level Agent ----------------------
class HighLevelAgent:
    def __init__(self, state_dim, num_options, lr=0.01, gamma=0.9, buffer_size=int(2e7), batch_size=64):
        self.q_network = HighLevelNetwork(state_dim, num_options)
        self.target_network = HighLevelNetwork(state_dim, num_options)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.target_update_freq = 100

    def select_option(self, state, epsilon=0.9):
        if random.random() < epsilon:
            return random.randint(0, self.q_network.out.out_features - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, option, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        option = torch.LongTensor(option)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self.q_network(state).gather(1, option.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_state).max(1)[0]
        expected_q = reward + self.gamma * next_q_values * (1 - done)

        loss = nn.MSELoss()(q_values, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# ---------------------- DHRL-RM Agent ----------------------
class DHRL_RM:
    def __init__(self, env, rm, state_dim, action_dim, num_options, option_length=10):
        self.env = env
        self.rm = rm  # Reward Machine object with reset() and get_next_state()
        self.num_options = num_options
        self.option_length = option_length
        self.low_level_agents = [LowLevelAgent(state_dim, action_dim) for _ in range(num_options)]
        self.high_level_agent = HighLevelAgent(state_dim + rm.num_states, num_options)

    def rm_one_hot(self, rm_state):
        one_hot = np.zeros(self.rm.num_states)
        one_hot[rm_state] = 1
        return one_hot

    def train(self, num_episodes=1000, epsilon=0.9, max_steps=200):
        for ep in range(num_episodes):
            state = self.env.reset()
            rm_state = self.rm.reset()
            done = False
            step_count = 0

            while not done and step_count < max_steps:
                hl_state = np.concatenate([state, self.rm_one_hot(rm_state)])
                option = self.high_level_agent.select_option(hl_state, epsilon)

                for _ in range(self.option_length):
                    action = self.low_level_agents[option].select_action(state, epsilon)
                    next_state, reward, done, _ = self.env.step(action)

                    next_rm_state, rm_reward = self.rm.get_next_state(rm_state, next_state)

                    self.low_level_agents[option].replay_buffer.push(state, action, rm_reward, next_state, done)
                    self.high_level_agent.replay_buffer.push(hl_state, option, rm_reward, np.concatenate([next_state, self.rm_one_hot(next_rm_state)]), done)

                    self.low_level_agents[option].update()
                    self.high_level_agent.update()

                    state = next_state
                    rm_state = next_rm_state
                    step_count += 1

                    if done:
                        break

            for agent in self.low_level_agents:
                agent.update_target()
            self.high_level_agent.update_target()

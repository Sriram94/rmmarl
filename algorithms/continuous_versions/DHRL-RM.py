import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

class MetaActor(nn.Module):
    def __init__(self, state_dim, rm_state_dim, num_rm_states, hidden_size=256):
        super(MetaActor, self).__init__()
        input_dim = state_dim + rm_state_dim
        self.num_rm_states = num_rm_states

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_rm_states)
        )

    def forward(self, s, u_current):
        augmented_state = torch.cat([s, u_current], dim=-1)
        return self.net(augmented_state)

class MetaCritic(nn.Module):
    def __init__(self, state_dim, rm_state_dim, hidden_size=256):
        super(MetaCritic, self).__init__()
        input_dim = state_dim + (2 * rm_state_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, s, u_current, u_target):
        combined_input = torch.cat([s, u_current, u_target], dim=-1)
        return self.net(combined_input)

class LowLevelActor(nn.Module):
    def __init__(self, state_dim, rm_state_dim, action_dim, hidden_size=256):
        super(LowLevelActor, self).__init__()
        input_dim = state_dim + rm_state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

    def forward(self, s, u_target):
        augmented_state = torch.cat([s, u_target], dim=-1)
        return self.net(augmented_state)

class LowLevelCritic(nn.Module):
    def __init__(self, state_dim, rm_state_dim, action_dim, hidden_size=256):
        super(LowLevelCritic, self).__init__()
        input_dim = state_dim + rm_state_dim + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, s, u_target, a):
        combined_input = torch.cat([s, u_target, a], dim=-1)
        return self.net(combined_input)

class DHRL_RM_Agent:
    def __init__(self, state_dim, rm_state_dim, action_dim, num_rm_states, 
                 lr_meta=1e-4, lr_low=1e-3, gamma_meta=0.99, gamma_low=0.9, tau=0.01):
        
        self.gamma_meta = gamma_meta
        self.gamma_low = gamma_low
        self.tau = tau
        self.rm_state_dim = rm_state_dim
        self.num_rm_states = num_rm_states
        self.rm_states_tensor = torch.eye(num_rm_states)

        self.meta_actor = MetaActor(state_dim, rm_state_dim, num_rm_states)
        self.meta_target_actor = MetaActor(state_dim, rm_state_dim, num_rm_states)
        self.meta_target_actor.load_state_dict(self.meta_actor.state_dict())
        self.meta_actor_optimizer = optim.Adam(self.meta_actor.parameters(), lr=lr_meta)

        self.meta_critic = MetaCritic(state_dim, rm_state_dim)
        self.meta_target_critic = MetaCritic(state_dim, rm_state_dim)
        self.meta_target_critic.load_state_dict(self.meta_critic.state_dict())
        self.meta_critic_optimizer = optim.Adam(self.meta_critic.parameters(), lr=lr_meta)

        self.low_actor = LowLevelActor(state_dim, rm_state_dim, action_dim)
        self.low_target_actor = LowLevelActor(state_dim, rm_state_dim, action_dim)
        self.low_target_actor.load_state_dict(self.low_actor.state_dict())
        self.low_actor_optimizer = optim.Adam(self.low_actor.parameters(), lr=lr_low)
        
        self.low_critic = LowLevelCritic(state_dim, rm_state_dim, action_dim)
        self.low_target_critic = LowLevelCritic(state_dim, rm_state_dim, action_dim)
        self.low_target_critic.load_state_dict(self.low_critic.state_dict())
        self.low_critic_optimizer = optim.Adam(self.low_critic.parameters(), lr=lr_low)

    def select_action(self, s, u_current, u_target, explore=True, noise_scale=0.1):
        s_tensor = torch.FloatTensor(s).unsqueeze(0)
        u_target_tensor = torch.FloatTensor(u_target).unsqueeze(0)
        
        self.low_actor.eval()
        with torch.no_grad():
            action = self.low_actor(s_tensor, u_target_tensor).cpu().data.numpy().flatten()
        self.low_actor.train()

        if explore:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
            
        return action

    def select_sub_goal(self, s, u_current, explore=True):
        s_tensor = torch.FloatTensor(s).unsqueeze(0)
        u_current_tensor = torch.FloatTensor(u_current).unsqueeze(0)
        
        self.meta_actor.eval()
        with torch.no_grad():
            preferences = self.meta_actor(s_tensor, u_current_tensor)
        self.meta_actor.train()
        
        if explore:
            probs = torch.softmax(preferences, dim=-1)
            target_rm_idx = torch.multinomial(probs, num_samples=1).item()
        else:
            target_rm_idx = preferences.argmax(dim=-1).item()
            
        u_target = self.rm_states_tensor[target_rm_idx].cpu().numpy()
        return u_target, target_rm_idx

    def soft_update(self):
        for target_param, param in zip(self.low_target_critic.parameters(), self.low_critic.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
        for target_param, param in zip(self.low_target_actor.parameters(), self.low_actor.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
        
        for target_param, param in zip(self.meta_target_critic.parameters(), self.meta_critic.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
        for target_param, param in zip(self.meta_target_actor.parameters(), self.meta_actor.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)

    def update_low_level(self, batch):
        s, u_target, a, r, s_prime, terminal = batch['s'], batch['u_target'], batch['a'], batch['r'], batch['s_prime'], batch['terminal']
        
        with torch.no_grad():
            a_prime = self.low_target_actor(s_prime, u_target)
            Q_target_prime = self.low_target_critic(s_prime, u_target, a_prime).squeeze(-1)
            y = r + self.gamma_low * Q_target_prime * (1 - terminal)
            
        Q_current = self.low_critic(s, u_target, a).squeeze(-1)
        critic_loss = nn.MSELoss()(Q_current, y)

        self.low_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.low_critic_optimizer.step()

        a_current = self.low_actor(s, u_target)
        Q_value = self.low_critic(s, u_target, a_current)
        actor_loss = -Q_value.mean()

        self.low_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.low_actor_optimizer.step()
        
        return critic_loss, actor_loss

    def update_meta_level(self, batch):
        s, u_current, u_target, r_rm, s_prime, u_prime, terminal_rm = \
            batch['s'], batch['u_current'], batch['u_target'], batch['r_rm'], batch['s_prime'], batch['u_prime'], batch['terminal_rm']
        
        with torch.no_grad():
            u_target_prime_preferences = self.meta_target_actor(s_prime, u_prime)
            u_target_prime_idx = torch.argmax(u_target_prime_preferences, dim=-1).unsqueeze(-1)
            
            batch_size = s_prime.size(0)
            u_target_prime = self.rm_states_tensor[u_target_prime_idx.squeeze()].to(s_prime.device)
            if batch_size == 1: u_target_prime = u_target_prime.unsqueeze(0)

            Q_target_prime = self.meta_target_critic(s_prime, u_prime, u_target_prime).squeeze(-1)
            
            y = r_rm + self.gamma_meta * Q_target_prime * (1 - terminal_rm)
            
        Q_current = self.meta_critic(s, u_current, u_target).squeeze(-1)
        critic_loss = nn.MSELoss()(Q_current, y)

        self.meta_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.meta_critic_optimizer.step()

        u_target_preferences = self.meta_actor(s, u_current)
        
        u_target_idx = torch.argmax(u_target_preferences, dim=-1).unsqueeze(-1)
        u_target_current_policy = self.rm_states_tensor[u_target_idx.squeeze()].to(s.device)
        if batch_size == 1: u_target_current_policy = u_target_current_policy.unsqueeze(0)
        
        Q_value = self.meta_critic(s, u_current, u_target_current_policy)
        actor_loss = -Q_value.mean()

        self.meta_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.meta_actor_optimizer.step()
        
        return critic_loss, actor_loss

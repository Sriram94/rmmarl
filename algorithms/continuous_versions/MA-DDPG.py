import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, rm_state_dim, action_dim, hidden_size=256):
        super(ActorNetwork, self).__init__()
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

    def forward(self, s, u_j):
        augmented_state = torch.cat([s, u_j], dim=-1)
        return self.net(augmented_state)

class CriticNetwork(nn.Module):
    def __init__(self, global_state_dim, total_rm_state_dim, total_action_dim, hidden_size=256):
        super(CriticNetwork, self).__init__()
        input_dim = global_state_dim + total_rm_state_dim + total_action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, s, all_u, all_a):
        combined_input = torch.cat([s, all_u, all_a], dim=-1)
        return self.net(combined_input)

class DCROM_Agent:
    def __init__(self, agent_id, state_dim, rm_state_dim, action_dim, 
                 global_state_dim, total_rm_state_dim, total_action_dim, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01):
        
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau 
        
        self.actor = ActorNetwork(state_dim, rm_state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, rm_state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, s, u_j, explore=True, noise_scale=0.1):
        s_tensor = torch.FloatTensor(s).unsqueeze(0)
        u_j_tensor = torch.FloatTensor(u_j).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(s_tensor, u_j_tensor).cpu().data.numpy().flatten()
        self.actor.train()

        if explore:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
            
        return action

    def soft_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)

    def update_critic(self, batch, all_target_actors, compute_joint_action):
        s, u_j, all_a, r_j, s_prime, u_j_prime, all_u, all_u_prime, terminal = \
            batch['s'], batch['u_j'], batch['all_a'], batch['r_j'], batch['s_prime'], batch['u_j_prime'], batch['all_u'], batch['all_u_prime'], batch['terminal']
        
        with torch.no_grad():
            all_a_prime = compute_joint_action(s_prime, all_u_prime, all_target_actors)
            
            Q_target_prime = self.target_critic(s_prime, all_u_prime, all_a_prime).squeeze(-1)
            
            y_j = r_j + self.gamma * Q_target_prime * (1 - terminal)
        
        Q_current = self.critic(s, all_u, all_a).squeeze(-1)
        critic_loss = nn.MSELoss()(Q_current, y_j)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def update_actor(self, batch, all_current_actors, compute_joint_action):
        s, u_j, all_u = batch['s'], batch['u_j'], batch['all_u']
        
        a_j_current = self.actor(s, u_j)
        
        all_a_current_from_actors = compute_joint_action(s, all_u, all_current_actors)
        
        Q_value = self.critic(s, all_u, all_a_current_from_actors)
        
        actor_loss = -Q_value.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

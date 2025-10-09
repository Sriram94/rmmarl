import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, rm_state_dim, action_dim, hidden_size=256):
        super(ActorNetwork, self).__init__()
        input_dim = state_dim + rm_state_dim
        
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
        action = self.net(augmented_state)
        return action

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
        q_value = self.net(combined_input)
        return q_value


class MADDPG_RM_Agent:
    def __init__(self, agent_id, state_dim, rm_state_dim, action_dim, 
                 global_state_dim, total_rm_state_dim, total_action_dim, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.9, tau=0.01):
        
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
        s = torch.FloatTensor(s).unsqueeze(0)
        u_j = torch.FloatTensor(u_j).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(s, u_j).cpu().data.numpy().flatten()
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

    def update_critic(self, batch, all_target_actors, next_actions_all):
        s, u_j, all_a, r_j, s_prime, u_j_prime, terminal = batch 
        
        with torch.no_grad():
            a_j_prime = self.target_actor(s_prime, u_j_prime)
            
            all_a_prime = next_actions_all(s_prime, all_target_actors) 
            
            all_u_prime = torch.cat([u_i_prime for u_i_prime in batch['all_u_prime']], dim=-1) 
            
            Q_target_prime = self.target_critic(s_prime, all_u_prime, all_a_prime).squeeze(-1)
            
            y_j = r_j + self.gamma * Q_target_prime * (1 - terminal)
        
        
        all_u = torch.cat([u_i for u_i in batch['all_u']], dim=-1) 
        Q_current = self.critic(s, all_u, all_a).squeeze(-1)

        critic_loss = nn.MSELoss()(Q_current, y_j)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, batch, all_current_actors):
        s, u_j = batch['s'], batch['u_j']
        
        a_j_current = self.actor(s, u_j)
        
        all_a_current_from_actors = self.compute_joint_action(s, all_current_actors)
        
        all_u = torch.cat([u_i for u_i in batch['all_u']], dim=-1)
        Q_value = self.critic(s, all_u, all_a_current_from_actors)
        
        actor_loss = -Q_value.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


class RM_ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
             return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        s_batch = torch.FloatTensor([e['s'] for e in batch])
        return {'s': s_batch, 'u_j': s_batch} 

    def __len__(self):
        return len(self.buffer)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

    def forward(self, s):
        return self.net(s)

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

class DQRM_SubAgent:
    def __init__(self, state_dim, action_dim, global_state_dim, total_rm_state_dim, total_action_dim, lr_actor, lr_critic, tau):
        self.tau = tau
        
        self.actor = ActorNetwork(state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def soft_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)

class DQRM_Agent:
    def __init__(self, agent_id, num_agents, num_rm_states, state_dim, action_dim, 
                 global_state_dim, total_rm_state_dim, total_action_dim, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01):
        
        self.agent_id = agent_id
        self.gamma = gamma
        self.num_rm_states = num_rm_states
        
        self.rm_sub_agents: List[DQRM_SubAgent] = []
        for u_i in range(num_rm_states):
            sub_agent = DQRM_SubAgent(state_dim, action_dim, global_state_dim, total_rm_state_dim, total_action_dim, lr_actor, lr_critic, tau)
            self.rm_sub_agents.append(sub_agent)

    def select_action(self, s, u_j_idx, explore=True, noise_scale=0.1):
        sub_agent = self.rm_sub_agents[u_j_idx]
        
        s = torch.FloatTensor(s).unsqueeze(0)
        
        sub_agent.actor.eval()
        with torch.no_grad():
            action = sub_agent.actor(s).cpu().data.numpy().flatten()
        sub_agent.actor.train()

        if explore:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
            
        return action

    def update(self, batch, all_rm_sub_agents_list):
        u_j_idx = batch['u_j_idx'][0].item()
        sub_agent = self.rm_sub_agents[u_j_idx]
        gamma = self.gamma

        s, all_a, r_j, s_prime, terminal = batch['s'], batch['all_a'], batch['r_j'], batch['s_prime'], batch['terminal']
        u_j_prime_idx = batch['u_j_prime_idx'][0].item()
        all_u, all_u_prime = batch['all_u'], batch['all_u_prime']

        with torch.no_grad():
            all_a_prime_list = []
            
            for i in range(len(all_rm_sub_agents_list)):
                u_i_prime_idx = batch['all_u_prime_idx'][i][0].item()
                agent_i_target_sub_agent = all_rm_sub_agents_list[i][u_i_prime_idx]
                
                a_i_prime = agent_i_target_sub_agent.target_actor(s_prime)
                all_a_prime_list.append(a_i_prime)
                
            all_a_prime = torch.cat(all_a_prime_list, dim=-1)

            target_sub_agent_prime = self.rm_sub_agents[u_j_prime_idx]
            Q_target_prime = target_sub_agent_prime.target_critic(s_prime, all_u_prime, all_a_prime).squeeze(-1)
            y_j = r_j + gamma * Q_target_prime * (1 - terminal)

        Q_current = sub_agent.critic(s, all_u, all_a).squeeze(-1)
        critic_loss = nn.MSELoss()(Q_current, y_j)

        sub_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        sub_agent.critic_optimizer.step()

        a_j_current = sub_agent.actor(s)
        
        all_a_current_list = []
        for i in range(len(all_rm_sub_agents_list)):
            if i == self.agent_id:
                all_a_current_list.append(a_j_current)
            else:
                u_i_idx = batch['all_u_idx'][i][0].item()
                agent_i_current_sub_agent = all_rm_sub_agents_list[i][u_i_idx]
                
                with torch.no_grad():
                    a_i_current = agent_i_current_sub_agent.actor(s)
                all_a_current_list.append(a_i_current)
        
        all_a_current_from_actors = torch.cat(all_a_current_list, dim=-1)
        
        Q_value = sub_agent.critic(s, all_u, all_a_current_from_actors)
        
        actor_loss = -Q_value.mean()
        
        sub_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        sub_agent.actor_optimizer.step()
        
        sub_agent.soft_update()
        
        return critic_loss, actor_loss

    def soft_update_all(self):
        for sub_agent in self.rm_sub_agents:
            sub_agent.soft_update()

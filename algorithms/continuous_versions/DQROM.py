import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim # Needed for joint action construction
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh() # Scales action to [-1, 1]
        )

    def forward(self, s):
        return self.net(s)

class CriticNetwork(nn.Module):
    def __init__(self, global_state_dim, total_rm_state_dim, total_action_dim, hidden_size=256):
        super(CriticNetwork, self).__init__()
        # Centralized Input: S + U + A
        input_dim = global_state_dim + total_rm_state_dim + total_action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # Output: Q-value
        )

    def forward(self, s, all_u, all_a):
        combined_input = torch.cat([s, all_u, all_a], dim=-1)
        return self.net(combined_input)


class OpponentModel(nn.Module):
    def __init__(self, state_dim, rm_state_dim, action_dim, hidden_size=64):
        super(OpponentModel, self).__init__()
        input_dim = state_dim + rm_state_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax()
        )

    def forward(self, s, u_k):
        augmented_state = torch.cat([s, u_k], dim=-1)
        return self.net(augmented_state)


class DQROM_SubAgent:
    def __init__(self, state_dim, action_dim, global_state_dim, total_rm_state_dim, total_action_dim, lr_actor, lr_critic, tau):
        self.tau = tau
        
        # Actor Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic Networks
        self.critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def soft_update(self):
        """Soft update targets."""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)



class DQROM_Agent:
    def __init__(self, agent_id, num_agents, num_rm_states, state_dim, rm_state_dim, action_dim, 
                 global_state_dim, total_rm_state_dim, total_action_dim, 
                 lr_actor=1e-4, lr_critic=1e-3, lr_om=1e-4, gamma=0.9, tau=0.01):
        
        self.agent_id = agent_id
        self.gamma = gamma
        self.num_agents = num_agents
        self.num_rm_states = num_rm_states
        self.state_dim = state_dim
        self.rm_state_dim = rm_state_dim
        self.action_dim = action_dim
        
        self.rm_sub_agents: List[DQROM_SubAgent] = []
        for u_i in range(num_rm_states):
            sub_agent = DQROM_SubAgent(state_dim, action_dim, global_state_dim, total_rm_state_dim, total_action_dim, lr_actor, lr_critic, tau)
            self.rm_sub_agents.append(sub_agent)
            
        self.opponent_models: Dict[int, OpponentModel] = {}
        self.om_optimizers: Dict[int, optim.Adam] = {}
        self.opponent_agents = [k for k in range(num_agents) if k != agent_id]
        
        for k in self.opponent_agents:
            om_k = OpponentModel(state_dim, rm_state_dim, action_dim)
            self.opponent_models[k] = om_k
            self.om_optimizers[k] = optim.Adam(om_k.parameters(), lr=lr_om)

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

    def predict_opponent_actions(self, s, all_u):
        predicted_actions = []
        
        for k in range(self.num_agents):
            if k == self.agent_id:
                predicted_actions.append(None) 
            else:
                om_k = self.opponent_models[k]
                u_k_idx_start = k * self.rm_state_dim
                u_k = all_u[:, u_k_idx_start : u_k_idx_start + self.rm_state_dim]
                
                om_k.eval()
                with torch.no_grad():
                    a_k_pred = om_k(s, u_k)
                om_k.train()
                predicted_actions.append(a_k_pred)
                
        return predicted_actions

    def update_opponent_models(self, batch):
        pass 

    def update(self, batch, all_rm_sub_agents: List[DQROM_SubAgent], all_om_models: List[Dict[int, OpponentModel]]):
        u_j_idx = batch['u_j_idx'][0].item() # Assume batch size 1 for simplicity here
        sub_agent = self.rm_sub_agents[u_j_idx]
        gamma = self.gamma

        s, all_a, r_j, s_prime, terminal = batch['s'], batch['all_a'], batch['r_j'], batch['s_prime'], batch['terminal']
        u_j_prime_idx = batch['u_j_prime_idx'][0].item()
        all_u, all_u_prime = batch['all_u'], batch['all_u_prime']

        with torch.no_grad():
            target_sub_agent_prime = self.rm_sub_agents[u_j_prime_idx]
            a_j_prime = target_sub_agent_prime.target_actor(s_prime)
            
            predicted_actions_prime = self.predict_opponent_actions(s_prime, all_u_prime)
            
            predicted_actions_prime[self.agent_id] = a_j_prime
            all_a_prime = torch.cat(predicted_actions_prime, dim=-1)

            Q_target_prime = target_sub_agent_prime.target_critic(s_prime, all_u_prime, all_a_prime).squeeze(-1)
            y_j = r_j + gamma * Q_target_prime * (1 - terminal)

        Q_current = sub_agent.critic(s, all_u, all_a).squeeze(-1)
        critic_loss = nn.MSELoss()(Q_current, y_j)

        sub_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        sub_agent.critic_optimizer.step()

        a_j_current = sub_agent.actor(s)
        
        predicted_actions = self.predict_opponent_actions(s, all_u)
        
        predicted_actions[self.agent_id] = a_j_current
        all_a_current_from_om = torch.cat(predicted_actions, dim=-1)
        
        Q_value = sub_agent.critic(s, all_u, all_a_current_from_om)
        
        actor_loss = -Q_value.mean()
        
        sub_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        sub_agent.actor_optimizer.step()
        
        sub_agent.soft_update()
        
        return critic_loss, actor_loss

    def soft_update_all(self):
        for sub_agent in self.rm_sub_agents:
            sub_agent.soft_update()

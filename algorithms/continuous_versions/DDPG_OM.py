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




class OpponentModel(nn.Module):
    def __init__(self, state_dim, rm_state_dim, action_dim, hidden_size=64):
        super(OpponentModel, self).__init__()
        input_dim = state_dim + rm_state_dim
        
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
        action = self.net(augmented_state)
        return action


class MADDPG_OM_Agent:
    def __init__(self, agent_id, num_agents, state_dim, rm_state_dim, action_dim, 
                 global_state_dim, total_rm_state_dim, total_action_dim, 
                 lr_actor=1e-4, lr_critic=1e-3, lr_om=1e-4, gamma=0.9, tau=0.01):
        
        
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau
        self.state_dim = state_dim
        self.rm_state_dim = rm_state_dim
        self.num_agents = num_agents

        self.opponent_models = {}
        self.om_optimizers = {}
        self.opponent_agents = [k for k in range(num_agents) if k != agent_id]
        
        for k in self.opponent_agents:
            om_k = OpponentModel(state_dim, rm_state_dim, action_dim)
            self.opponent_models[k] = om_k
            self.om_optimizers[k] = optim.Adam(om_k.parameters(), lr=lr_om)

        self.actor = ActorNetwork(state_dim, rm_state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, rm_state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)


    
    def predict_opponent_actions(self, s, all_u):
        predicted_actions = []
        
        for k in range(self.num_agents):
            if k == self.agent_id:
                predicted_actions.append(None) 
            else:
                om_k = self.opponent_models[k]
                u_k = all_u[:, k * self.rm_state_dim : (k+1) * self.rm_state_dim]
                
                om_k.eval()
                with torch.no_grad():
                    a_k_pred = om_k(s, u_k)
                om_k.train()
                predicted_actions.append(a_k_pred)
                
        return predicted_actions 


    def update_opponent_models(self, batch):
        s, all_u, all_a = batch['s'], batch['all_u'], batch['all_a']
        
        om_loss = 0
        for k in self.opponent_agents:
            om_k = self.opponent_models[k]
            om_optimizer = self.om_optimizers[k]

            u_k = all_u[:, k * self.rm_state_dim : (k+1) * self.rm_state_dim]
            action_dim = self.actor.action_dim 
            a_k_observed = all_a[:, k * action_dim : (k+1) * action_dim]
            
            a_k_predicted = om_k(s, u_k)
            
            loss = nn.MSELoss()(a_k_predicted, a_k_observed)
            om_loss += loss

            om_optimizer.zero_grad()
            loss.backward()
            om_optimizer.step()
        
        return om_loss / len(self.opponent_agents) if self.opponent_agents else 0


    def update_critic(self, batch, all_u_prime, all_u):
        s, u_j, all_a, r_j, s_prime, u_j_prime, terminal = batch['s'], batch['u_j'], batch['all_a'], batch['r_j'], batch['s_prime'], batch['u_j_prime'], batch['terminal']
        
        with torch.no_grad():
            a_j_prime = self.target_actor(s_prime, u_j_prime)
            
            predicted_actions_prime = self.predict_opponent_actions(s_prime, all_u_prime)
            
            predicted_actions_prime[self.agent_id] = a_j_prime
            all_a_prime = torch.cat(predicted_actions_prime, dim=-1) 
            
            Q_target_prime = self.target_critic(s_prime, all_u_prime, all_a_prime).squeeze(-1)
            y_j = r_j + self.gamma * Q_target_prime * (1 - terminal)
        
        Q_current = self.critic(s, all_u, all_a).squeeze(-1)
        critic_loss = nn.MSELoss()(Q_current, y_j)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, batch, all_u):
        s, u_j = batch['s'], batch['u_j']
        
        a_j_current = self.actor(s, u_j)
        
        predicted_actions = self.predict_opponent_actions(s, all_u)
        
        predicted_actions[self.agent_id] = a_j_current
        all_a_current_from_om = torch.cat(predicted_actions, dim=-1)
        
        Q_value = self.critic(s, all_u, all_a_current_from_om)
        
        actor_loss = -Q_value.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

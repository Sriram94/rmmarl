import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Network Architecture ---

class ActorNetwork(nn.Module):
    """
    The Actor (Policy) network. Input is the local augmented state (s, u_j).
    Output is a continuous action a_j.
    """
    def __init__(self, state_dim, rm_state_dim, action_dim, hidden_size=256):
        super(ActorNetwork, self).__init__()
        # Input: Environment State (s) + RM State (u_j)
        input_dim = state_dim + rm_state_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh() # Tanh scales the output action to [-1, 1]
        )

    def forward(self, s, u_j):
        # Concatenate environment state and RM state
        augmented_state = torch.cat([s, u_j], dim=-1)
        action = self.net(augmented_state)
        return action

class CriticNetwork(nn.Module):
    """
    The Critic (Q-function) network.
    Input is the joint observation (all states s, all RM states U) and joint action vector (A).
    In the multi-agent context (MADDPG), the Critic is centralized.
    """
    def __init__(self, global_state_dim, total_rm_state_dim, total_action_dim, hidden_size=256):
        super(CriticNetwork, self).__init__()
        # Input: Global State (s) + All RM States (U) + All Actions (A)
        input_dim = global_state_dim + total_rm_state_dim + total_action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # Output: Q-value (a scalar)
        )

    def forward(self, s, all_u, all_a):
        # Concatenate global state, all RM states, and all actions
        combined_input = torch.cat([s, all_u, all_a], dim=-1)
        q_value = self.net(combined_input)
        return q_value

# --- 2. DDPG Agent Class ---

class MADDPG_RM_Agent:
    """
    DDPG Agent with Reward Machine State Augmentation (DCROM/DDPG-based).
    The agent's policy is decentralized, and the value function is centralized.
    """
    def __init__(self, agent_id, state_dim, rm_state_dim, action_dim, 
                 global_state_dim, total_rm_state_dim, total_action_dim, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.9, tau=0.01):
        
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau # For soft target updates
        
        # 1. Actor (Policy) Networks: Decentralized input (s, u_j) -> action a_j
        self.actor = ActorNetwork(state_dim, rm_state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, rm_state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # 2. Critic (Q-function) Networks: Centralized input (s, U, A) -> Q_j
        self.critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic = CriticNetwork(global_state_dim, total_rm_state_dim, total_action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, s, u_j, explore=True, noise_scale=0.1):
        """Select action using the current policy and optional exploration noise."""
        s = torch.FloatTensor(s).unsqueeze(0)
        u_j = torch.FloatTensor(u_j).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(s, u_j).cpu().data.numpy().flatten()
        self.actor.train()

        if explore:
            # Add Gaussian exploration noise (standard DDPG/MADDPG approach)
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
            
        return action

    def soft_update(self):
        """Soft update the target networks towards the primary networks."""
        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
        
        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)

    def update_critic(self, batch, all_target_actors, next_actions_all):
        """
        Update the Critic (Q-function) network using the Bellman equation.
        batch: (s, u_j, a_j, r_j, s', u_j', terminal)
        all_target_actors: list of target actor networks for all agents
        next_actions_all: a function that computes joint actions for the next state s'
        """
        # Unpack batch data (Assuming batch data is properly formatted)
        s, u_j, all_a, r_j, s_prime, u_j_prime, terminal = batch 
        
        # --- 1. Compute target Q-value (y_j) ---
        
        # 1a. Compute next joint action (A') from target policies (Decentralized Execution)
        with torch.no_grad():
            # Get the next action for agent j from its own target actor
            a_j_prime = self.target_actor(s_prime, u_j_prime)
            
            # Use all agents' target actors to predict the joint next action (A')
            # NOTE: In a full MADDPG-RM setup, this part requires all agents' (s', u_i') and target_actor_i
            # We assume 'next_actions_all' handles the collection of all a_i_prime to form A'
            all_a_prime = next_actions_all(s_prime, all_target_actors) # A' = (a_1', ..., a_N')
            
            # 1b. Compute Q'(s', U', A') from the target critic (Centralized Evaluation)
            # The next global RM state (U') would be a concatenation of all u_i_prime
            all_u_prime = torch.cat([u_i_prime for u_i_prime in batch['all_u_prime']], dim=-1) # Placeholder logic
            
            # Compute Q' for the next state
            Q_target_prime = self.target_critic(s_prime, all_u_prime, all_a_prime).squeeze(-1)
            
            # Compute the final target value (y_j = r_j + gamma * Q'(...) * (1 - terminal))
            y_j = r_j + self.gamma * Q_target_prime * (1 - terminal)
        
        # --- 2. Compute current Q-value (Q_j) and Loss ---
        
        # Compute Q(s, U, A) from the current critic network
        all_u = torch.cat([u_i for u_i in batch['all_u']], dim=-1) # Placeholder logic
        Q_current = self.critic(s, all_u, all_a).squeeze(-1)

        # Compute the Critic Loss (Mean Squared Error)
        critic_loss = nn.MSELoss()(Q_current, y_j)

        # --- 3. Optimization Step ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, batch, all_current_actors):
        """
        Update the Actor (Policy) network.
        Minimize the negative Q-value to maximize expected return.
        """
        s, u_j = batch['s'], batch['u_j']
        
        # 1. Compute the current action from the current actor
        a_j_current = self.actor(s, u_j)
        
        # 2. Form the joint action vector A, replacing agent j's action with a_j_current
        # NOTE: This requires coordination with other agents' current actors (all_current_actors)
        # Assuming 'all_a_current_from_actors' computes the joint action vector A
        all_a_current_from_actors = self.compute_joint_action(s, all_current_actors)
        
        # 3. Compute Q(s, U, A) from the current critic using the new joint action A
        all_u = torch.cat([u_i for u_i in batch['all_u']], dim=-1)
        Q_value = self.critic(s, all_u, all_a_current_from_actors)
        
        # 4. Compute Actor Loss (Policy gradient: maximize Q -> minimize -Q)
        actor_loss = -Q_value.mean()
        
        # 5. Optimization Step
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# --- 3. Replay Buffer (RM-Augmented Experience) ---

class RM_ReplayBuffer:
    """
    Experience replay buffer to store the augmented experience tuple.
    Experience tuple is: (s, u_j, a, r_j, s', u_j', terminal)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # experience is assumed to be a dictionary/tuple containing 
        # s, u_j, all_a, r_j, s_prime, u_j_prime, all_u, all_u_prime, terminal
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Implementation to sample a batch of experiences
        # Returns a batch dictionary/tuple ready for training
        if len(self.buffer) < batch_size:
             return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        # Example of converting to a dictionary of tensors (full implementation would be more complex)
        s_batch = torch.FloatTensor([e['s'] for e in batch])
        # ... and so on for all components (u_j, all_a, r_j, s', u_j', terminal)
        
        # Simplified return for concept
        return {'s': s_batch, 'u_j': s_batch} # Placeholder

    def __len__(self):
        return len(self.buffer)


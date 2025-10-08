"""
CROM: Counterfactual Experiences for Reward Machines with Opponent Modelling
Implementation based on Algorithm 1 from the paper:
"Multi-Agent Reinforcement Learning with Reward Machines for 
Mixed Cooperative-Competitive Environments"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import random


class OpponentModel(nn.Module):
    """Neural network for predicting opponent actions"""
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super(OpponentModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),  # state + own_rm_state + prev_action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state, rm_state, prev_action):
        """
        Args:
            state: Environment state
            rm_state: Agent's own RM state
            prev_action: Previous action of the modeled agent
        """
        x = torch.cat([state, rm_state.unsqueeze(-1), prev_action.unsqueeze(-1)], dim=-1)
        return self.network(x)


class RewardMachine:
    """Reward Machine representation"""
    
    def __init__(self, states: List[int], initial_state: int, 
                 events: List[str], transitions: Dict, rewards: Dict):
        """
        Args:
            states: List of RM states
            initial_state: Initial RM state
            events: List of possible events
            transitions: Dict mapping (state, event) -> next_state
            rewards: Dict mapping (state, next_state) -> reward
        """
        self.states = states
        self.initial_state = initial_state
        self.events = events
        self.transitions = transitions
        self.rewards = rewards
        
    def get_next_state(self, current_state: int, event: str) -> int:
        """Get next RM state given current state and event"""
        return self.transitions.get((current_state, event), current_state)
    
    def get_reward(self, current_state: int, next_state: int) -> float:
        """Get reward for transitioning between RM states"""
        return self.rewards.get((current_state, next_state), 0.0)
    
    def label_function(self, env_state: Any) -> str:
        """
        Label function that maps environment state to events.
        This should be implemented based on the specific environment.
        """
        raise NotImplementedError("Label function must be implemented for specific environment")


class TabularCROM:
    """Tabular CROM implementation (Algorithm 1)"""
    
    def __init__(self, 
                 n_agents: int,
                 state_space_size: int,
                 action_spaces: List[int],
                 reward_machines: List[RewardMachine],
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1):
        """
        Args:
            n_agents: Number of agents
            state_space_size: Size of state space
            action_spaces: List of action space sizes for each agent
            reward_machines: List of reward machines for each agent
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.n_agents = n_agents
        self.state_space_size = state_space_size
        self.action_spaces = action_spaces
        self.reward_machines = reward_machines
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-tables for each agent
        # Q[agent][state][rm_state][joint_action_tuple]
        self.Q = []
        for agent_id in range(n_agents):
            agent_q = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            self.Q.append(agent_q)
        
        # Initialize opponent models (simplified for tabular case)
        # In practice, this would be neural networks
        self.opponent_models = []
        for agent_id in range(n_agents):
            agent_opponent_models = {}
            for other_agent in range(n_agents):
                if other_agent != agent_id:
                    # Store action frequencies for simple opponent modeling
                    agent_opponent_models[other_agent] = defaultdict(lambda: defaultdict(int))
            self.opponent_models.append(agent_opponent_models)
        
        # Track previous actions for opponent modeling
        self.prev_actions = [0] * n_agents
        
    def get_joint_action_tuple(self, actions: List[int]) -> Tuple:
        """Convert action list to tuple for dictionary key"""
        return tuple(actions)
    
    def select_action(self, agent_id: int, state: int, rm_state: int, 
                     other_actions: List[int] = None) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            agent_id: ID of the agent selecting action
            state: Current environment state
            rm_state: Current RM state for this agent
            other_actions: Actions of other agents (for Q-value lookup)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_spaces[agent_id] - 1)
        
        # Get predicted actions for other agents if not provided
        if other_actions is None:
            other_actions = self.predict_opponent_actions(agent_id, state, rm_state)
        
        # Find best action
        best_action = 0
        best_value = float('-inf')
        
        for action in range(self.action_spaces[agent_id]):
            joint_action = other_actions[:agent_id] + [action] + other_actions[agent_id:]
            joint_action_tuple = self.get_joint_action_tuple(joint_action)
            q_value = self.Q[agent_id][state][rm_state][joint_action_tuple]
            
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action
    
    def predict_opponent_actions(self, agent_id: int, state: int, rm_state: int) -> List[int]:
        """Predict actions of other agents using opponent models"""
        predicted_actions = []
        
        for other_agent in range(self.n_agents):
            if other_agent == agent_id:
                predicted_actions.append(0)  # Placeholder for own action
            else:
                # Use frequency-based prediction (simplified)
                state_key = (state, rm_state)
                action_counts = self.opponent_models[agent_id][other_agent][state_key]
                
                if not action_counts:
                    # Random prediction if no history
                    predicted_actions.append(random.randint(0, self.action_spaces[other_agent] - 1))
                else:
                    # Most frequent action
                    predicted_action = max(action_counts.items(), key=lambda x: x[1])[0]
                    predicted_actions.append(predicted_action)
        
        return predicted_actions
    
    def update(self, agent_id: int, state: int, rm_state: int, 
               joint_action: List[int], reward: float, 
               next_state: int, next_rm_state: int, terminal: bool):
        """
        Update Q-value for a single experience
        
        Args:
            agent_id: ID of the agent
            state: Current state
            rm_state: Current RM state
            joint_action: Joint action of all agents
            reward: Reward received
            next_state: Next state
            next_rm_state: Next RM state
            terminal: Whether episode terminated
        """
        joint_action_tuple = self.get_joint_action_tuple(joint_action)
        
        if terminal:
            target = reward
        else:
            # Predict next joint action
            next_other_actions = self.predict_opponent_actions(agent_id, next_state, next_rm_state)
            
            # Find max Q-value over agent's actions
            max_q = float('-inf')
            for next_action in range(self.action_spaces[agent_id]):
                next_joint_action = next_other_actions[:agent_id] + [next_action] + next_other_actions[agent_id:]
                next_joint_tuple = self.get_joint_action_tuple(next_joint_action)
                q_val = self.Q[agent_id][next_state][next_rm_state][next_joint_tuple]
                max_q = max(max_q, q_val)
            
            target = reward + self.gamma * max_q
        
        # Q-learning update
        current_q = self.Q[agent_id][state][rm_state][joint_action_tuple]
        self.Q[agent_id][state][rm_state][joint_action_tuple] = \
            current_q + self.alpha * (target - current_q)
    
    def counterfactual_update(self, agent_id: int, state: int, 
                             joint_action: List[int], next_state: int):
        """
        Perform counterfactual updates for all RM states
        
        Args:
            agent_id: ID of the agent
            state: Current environment state
            joint_action: Joint action taken
            next_state: Next environment state
        """
        rm = self.reward_machines[agent_id]
        
        # For each possible RM state
        for rm_state in rm.states:
            # Get the event that occurred
            event = rm.label_function(next_state)
            
            # Compute counterfactual next RM state
            next_rm_state = rm.get_next_state(rm_state, event)
            
            # Compute counterfactual reward
            reward = rm.get_reward(rm_state, next_rm_state)
            
            # Determine if terminal
            terminal = False  # This should be determined by environment/RM
            
            # Update Q-value for this counterfactual experience
            self.update(agent_id, state, rm_state, joint_action, 
                       reward, next_state, next_rm_state, terminal)
    
    def update_opponent_model(self, agent_id: int, state: int, rm_state: int,
                             other_agent: int, observed_action: int):
        """Update opponent model with observed action"""
        state_key = (state, rm_state)
        self.opponent_models[agent_id][other_agent][state_key][observed_action] += 1
    
    def train_episode(self, env, max_steps: int = 1000):
        """
        Train for one episode
        
        Args:
            env: Environment instance (should implement reset(), step(), etc.)
            max_steps: Maximum steps per episode
        """
        state = env.reset()
        rm_states = [rm.initial_state for rm in self.reward_machines]
        
        for step in range(max_steps):
            # Select actions for all agents
            joint_action = []
            for agent_id in range(self.n_agents):
                action = self.select_action(agent_id, state, rm_states[agent_id])
                joint_action.append(action)
            
            # Execute joint action
            next_state, rewards, done, info = env.step(joint_action)
            
            # Update RM states and get rewards
            next_rm_states = []
            for agent_id in range(self.n_agents):
                rm = self.reward_machines[agent_id]
                event = rm.label_function(next_state)
                next_rm_state = rm.get_next_state(rm_states[agent_id], event)
                next_rm_states.append(next_rm_state)
            
            # Perform regular and counterfactual updates for each agent
            for agent_id in range(self.n_agents):
                # Regular update
                self.update(agent_id, state, rm_states[agent_id], 
                           joint_action, rewards[agent_id], 
                           next_state, next_rm_states[agent_id], done)
                
                # Counterfactual updates
                self.counterfactual_update(agent_id, state, joint_action, next_state)
                
                # Update opponent models
                for other_agent in range(self.n_agents):
                    if other_agent != agent_id:
                        self.update_opponent_model(agent_id, state, rm_states[agent_id],
                                                  other_agent, joint_action[other_agent])
            
            # Update state
            state = next_state
            rm_states = next_rm_states
            
            if done:
                break
        
        return step + 1


# Example usage
if __name__ == "__main__":
    # Define a simple reward machine for demonstration
    class SimpleRM(RewardMachine):
        def __init__(self):
            states = [0, 1, 2]  # Initial, intermediate, goal
            initial_state = 0
            events = ['nothing', 'pickup', 'deliver']
            transitions = {
                (0, 'pickup'): 1,
                (1, 'deliver'): 2,
            }
            rewards = {
                (0, 1): 0.5,  # Small reward for pickup
                (1, 2): 1.0,  # Large reward for delivery
            }
            super().__init__(states, initial_state, events, transitions, rewards)
        
        def label_function(self, env_state):
            # This would depend on your environment
            # Here's a simple example
            if env_state == 'pickup_location':
                return 'pickup'
            elif env_state == 'delivery_location':
                return 'deliver'
            else:
                return 'nothing'
    
    # Initialize CROM
    n_agents = 2
    state_space_size = 10
    action_spaces = [4, 4]  # 4 actions per agent
    reward_machines = [SimpleRM(), SimpleRM()]
    
    crom = TabularCROM(
        n_agents=n_agents,
        state_space_size=state_space_size,
        action_spaces=action_spaces,
        reward_machines=reward_machines,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1
    )
    
    print("CROM initialized successfully!")
    print(f"Number of agents: {n_agents}")
    print(f"Action spaces: {action_spaces}")
    print(f"Ready to train with environment")

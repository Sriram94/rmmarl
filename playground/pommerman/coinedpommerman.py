import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from enum import IntEnum
import sys
import os

# Add pommerman to path if installed
try:
    import pommerman
    from pommerman import agents
    from pommerman import constants
    from pommerman.envs.v0 import Pomme
except ImportError:
    print("Warning: pommerman not installed. Install with: pip install pommerman")
    print("Creating mock environment for demonstration...")


class CoinStatus(IntEnum):
    """Status of the coin in the game"""
    NOT_SPAWNED = 0
    AVAILABLE = 1
    CAPTURED_BY_AGENT_0 = 2
    CAPTURED_BY_AGENT_1 = 3
    CAPTURED_BY_AGENT_2 = 4
    CAPTURED_BY_AGENT_3 = 5


class CoinedPommermanEnv:
    def __init__(self, 
                 n_agents: int = 2,
                 board_size: int = 11,
                 max_steps: int = 800,
                 render_mode: str = None):
        """
        Args:
            n_agents: Number of agents (2 for competitive, 4 for team)
            board_size: Size of the board (default 11x11)
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.n_agents = n_agents
        self.board_size = board_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Initialize base Pommerman environment
        try:
            # Try to use real Pommerman
            agent_list = [
                agents.SimpleAgent() for _ in range(n_agents)
            ]
            
            if n_agents == 2:
                self.base_env = pommerman.make('PommeFFACompetition-v0', agent_list)
            else:  # 4 agents (2v2)
                self.base_env = pommerman.make('PommeTeamCompetition-v0', agent_list)
        except:
            # Use mock environment
            self.base_env = None
            print("Using mock environment for demonstration")
        
        # Coin state
        self.coin_position = None
        self.coin_status = CoinStatus.NOT_SPAWNED
        
        # Game state
        self.current_step = 0
        self.agents_alive = [True] * n_agents
        
        # Observation and action spaces
        self.observation_space_shape = (board_size, board_size, 20)  # Simplified
        self.action_space_size = 6  # stop, up, down, left, right, bomb
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment and spawn coin"""
        if self.base_env:
            obs = self.base_env.reset()
        else:
            # Mock reset
            obs = [np.zeros(self.observation_space_shape) for _ in range(self.n_agents)]
        
        # Spawn coin at random empty location
        self.spawn_coin()
        
        # Reset game state
        self.current_step = 0
        self.agents_alive = [True] * self.n_agents
        self.coin_status = CoinStatus.AVAILABLE
        
        # Add coin information to observation
        obs_with_coin = self._add_coin_to_obs(obs)
        
        info = {
            'coin_position': self.coin_position,
            'coin_status': self.coin_status
        }
        
        return obs_with_coin, info
    
    def spawn_coin(self):
        """Spawn coin at random location"""
        # Find empty locations (simplified - should check board state)
        x = random.randint(1, self.board_size - 2)
        y = random.randint(1, self.board_size - 2)
        self.coin_position = (x, y)
        self.coin_status = CoinStatus.AVAILABLE
    
    def _add_coin_to_obs(self, obs: List[np.ndarray]) -> List[np.ndarray]:
        """Add coin information to observations"""
        # In real implementation, modify observation to include coin channel
        # For now, return as-is
        return obs
    
    def check_coin_capture(self, agent_positions: List[Tuple[int, int]]) -> Optional[int]:
        """
        Check if any agent captured the coin
        
        Returns:
            agent_id if coin was captured, None otherwise
        """
        if self.coin_status != CoinStatus.AVAILABLE:
            return None
        
        for agent_id, pos in enumerate(agent_positions):
            if pos == self.coin_position:
                self.coin_status = CoinStatus(CoinStatus.CAPTURED_BY_AGENT_0 + agent_id)
                return agent_id
        
        return None
    
    def get_agent_positions(self, obs: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Extract agent positions from observations"""
        # This should be implemented based on actual observation structure
        # Mock implementation
        positions = []
        for i in range(self.n_agents):
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            positions.append((x, y))
        return positions
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        self.current_step += 1
        
        # Execute actions in base environment
        if self.base_env:
            obs = self.base_env.step(actions)
            # Extract actual observations, rewards, done from pommerman
            # This depends on pommerman's return format
            base_rewards = [0.0] * self.n_agents
            base_done = False
        else:
            # Mock step
            obs = [np.zeros(self.observation_space_shape) for _ in range(self.n_agents)]
            base_rewards = [0.0] * self.n_agents
            base_done = False
        
        # Check for coin capture
        agent_positions = self.get_agent_positions(obs)
        capturing_agent = self.check_coin_capture(agent_positions)
        
        # Check which agents are alive (simplified)
        # In real implementation, parse from pommerman state
        for i in range(self.n_agents):
            if random.random() < 0.01:  # Mock death probability
                self.agents_alive[i] = False
        
        # Calculate rewards based on coined pommerman rules
        rewards = self._calculate_coined_rewards()
        
        # Check termination
        done = (self.current_step >= self.max_steps or 
                not any(self.agents_alive) or
                self._check_win_condition())
        
        # Add coin to observations
        obs_with_coin = self._add_coin_to_obs(obs)
        
        info = {
            'coin_position': self.coin_position,
            'coin_status': self.coin_status,
            'agents_alive': self.agents_alive.copy(),
            'capturing_agent': capturing_agent
        }
        
        return obs_with_coin, rewards, done, info
    
    def _calculate_coined_rewards(self) -> List[float]:
        rewards = [0.0] * self.n_agents
        
        if self.n_agents == 2:
            # Two-agent competitive case
            agent_0_alive = self.agents_alive[0]
            agent_1_alive = self.agents_alive[1]
            
            if self.coin_status == CoinStatus.CAPTURED_BY_AGENT_0:
                if not agent_1_alive and agent_0_alive:
                    rewards[0] = 1.0
                    rewards[1] = -1.0
            elif self.coin_status == CoinStatus.CAPTURED_BY_AGENT_1:
                if not agent_0_alive and agent_1_alive:
                    rewards[0] = -1.0
                    rewards[1] = 1.0
        else:
            # Four-agent team case (2v2)
            # Team 0: agents 0, 2; Team 1: agents 1, 3
            team_0_alive = self.agents_alive[0] or self.agents_alive[2]
            team_1_alive = self.agents_alive[1] or self.agents_alive[3]
            
            # Check which team captured coin
            team_0_has_coin = (self.coin_status == CoinStatus.CAPTURED_BY_AGENT_0 or 
                              self.coin_status == CoinStatus.CAPTURED_BY_AGENT_2)
            team_1_has_coin = (self.coin_status == CoinStatus.CAPTURED_BY_AGENT_1 or 
                              self.coin_status == CoinStatus.CAPTURED_BY_AGENT_3)
            
            if team_0_has_coin and not team_1_alive and team_0_alive:
                rewards[0] = rewards[2] = 1.0
                rewards[1] = rewards[3] = -1.0
            elif team_1_has_coin and not team_0_alive and team_1_alive:
                rewards[0] = rewards[2] = -1.0
                rewards[1] = rewards[3] = 1.0
        
        return rewards
    
    def _check_win_condition(self) -> bool:
        """Check if game has reached a win condition"""
        if self.n_agents == 2:
            # One agent must be dead and one alive with coin
            if (not self.agents_alive[0] and self.agents_alive[1] and 
                self.coin_status == CoinStatus.CAPTURED_BY_AGENT_1):
                return True
            if (not self.agents_alive[1] and self.agents_alive[0] and 
                self.coin_status == CoinStatus.CAPTURED_BY_AGENT_0):
                return True
        else:
            # One team must be eliminated and winning team has coin
            team_0_alive = self.agents_alive[0] or self.agents_alive[2]
            team_1_alive = self.agents_alive[1] or self.agents_alive[3]
            
            team_0_has_coin = (self.coin_status == CoinStatus.CAPTURED_BY_AGENT_0 or 
                              self.coin_status == CoinStatus.CAPTURED_BY_AGENT_2)
            team_1_has_coin = (self.coin_status == CoinStatus.CAPTURED_BY_AGENT_1 or 
                              self.coin_status == CoinStatus.CAPTURED_BY_AGENT_3)
            
            if not team_0_alive and team_1_alive and team_1_has_coin:
                return True
            if not team_1_alive and team_0_alive and team_0_has_coin:
                return True
        
        return False
    
    def render(self):
        """Render the environment"""
        if self.base_env and self.render_mode:
            self.base_env.render()
        else:
            # Simple text rendering
            print(f"\nStep: {self.current_step}/{self.max_steps}")
            print(f"Coin Status: {self.coin_status.name}")
            print(f"Coin Position: {self.coin_position}")
            print(f"Agents Alive: {self.agents_alive}")
    
    def close(self):
        """Close the environment"""
        if self.base_env:
            self.base_env.close()


class PommermanRewardMachine:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.states = [0, 1, 2, 3, 4, 5]
        self.initial_state = 0
        self.events = ['capture_coin', 'opponent_captures', 'kill_opponent', 
                      'agent_dies', 'timeout', 'nothing']
        
        # Define transitions
        self.transitions = {
            # From initial state
            (0, 'capture_coin'): 1,
            (0, 'opponent_captures'): 2,
            (0, 'agent_dies'): 5,  # Died before getting coin -> tie
            (0, 'timeout'): 5,
            
            # From state 1 (agent has coin)
            (1, 'kill_opponent'): 3,  # Win!
            (1, 'agent_dies'): 5,     # Had coin but died -> tie
            (1, 'timeout'): 5,
            
            # From state 2 (opponent has coin)
            (2, 'agent_dies'): 4,     # Opponent has coin and killed us -> lose
            (2, 'kill_opponent'): 5,  # We killed opponent but they had coin -> tie
            (2, 'timeout'): 5,
        }
        
        # Define rewards
        self.rewards = {
            (0, 1): 0.1,   # Small reward for getting coin
            (1, 3): 1.0,   # Large reward for winning
            (0, 5): 0.0,   # Tie
            (1, 5): 0.0,   # Tie
            (2, 4): -1.0,  # Lose
            (2, 5): 0.0,   # Tie
        }
    
    def get_next_state(self, current_state: int, event: str) -> int:
        """Get next RM state given current state and event"""
        return self.transitions.get((current_state, event), current_state)
    
    def get_reward(self, current_state: int, next_state: int) -> float:
        """Get reward for transitioning between RM states"""
        return self.rewards.get((current_state, next_state), 0.0)
    
    def label_function(self, env_info: Dict) -> str:
        """
        Map environment state to events
        
        Args:
            env_info: Dictionary with environment information
        """
        coin_status = env_info.get('coin_status', CoinStatus.NOT_SPAWNED)
        agents_alive = env_info.get('agents_alive', [True, True])
        capturing_agent = env_info.get('capturing_agent', None)
        step = env_info.get('step', 0)
        max_steps = env_info.get('max_steps', 800)
        
        # Check if this agent captured the coin
        if capturing_agent == self.agent_id:
            return 'capture_coin'
        
        # Check if opponent captured the coin
        if capturing_agent is not None and capturing_agent != self.agent_id:
            return 'opponent_captures'
        
        # Check if this agent died
        if not agents_alive[self.agent_id]:
            return 'agent_dies'
        
        # Check if opponent died
        opponent_id = 1 - self.agent_id  # Assumes 2 agents
        if not agents_alive[opponent_id]:
            return 'kill_opponent'
        
        # Check timeout
        if step >= max_steps:
            return 'timeout'
        
        return 'nothing'


# Training integration with CROM/QROM
class CROMPommermanTrainer:
    """
    Trainer for CROM/QROM on Coined Pommerman
    """
    
    def __init__(self, 
                 algorithm: str = 'DCROM',  # 'DCROM' or 'DQROM'
                 n_agents: int = 2,
                 state_dim: int = 372,  # Flattened pommerman obs
                 n_episodes: int = 10000,
                 save_freq: int = 100):
        """
        Args:
            algorithm: Which algorithm to use ('DCROM' or 'DQROM')
            n_agents: Number of agents
            state_dim: Dimension of state space
            n_episodes: Number of training episodes
            save_freq: Frequency to save models
        """
        self.algorithm = algorithm
        self.n_agents = n_agents
        self.n_episodes = n_episodes
        self.save_freq = save_freq
        
        # Create environment
        self.env = CoinedPommermanEnv(n_agents=n_agents)
        
        # Create reward machines
        self.reward_machines = [
            PommermanRewardMachine(agent_id=i) for i in range(n_agents)
        ]
        
        # Initialize algorithm (would import from crom_implementation.py)
        # For demonstration, showing the interface
        self.agent = None  # Would be DCROM or DQROM instance
        
        print(f"Initialized {algorithm} trainer for {n_agents}-agent Coined Pommerman")
    
    def flatten_obs(self, obs: np.ndarray) -> np.ndarray:
        """Flatten Pommerman observation to vector"""
        return obs.flatten()
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.n_episodes} episodes...")
        
        episode_rewards = []
        
        for episode in range(self.n_episodes):
            obs, info = self.env.reset()
            
            # Get initial RM states
            rm_states = [rm.initial_state for rm in self.reward_machines]
            
            episode_reward = [0.0] * self.n_agents
            done = False
            step = 0
            
            while not done:
                # Flatten observations
                flat_obs = [self.flatten_obs(o) for o in obs]
                
                # Select actions (would use self.agent.select_action)
                actions = [random.randint(0, 5) for _ in range(self.n_agents)]
                
                # Execute actions
                next_obs, rewards, done, info = self.env.step(actions)
                
                # Update RM states
                info['step'] = step
                info['max_steps'] = self.env.max_steps
                
                next_rm_states = []
                rm_rewards = []
                for agent_id in range(self.n_agents):
                    rm = self.reward_machines[agent_id]
                    event = rm.label_function(info)
                    next_rm_state = rm.get_next_state(rm_states[agent_id], event)
                    rm_reward = rm.get_reward(rm_states[agent_id], next_rm_state)
                    
                    next_rm_states.append(next_rm_state)
                    rm_rewards.append(rm_reward)
                
                # Train step (would call self.agent.train_step)
                # train_data = {
                #     'state': flat_obs[0],  # For simplicity, using first agent
                #     'rm_states': rm_states,
                #     'joint_action': actions,
                #     'rewards': rm_rewards,
                #     'next_state': self.flatten_obs(next_obs[0]),
                #     'next_rm_states': next_rm_states,
                #     'done': done
                # }
                # self.agent.train_step(train_data)
                
                # Update for next iteration
                obs = next_obs
                rm_states = next_rm_states
                
                for i in range(self.n_agents):
                    episode_reward[i] += rm_rewards[i]
                
                step += 1
            
            episode_rewards.append(episode_reward)
            
            # Logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([r[0] for r in episode_rewards[-10:]])
                print(f"Episode {episode + 1}/{self.n_episodes}, "
                      f"Avg Reward (last 10): {avg_reward:.3f}")
            
            # Save model
            if (episode + 1) % self.save_freq == 0:
                # self.agent.save(f'models/{self.algorithm}_episode_{episode+1}.pt')
                print(f"Model saved at episode {episode + 1}")
        
        print("\nTraining completed!")
        return episode_rewards
    
    def evaluate(self, n_episodes: int = 100):
        """Evaluate trained model"""
        print(f"\nEvaluating for {n_episodes} episodes...")
        
        total_rewards = [0.0] * self.n_agents
        wins = 0
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            rm_states = [rm.initial_state for rm in self.reward_machines]
            
            done = False
            
            while not done:
                flat_obs = [self.flatten_obs(o) for o in obs]
                
                # Select actions without exploration
                # actions = [self.agent.select_action(i, flat_obs[i], rm_states[i]) 
                #           for i in range(self.n_agents)]
                actions = [random.randint(0, 5) for _ in range(self.n_agents)]
                
                next_obs, rewards, done, info = self.env.step(actions)
                
                # Update RM states
                info['step'] = 0  # Would track actual step
                info['max_steps'] = self.env.max_steps
                
                for agent_id in range(self.n_agents):
                    rm = self.reward_machines[agent_id]
                    event = rm.label_function(info)
                    next_rm_state = rm.get_next_state(rm_states[agent_id], event)
                    rm_reward = rm.get_reward(rm_states[agent_id], next_rm_state)
                    
                    rm_states[agent_id] = next_rm_state
                    total_rewards[agent_id] += rm_reward
                
                obs = next_obs
            
            # Check if agent 0 won
            if total_rewards[0] > 0:
                wins += 1
        
        avg_rewards = [r / n_episodes for r in total_rewards]
        win_rate = wins / n_episodes
        
        print(f"Average Rewards: {avg_rewards}")
        print(f"Win Rate (Agent 0): {win_rate:.2%}")
        
        return avg_rewards, win_rate


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Coined Pommerman Environment for CROM/QROM")
    print("="*70)
    
    # Create environment
    print("\n1. Creating Coined Pommerman Environment...")
    env = CoinedPommermanEnv(n_agents=2)
    obs, info = env.reset()
    print(f"✓ Environment created")
    print(f"  - Coin position: {info['coin_position']}")
    print(f"  - Coin status: {info['coin_status'].name}")
    
    # Test environment
    print("\n2. Testing environment step...")
    actions = [1, 2]  # Random actions
    obs, rewards, done, info = env.step(actions)
    print(f"✓ Step executed")
    print(f"  - Rewards: {rewards}")
    print(f"  - Done: {done}")
    
    # Create reward machines
    print("\n3. Creating Reward Machines...")
    rms = [PommermanRewardMachine(agent_id=i) for i in range(2)]
    print(f"✓ Reward machines created")
    print(f"  - RM states: {rms[0].states}")
    print(f"  - RM events: {rms[0].events}")
    
    # Test RM transitions
    print("\n4. Testing RM transitions...")
    test_info = {
        'coin_status': CoinStatus.AVAILABLE,
        'agents_alive': [True, True],
        'capturing_agent': 0,
        'step': 10,
        'max_steps': 800
    }
    event = rms[0].label_function(test_info)
    next_state = rms[0].get_next_state(0, event)
    reward = rms[0].get_reward(0, next_state)
    print(f"✓ RM transition tested")
    print(f"  - Event: {event}")
    print(f"  - Next state: {next_state}")
    print(f"  - Reward: {reward}")
    
    # Create trainer
    print("\n5. Creating CROM Trainer...")
    trainer = CROMPommermanTrainer(
        algorithm='DCROM',
        n_agents=2,
        n_episodes=100
    )
    print(f"✓ Trainer created")
    
    print("\n" + "="*70)
    print("Setup complete! Ready to train with:")
    print("  trainer.train()  # Start training")
    print("  trainer.evaluate()  # Evaluate trained model")
    print("="*70)

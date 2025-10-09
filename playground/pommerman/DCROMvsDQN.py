
import numpy as np
import torch
import random
import os
import json
from collections import deque
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from DCROM import DCROMAgent, RewardMachine as DCROMRewardMachine
from DQN import DQNCrossProductAgent, RewardMachine as DQNRewardMachine

from coined_pommerman import CoinedPommermanEnv, CoinStatus

class CoinStatus:
    NOT_SPAWNED = 0
    AVAILABLE = 1
    CAPTURED_BY_AGENT_0 = 2
    CAPTURED_BY_AGENT_1 = 3


class SimplifiedPommermanEnv:
    def __init__(self, board_size=11, max_steps=800):
        self.board_size = board_size
        self.max_steps = max_steps
        self.state_dim = board_size * board_size + 10  
        self.n_actions = 6  
        self.reset()
    
    def reset(self):
        self.step_count = 0
        self.coin_position = (random.randint(0, self.board_size-1), 
                             random.randint(0, self.board_size-1))
        self.coin_status = CoinStatus.AVAILABLE
        self.agent_positions = [
            (random.randint(0, self.board_size-1), random.randint(0, self.board_size-1)),
            (random.randint(0, self.board_size-1), random.randint(0, self.board_size-1))
        ]
        self.agents_alive = [True, True]
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        obs = []
        for agent_id in range(2):
            board = np.random.rand(self.board_size * self.board_size) * 0.1
            extra = np.array([
                self.agent_positions[agent_id][0] / self.board_size,
                self.agent_positions[agent_id][1] / self.board_size,
                self.coin_position[0] / self.board_size,
                self.coin_position[1] / self.board_size,
                float(self.coin_status),
                float(self.agents_alive[agent_id]),
                float(self.agents_alive[1-agent_id]),
                self.step_count / self.max_steps,
                0.0, 0.0  
            ])
            obs_vector = np.concatenate([board, extra]).astype(np.float32)
            obs.append(obs_vector)
        return obs
    
    def _get_info(self):
        return {
            'coin_position': self.coin_position,
            'coin_status': self.coin_status,
            'agents_alive': self.agents_alive.copy(),
            'agent_positions': self.agent_positions.copy(),
            'step': self.step_count,
            'max_steps': self.max_steps
        }
    
    def step(self, actions):
        self.step_count += 1
        
        for agent_id, action in enumerate(actions):
            if not self.agents_alive[agent_id]:
                continue
            x, y = self.agent_positions[agent_id]
            if action == 1:  # up
                y = max(0, y - 1)
            elif action == 2:  # down
                y = min(self.board_size - 1, y + 1)
            elif action == 3:  # left
                x = max(0, x - 1)
            elif action == 4:  # right
                x = min(self.board_size - 1, x + 1)
            self.agent_positions[agent_id] = (x, y)
        
        capturing_agent = None
        if self.coin_status == CoinStatus.AVAILABLE:
            for agent_id in range(2):
                if self.agent_positions[agent_id] == self.coin_position:
                    self.coin_status = CoinStatus.CAPTURED_BY_AGENT_0 + agent_id
                    capturing_agent = agent_id
                    break
        
        if random.random() < 0.005:  
            victim = random.randint(0, 1)
            self.agents_alive[victim] = False
        
        done = (self.step_count >= self.max_steps or 
                not any(self.agents_alive))
        
        obs = self._get_obs()
        
        rewards = [0.0, 0.0]
        
        info = self._get_info()
        info['capturing_agent'] = capturing_agent
        
        return obs, rewards, done, info


def build_pommerman_rm_dcrom(agent_id: int) -> DCROMRewardMachine:
    n_states = 6
    
    delta = {
        # From state 0 (initial)
        (0, 0): 0,  # nothing
        (0, 1): 1,  # capture coin
        (0, 2): 2,  # opponent captures
        (0, 4): 5,  # agent dies before coin -> tie
        (0, 5): 5,  # timeout
        
        # From state 1 (agent has coin)
        (1, 0): 1,  # nothing
        (1, 3): 3,  # kill opponent -> WIN
        (1, 4): 5,  # agent dies -> tie
        (1, 5): 5,  # timeout
        
        # From state 2 (opponent has coin)
        (2, 0): 2,  # nothing
        (2, 3): 5,  # kill opponent but they had coin -> tie
        (2, 4): 4,  # agent dies -> LOSE
        (2, 5): 5,  # timeout
        
        # Terminal states self-loop
        (3, 0): 3, (3, 1): 3, (3, 2): 3, (3, 3): 3, (3, 4): 3, (3, 5): 3,
        (4, 0): 4, (4, 1): 4, (4, 2): 4, (4, 3): 4, (4, 4): 4, (4, 5): 4,
        (5, 0): 5, (5, 1): 5, (5, 2): 5, (5, 3): 5, (5, 4): 5, (5, 5): 5,
    }
    
    # Define rewards: (state, next_state) -> reward
    sigma = {
        (0, 1): 0.0,   # No reward for getting coin
        (1, 3): 1.0,   # Big reward for winning
        (0, 5): -1.0,   # Tie
        (1, 5): -1.0,   # Tie
        (2, 4): -1.0,  # Penalty for losing
        (2, 5): -1.0,   # Tie
    }
    
    # Define label function
    def label_fn(env_state):
        if not isinstance(env_state, dict):
            return 0  # nothing
        
        coin_status = env_state.get('coin_status', CoinStatus.NOT_SPAWNED)
        agents_alive = env_state.get('agents_alive', [True, True])
        capturing_agent = env_state.get('capturing_agent', None)
        step = env_state.get('step', 0)
        max_steps = env_state.get('max_steps', 800)
        
        if capturing_agent == agent_id:
            return 1  
        
        if capturing_agent is not None and capturing_agent != agent_id:
            return 2  
        
        if not agents_alive[agent_id]:
            return 4  
        
        opponent_id = 1 - agent_id
        if not agents_alive[opponent_id]:
            return 3  
        
        if step >= max_steps:
            return 5  
        
        return 0  
    
    terminal_states = [3, 4, 5]
    
    return DCROMRewardMachine(n_states, delta, sigma, label_fn, terminal_states)


def build_pommerman_rm_dqn(agent_id: int) -> DQNRewardMachine:
    n_states = 6
    
    delta = {
        (0, 0): 0,  # nothing
        (0, 1): 1,  # capture coin
        (0, 2): 2,  # opponent captures
        (0, 4): 5,  # agent dies before coin -> tie
        (0, 5): 5,  # timeout
        
        # From state 1 (agent has coin)
        (1, 0): 1,  # nothing
        (1, 3): 3,  # kill opponent -> WIN
        (1, 4): 5,  # agent dies -> tie
        (1, 5): 5,  # timeout
        
        # From state 2 (opponent has coin)
        (2, 0): 2,  # nothing
        (2, 3): 5,  # kill opponent but they had coin -> tie
        (2, 4): 4,  # agent dies -> LOSE
        (2, 5): 5,  # timeout
        
        # Terminal states self-loop
        (3, 0): 3, (3, 1): 3, (3, 2): 3, (3, 3): 3, (3, 4): 3, (3, 5): 3,
        (4, 0): 4, (4, 1): 4, (4, 2): 4, (4, 3): 4, (4, 4): 4, (4, 5): 4,
        (5, 0): 5, (5, 1): 5, (5, 2): 5, (5, 3): 5, (5, 4): 5, (5, 5): 5,
    }
    
    sigma = {
        (0, 1): 0.1,   # Small reward for getting coin
        (1, 3): 1.0,   # Big reward for winning
        (0, 5): 0.0,   # Tie
        (1, 5): 0.0,   # Tie
        (2, 4): -1.0,  # Penalty for losing
        (2, 5): 0.0,   # Tie
    }
    
    def label_fn(env_state):
        if not isinstance(env_state, dict):
            return 0  # nothing
        
        coin_status = env_state.get('coin_status', CoinStatus.NOT_SPAWNED)
        agents_alive = env_state.get('agents_alive', [True, True])
        capturing_agent = env_state.get('capturing_agent', None)
        step = env_state.get('step', 0)
        max_steps = env_state.get('max_steps', 800)
        
        if capturing_agent == agent_id:
            return 1  # capture_coin
        
        if capturing_agent is not None and capturing_agent != agent_id:
            return 2  # opponent_captures
        
        if not agents_alive[agent_id]:
            return 4  # agent_dies
        
        opponent_id = 1 - agent_id
        if not agents_alive[opponent_id]:
            return 3  # kill_opponent
        
        if step >= max_steps:
            return 5  # timeout
        
        return 0  # nothing
    
    terminal_states = [3, 4, 5]
    
    return DQNRewardMachine(n_states, delta, sigma, label_fn, terminal_states)


class PommermanDCROMvsDQNTrainer:
    
    def __init__(self,
                 env,
                 dcrom_agent: DCROMAgent,  # Agent 0 uses DCROM
                 dqn_agent: DQNCrossProductAgent,  # Agent 1 uses DQN
                 n_episodes: int = 10000,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 save_dir: str = './checkpoints',
                 log_freq: int = 10):
        
        self.env = env
        self.dcrom_agent = dcrom_agent
        self.dqn_agent = dqn_agent
        self.n_episodes = n_episodes
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.save_dir = save_dir
        self.log_freq = log_freq
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_rewards = {0: [], 1: []}  # 0=DCROM, 1=DQN
        self.episode_lengths = []
        self.win_counts = {0: 0, 1: 0, 'tie': 0}  # 0=DCROM wins, 1=DQN wins
        
        self.prev_dqn_action = 0
    
    def train(self):
        print("=" * 70)
        print(f"Starting Training: DCROM (Agent 0) vs DQN (Agent 1)")
        print(f"Episodes: {self.n_episodes}")
        print("=" * 70)
        
        for episode in range(self.n_episodes):
            episode_reward, episode_length, winner = self._run_episode()
            
            self.episode_rewards[0].append(episode_reward[0])
            self.episode_rewards[1].append(episode_reward[1])
            self.episode_lengths.append(episode_length)
            if winner == 0:
                self.win_counts[0] += 1
            elif winner == 1:
                self.win_counts[1] += 1
            else:
                self.win_counts['tie'] += 1
            
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % self.log_freq == 0:
                self._log_progress(episode)
            
            if (episode + 1) % 100 == 0:
                self._save_checkpoint(episode)
        
        print("\n" + "=" * 70)
        print("Training completed!")
        print("=" * 70)
        self._final_report()
    
    def _run_episode(self) -> Tuple[List[float], int, int]:
        obs_list, info = self.env.reset()
        
        rm_states = [0, 0]  # Both start at initial state
        
        episode_rewards = [0.0, 0.0]
        episode_length = 0
        done = False
        
        self.prev_dqn_action = 0
        
        while not done:
            dcrom_obs = obs_list[0]
            dcrom_rm_state = rm_states[0]
            dcrom_action = self.dcrom_agent.select_action(
                dcrom_obs, dcrom_rm_state, self.prev_dqn_action, eps=self.epsilon
            )
            
            dqn_obs = obs_list[1]
            dqn_rm_state = rm_states[1]
            dqn_action = self.dqn_agent.select_action(
                dqn_obs, dqn_rm_state, eps=self.epsilon
            )
            
            actions = [dcrom_action, dqn_action]
            
            next_obs_list, env_rewards, done, info = self.env.step(actions)
            
            dcrom_next_rm_state = self.dcrom_agent.rm.next_state(rm_states[0], info)
            dcrom_rm_reward = self.dcrom_agent.rm.reward(rm_states[0], dcrom_next_rm_state)
            
            dqn_next_rm_state = self.dqn_agent.rm.next_state(rm_states[1], info)
            dqn_rm_reward = self.dqn_agent.rm.reward(rm_states[1], dqn_next_rm_state)
            
            self.dcrom_agent.store_transition(
                s=obs_list[0],
                u=rm_states[0],
                a=dcrom_action,
                r=dcrom_rm_reward,
                s_next=next_obs_list[0],
                u_next=dcrom_next_rm_state,
                done=done,
                prev_op_action_idx=self.prev_dqn_action,
                op_action_idx=dqn_action
            )
            
            self.dqn_agent.store_transition(
                s=obs_list[1],
                u=rm_states[1],
                a=dqn_action,
                r=dqn_rm_reward,
                s_next=next_obs_list[1],
                u_next=dqn_next_rm_state,
                done=done
            )
            
            self.dcrom_agent.train_step()
            self.dqn_agent.train_step()
            
            obs_list = next_obs_list
            rm_states = [dcrom_next_rm_state, dqn_next_rm_state]
            self.prev_dqn_action = dqn_action
            
            episode_rewards[0] += dcrom_rm_reward
            episode_rewards[1] += dqn_rm_reward
            episode_length += 1
        
        winner = -1  # tie
        if episode_rewards[0] > episode_rewards[1]:
            winner = 0  # DCROM wins
        elif episode_rewards[1] > episode_rewards[0]:
            winner = 1  # DQN wins
        
        return episode_rewards, episode_length, winner
    
    def _log_progress(self, episode: int):
        window = min(100, episode + 1)
        avg_reward_dcrom = np.mean(self.episode_rewards[0][-window:])
        avg_reward_dqn = np.mean(self.episode_rewards[1][-window:])
        avg_length = np.mean(self.episode_lengths[-window:])
        
        total_games = episode + 1
        dcrom_win_rate = self.win_counts[0] / total_games
        dqn_win_rate = self.win_counts[1] / total_games
        tie_rate = self.win_counts['tie'] / total_games
        
        print(f"\nEpisode {episode + 1}/{self.n_episodes}")
        print(f"  Epsilon: {self.epsilon:.3f}")
        print(f"  Avg Reward (last {window}):")
        print(f"    DCROM (Agent 0): {avg_reward_dcrom:.3f}")
        print(f"    DQN (Agent 1):   {avg_reward_dqn:.3f}")
        print(f"  Avg Episode Length: {avg_length:.1f}")
        print(f"  Win Rates:")
        print(f"    DCROM: {dcrom_win_rate:.2%}")
        print(f"    DQN:   {dqn_win_rate:.2%}")
        print(f"    Tie:   {tie_rate:.2%}")
    
    def _save_checkpoint(self, episode: int):
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_ep{episode+1}.pt')
        torch.save({
            'episode': episode,
            'dcrom_q_net': self.dcrom_agent.q_net.state_dict(),
            'dcrom_opponent_model': self.dcrom_agent.opponent_model.state_dict(),
            'dqn_q_net': self.dqn_agent.q_net.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'win_counts': self.win_counts
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def _final_report(self):
        total_games = self.n_episodes
        print(f"\nFinal Statistics:")
        print(f"  Total Episodes: {total_games}")
        print(f"\n  DCROM (Agent 0):")
        print(f"    Wins: {self.win_counts[0]} ({self.win_counts[0]/total_games:.2%})")
        print(f"    Avg Reward: {np.mean(self.episode_rewards[0]):.3f}")
        print(f"\n  DQN (Agent 1):")
        print(f"    Wins: {self.win_counts[1]} ({self.win_counts[1]/total_games:.2%})")
        print(f"    Avg Reward: {np.mean(self.episode_rewards[1]):.3f}")
        print(f"\n  Ties: {self.win_counts['tie']} ({self.win_counts['tie']/total_games:.2%})")
        print(f"  Avg Episode Length: {np.mean(self.episode_lengths):.1f}")
        
        if self.win_counts[0] > self.win_counts[1]:
            advantage = (self.win_counts[0] - self.win_counts[1]) / total_games
            print(f"\n  ✓ DCROM outperforms DQN by {advantage:.2%}")
        elif self.win_counts[1] > self.win_counts[0]:
            advantage = (self.win_counts[1] - self.win_counts[0]) / total_games
            print(f"\n  ✗ DQN outperforms DCROM by {advantage:.2%}")
        else:
            print(f"\n  = Both algorithms perform equally")
    
    def plot_training_curves(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        window = 100
        if len(self.episode_rewards[0]) >= window:
            smoothed_dcrom = np.convolve(self.episode_rewards[0], 
                                         np.ones(window)/window, mode='valid')
            smoothed_dqn = np.convolve(self.episode_rewards[1], 
                                       np.ones(window)/window, mode='valid')
            axes[0, 0].plot(smoothed_dcrom, label='DCROM (Agent 0)', linewidth=2)
            axes[0, 0].plot(smoothed_dqn, label='DQN (Agent 1)', linewidth=2)
            axes[0, 0].set_title(f'Smoothed Rewards (window={window})', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        if len(self.episode_lengths) >= window:
            smoothed_lengths = np.convolve(self.episode_lengths, 
                                          np.ones(window)/window, mode='valid')
            axes[0, 1].plot(smoothed_lengths, color='purple', linewidth=2)
            axes[0, 1].set_title(f'Smoothed Episode Length (window={window})', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].grid(True, alpha=0.3)
        
        episodes = list(range(1, len(self.episode_rewards[0]) + 1))
        cumulative_dcrom = []
        cumulative_dqn = []
        wins_dcrom, wins_dqn = 0, 0
        for i, (r0, r1) in enumerate(zip(self.episode_rewards[0], self.episode_rewards[1])):
            if r0 > r1:
                wins_dcrom += 1
            elif r1 > r0:
                wins_dqn += 1
            cumulative_dcrom.append(wins_dcrom / (i + 1))
            cumulative_dqn.append(wins_dqn / (i + 1))
        
        axes[1, 0].plot(episodes, cumulative_dcrom, label='DCROM', linewidth=2, color='blue')
        axes[1, 0].plot(episodes, cumulative_dqn, label='DQN', linewidth=2, color='orange')
        axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (even)')
        axes[1, 0].set_title('Cumulative Win Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        axes[1, 1].hist(self.episode_rewards[0], alpha=0.6, label='DCROM', bins=30, color='blue')
        axes[1, 1].hist(self.episode_rewards[1], alpha=0.6, label='DQN', bins=30, color='orange')
        axes[1, 1].set_title('Reward Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('DCROM vs DQN Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        plt.show()0
        self.agent_1 = agent_1
        self.n_episodes = n_episodes
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.save_dir = save_dir
        self.log_freq = log_freq
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_rewards = {0: [], 1: []}
        self.episode_lengths = []
        self.win_counts = {0: 0, 1: 0, 'tie': 0}
        
        self.prev_op_actions = [0, 0]
    
    def train(self):
        print("=" * 70)
        print(f"Starting DCROM Training on Coined Pommerman")
        print(f"Episodes: {self.n_episodes}")
        print("=" * 70)
        
        for episode in range(self.n_episodes):
            episode_reward, episode_length, winner = self._run_episode()
            
            self.episode_rewards[0].append(episode_reward[0])
            self.episode_rewards[1].append(episode_reward[1])
            self.episode_lengths.append(episode_length)
            if winner == 0:
                self.win_counts[0] += 1
            elif winner == 1:
                self.win_counts[1] += 1
            else:
                self.win_counts['tie'] += 1
            
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % self.log_freq == 0:
                self._log_progress(episode)
            
            if (episode + 1) % 100 == 0:
                self._save_checkpoint(episode)
        
        print("\n" + "=" * 70)
        print("Training completed!")
        print("=" * 70)
        self._final_report()
    
    def _run_episode(self) -> Tuple[List[float], int, int]:
        obs_list, info = self.env.reset()
        
        rm_states = [0, 0]  
        
        episode_rewards = [0.0, 0.0]
        episode_length = 0
        done = False
        
        self.prev_op_actions = [0, 0]
        
        while not done:
            actions = []
            for agent_id in range(2):
                agent = self.agent_0 if agent_id == 0 else self.agent_1
                obs = obs_list[agent_id]
                rm_state = rm_states[agent_id]
                prev_op_action = self.prev_op_actions[1 - agent_id]
                
                action = agent.select_action(obs, rm_state, prev_op_action, eps=self.epsilon)
                actions.append(action)
            
            next_obs_list, env_rewards, done, info = self.env.step(actions)
            
            next_rm_states = []
            rm_rewards = []
            for agent_id in range(2):
                agent = self.agent_0 if agent_id == 0 else self.agent_1
                rm = agent.rm
                
                next_rm_state = rm.next_state(rm_states[agent_id], info)
                next_rm_states.append(next_rm_state)
                
                rm_reward = rm.reward(rm_states[agent_id], next_rm_state)
                rm_rewards.append(rm_reward)
            
            for agent_id in range(2):
                agent = self.agent_0 if agent_id == 0 else self.agent_1
                opponent_id = 1 - agent_id
                
                agent.store_transition(
                    s=obs_list[agent_id],
                    u=rm_states[agent_id],
                    a=actions[agent_id],
                    r=rm_rewards[agent_id],
                    s_next=next_obs_list[agent_id],
                    u_next=next_rm_states[agent_id],
                    done=done,
                    prev_op_action_idx=self.prev_op_actions[opponent_id],
                    op_action_idx=actions[opponent_id]
                )
            
            self.agent_0.train_step()
            self.agent_1.train_step()
            
            obs_list = next_obs_list
            rm_states = next_rm_states
            self.prev_op_actions = actions
            
            episode_rewards[0] += rm_rewards[0]
            episode_rewards[1] += rm_rewards[1]
            episode_length += 1
        
        winner = -1  
        if episode_rewards[0] > episode_rewards[1]:
            winner = 0
        elif episode_rewards[1] > episode_rewards[0]:
            winner = 1
        
        return episode_rewards, episode_length, winner
    
    def _log_progress(self, episode: int):
        window = min(100, episode + 1)
        avg_reward_0 = np.mean(self.episode_rewards[0][-window:])
        avg_reward_1 = np.mean(self.episode_rewards[1][-window:])
        avg_length = np.mean(self.episode_lengths[-window:])
        
        total_games = episode + 1
        win_rate_0 = self.win_counts[0] / total_games
        win_rate_1 = self.win_counts[1] / total_games
        tie_rate = self.win_counts['tie'] / total_games
        
        print(f"\nEpisode {episode + 1}/{self.n_episodes}")
        print(f"  Epsilon: {self.epsilon:.3f}")
        print(f"  Avg Reward (last {window}): Agent 0: {avg_reward_0:.3f}, Agent 1: {avg_reward_1:.3f}")
        print(f"  Avg Episode Length: {avg_length:.1f}")
        print(f"  Win Rates: Agent 0: {win_rate_0:.2%}, Agent 1: {win_rate_1:.2%}, Tie: {tie_rate:.2%}")
    
    def _save_checkpoint(self, episode: int):
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_ep{episode+1}.pt')
        torch.save({
            'episode': episode,
            'agent_0_q_net': self.agent_0.q_net.state_dict(),
            'agent_0_opponent_model': self.agent_0.opponent_model.state_dict(),
            'agent_1_q_net': self.agent_1.q_net.state_dict(),
            'agent_1_opponent_model': self.agent_1.opponent_model.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'win_counts': self.win_counts
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def _final_report(self):
        total_games = self.n_episodes
        print(f"\nFinal Statistics:")
        print(f"  Total Episodes: {total_games}")
        print(f"  Agent 0 Wins: {self.win_counts[0]} ({self.win_counts[0]/total_games:.2%})")
        print(f"  Agent 1 Wins: {self.win_counts[1]} ({self.win_counts[1]/total_games:.2%})")
        print(f"  Ties: {self.win_counts['tie']} ({self.win_counts['tie']/total_games:.2%})")
        print(f"  Avg Reward Agent 0: {np.mean(self.episode_rewards[0]):.3f}")
        print(f"  Avg Reward Agent 1: {np.mean(self.episode_rewards[1]):.3f}")
        print(f"  Avg Episode Length: {np.mean(self.episode_lengths):.1f}")
    
    def plot_training_curves(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        window = 100
        if len(self.episode_rewards[0]) >= window:
            smoothed_0 = np.convolve(self.episode_rewards[0], 
                                     np.ones(window)/window, mode='valid')
            smoothed_1 = np.convolve(self.episode_rewards[1], 
                                     np.ones(window)/window, mode='valid')
            axes[0, 0].plot(smoothed_0, label='Agent 0')
            axes[0, 0].plot(smoothed_1, label='Agent 1')
            axes[0, 0].set_title(f'Smoothed Rewards (window={window})')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        if len(self.episode_lengths) >= window:
            smoothed_lengths = np.convolve(self.episode_lengths, 
                                          np.ones(window)/window, mode='valid')
            axes[0, 1].plot(smoothed_lengths)
            axes[0, 1].set_title(f'Smoothed Episode Length (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].grid(True)
        
        episodes = list(range(1, len(self.episode_rewards[0]) + 1))
        cumulative_0 = []
        cumulative_1 = []
        wins_0, wins_1 = 0, 0
        for i, (r0, r1) in enumerate(zip(self.episode_rewards[0], self.episode_rewards[1])):
            if r0 > r1:
                wins_0 += 1
            elif r1 > r0:
                wins_1 += 1
            cumulative_0.append(wins_0 / (i + 1))
            cumulative_1.append(wins_1 / (i + 1))
        
        axes[1, 0].plot(episodes, cumulative_0, label='Agent 0')
        axes[1, 0].plot(episodes, cumulative_1, label='Agent 1')
        axes[1, 0].set_title('Cumulative Win Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].hist(self.episode_rewards[0], alpha=0.5, label='Agent 0', bins=30)
        axes[1, 1].hist(self.episode_rewards[1], alpha=0.5, label='Agent 1', bins=30)
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Training curves saved to {save_path}")
        plt.show()


def main():
    print("Initializing Coined Pommerman with DCROM...")
    
    env = SimplifiedPommermanEnv(board_size=11, max_steps=800)
    obs_dim = env.state_dim
    n_actions = env.n_actions
    
    print(f"Environment created: obs_dim={obs_dim}, n_actions={n_actions}")
    
    rm_0 = build_pommerman_rm(agent_id=0)
    rm_1 = build_pommerman_rm(agent_id=1)
    
    print("Reward machines created")
    
    n_op_actions = n_actions  
    
    agent_0 = DCROMAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        rm=rm_0,
        n_op_actions=n_op_actions,
        prev_op_action_dim=n_op_actions,
        lr=1e-3,
        gamma=0.99,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    )
    
    agent_1 = DCROMAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        rm=rm_1,
        n_op_actions=n_op_actions,
        prev_op_action_dim=n_op_actions,
        lr=1e-3,
        gamma=0.99,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    )
    
    print("DCROM agents created")
    
    trainer = PommermanDCROMTrainer(
        env=env,
        agent_0=agent_0,
        agent_1=agent_1,
        n_episodes=1000,  
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        save_dir='./checkpoints',
        log_freq=10
    )
    
    print("Trainer initialized\n")
    
    trainer.train()
    
    trainer.plot_training_curves(save_path='./training_curves.png')


if __name__ == '__main__':
    main()

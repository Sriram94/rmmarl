""" dcrom.py

Deep CROM (DCROM) implementation.


Key classes:
- RewardMachine: simple automaton interface (states, delta, sigma, labeling function)
- OpponentModel: a small MLP predicting opponent action distribution
- QNetwork: shared Q-network that conditions on env-state, RM-state, and opponent-action
- ReplayBuffer: standard replay buffer
- DCROMAgent: the agent implementing training with counterfactual updates over RM states

"""

import random
from typing import Callable, List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Reward Machine (simple API)
# -----------------------------

class RewardMachine:
    # Simple Reward Machine representation.

    """States are integers 0..(n_states-1). The RM exposes:
      - delta(u, event) -> u'   (transition function)
      - sigma(u, u') -> reward  (output reward function)
      - label_fn(state) -> event (labelling function mapping env-state -> event)
    For portability we accept arbitrary python objects as events (e.g., strings or ints).
    """

    def __init__(self, n_states: int, delta: Dict[Tuple[int, Any], int],
                 sigma: Dict[Tuple[int, int], float],
                 label_fn: Callable[[Any], Any],
                 terminal_states: List[int] = None):
        self.n_states = n_states
        self.delta = delta
        self.sigma = sigma
        self.label_fn = label_fn
        self.terminal_states = set(terminal_states or [])

    def next_state(self, u: int, env_state: Any):
        #Given current RM state u and environment state, return u' = delta(u, label(env_state)).
        event = self.label_fn(env_state)
        return self.delta.get((u, event), u)  # default: self-loop if not specified

    def reward(self, u: int, u_next: int) -> float:
        return self.sigma.get((u, u_next), 0.0)

    def is_terminal(self, u: int) -> bool:
        return u in self.terminal_states

# -----------------------------
# Opponent model
# -----------------------------
class OpponentModel(nn.Module):
    def __init__(self, obs_dim: int, rm_state_dim: int, prev_op_action_dim: int, hidden_sizes=[64, 64, 64, 64], n_op_actions=10):
        super().__init__()
        self.n_op_actions = n_op_actions
        layers = []
        in_dim = obs_dim + rm_state_dim + prev_op_action_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Softmax(nn.Linear(in_dim, n_op_actions)))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, rm_onehot: torch.Tensor, prev_op_action_onehot: torch.Tensor):
        x = torch.cat([obs, rm_onehot, prev_op_action_onehot], dim=-1)
        logits = self.net(x)
        return logits  

# -----------------------------
# Q-network
# -----------------------------
class QNetwork(nn.Module):
    #Shared Q-network that takes (obs, rm_onehot, predicted_op_action_onehot) and outputs Q-values for own actions.
    def __init__(self, obs_dim: int, rm_state_dim: int, op_action_dim: int, n_actions: int, hidden_sizes=[1024, 1024, 1024, 1024, 1024, 1024]):
        super().__init__()
        in_dim = obs_dim + rm_state_dim + op_action_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, rm_onehot: torch.Tensor, op_action_onehot: torch.Tensor):
        x = torch.cat([obs, rm_onehot, op_action_onehot], dim=-1)
        q = self.net(x)
        return q

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition: Tuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# DCROM Agent
# -----------------------------
class DCROMAgent:
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 rm: RewardMachine,
                 n_op_actions: int,
                 prev_op_action_dim: int,
                 lr: float = 0.01,
                 gamma: float = 0.9,
                 buffer_size: int = int(2e7),
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 device: torch.device = device):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rm = rm
        self.n_rm_states = rm.n_states
        self.n_op_actions = n_op_actions
        self.prev_op_action_dim = prev_op_action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        # opponent model: predicts opponents' joint-action id
        self.opponent_model = OpponentModel(obs_dim, self.n_rm_states, prev_op_action_dim, n_op_actions=n_op_actions).to(device)
        # shared Q-network and target
        self.q_net = QNetwork(obs_dim, self.n_rm_states, n_op_actions, n_actions).to(device)
        self.q_target = QNetwork(obs_dim, self.n_rm_states, n_op_actions, n_actions).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        # optimizers
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.op_optimizer = torch.optim.Adam(self.opponent_model.parameters(), lr=lr)

        # replay
        self.replay = ReplayBuffer(buffer_size)

        # training bookkeeping
        self.update_count = 0
        self.target_update_freq = target_update_freq

        # opponent model loss
        self.opponent_loss_fn = nn.CrossEntropyLoss()

    def obs_to_tensor(self, obs):
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def one_hot(self, indices: np.ndarray, depth: int):
        arr = np.zeros((len(indices), depth), dtype=np.float32)
        for i, idx in enumerate(indices):
            arr[i, int(idx)] = 1.0
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def select_action(self, obs: np.ndarray, rm_state: int, prev_op_action_idx: int, eps: float = 0.1):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        rm_oh = torch.zeros((1, self.n_rm_states), dtype=torch.float32, device=self.device)
        rm_oh[0, rm_state] = 1.0
        prev_op_oh = torch.zeros((1, self.n_op_actions), dtype=torch.float32, device=self.device)
        prev_op_oh[0, prev_op_action_idx] = 1.0

        # predict opponent action logits -> take argmax as predicted
        with torch.no_grad():
            op_logits = self.opponent_model(obs_t, rm_oh, prev_op_oh)
            op_pred = torch.argmax(op_logits, dim=-1)
            op_pred_oh = torch.zeros((1, self.n_op_actions), device=self.device)
            op_pred_oh[0, op_pred.item()] = 1.0

            qvals = self.q_net(obs_t, rm_oh, op_pred_oh)
            if random.random() < eps:
                return random.randrange(self.n_actions)
            else:
                return int(torch.argmax(qvals, dim=-1).item())

    def store_transition(self, s, u, a, r, s_next, u_next, done, prev_op_action_idx, op_action_idx):
        # transition format: s, u, a, r, s_next, u_next, done, prev_op_action_idx, op_action_idx
        self.replay.push((s, u, a, r, s_next, u_next, float(done), prev_op_action_idx, op_action_idx))

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None
        # sample batch
        s_batch, u_batch, a_batch, r_batch, s_next_batch, u_next_batch, done_batch, prev_op_batch, op_act_batch = self.replay.sample(self.batch_size)
        # convert to tensors
        s_batch = torch.tensor(np.vstack(s_batch), dtype=torch.float32, device=self.device)
        s_next_batch = torch.tensor(np.vstack(s_next_batch), dtype=torch.float32, device=self.device)
        u_batch = torch.tensor(u_batch, dtype=torch.long, device=self.device)
        u_next_batch = torch.tensor(u_next_batch, dtype=torch.long, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.long, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)
        prev_op_batch = torch.tensor(prev_op_batch, dtype=torch.long, device=self.device)
        op_act_batch = torch.tensor(op_act_batch, dtype=torch.long, device=self.device)

        # prepare one-hots
        rm_onehot = F.one_hot(u_batch, num_classes=self.n_rm_states).float()
        rm_next_onehot = F.one_hot(u_next_batch, num_classes=self.n_rm_states).float()
        prev_op_onehot = F.one_hot(prev_op_batch, num_classes=self.n_op_actions).float()

        # -------------------------
        # Train opponent model
        # -------------------------
        op_logits = self.opponent_model(s_batch, rm_onehot, prev_op_onehot)
        op_loss = self.opponent_loss_fn(op_logits, op_act_batch.to(self.device))
        self.op_optimizer.zero_grad()
        op_loss.backward()
        self.op_optimizer.step()

        # -------------------------
        # Counterfactual Q-updates (vectorized-ish)
        # For each possible RM state u_cf in 0..n_rm_states-1, compute what the reward would have been
        # and the next RM state given the observed s_next. Then compute targets and update Q.
        # -------------------------
        batch_size = s_batch.shape[0]
        n_rm = self.n_rm_states

        # We will compute target Q for each element in the batch by aggregating counterfactual rewards.
        # For stability, we apply standard DQN-style target: r_cf + gamma * max_a' Q_target(s', u'_cf, predicted_op_action)
        # Then average over the counterfactual updates (as in CRM/DCROM: update all q(s,u_cf,a) using the
        # experience with reward determined by counterfactual).
        # Implementation note: we compute a target for each (batch, u_cf) and then update the q-values
        # corresponding to the actual u in the batch (i.e., for q(s, u_actual, a) we update using the
        # target computed for that u_actual). Additionally, we also update q(s, u_cf, a) for *all* u_cf
        # (counterfactual) by writing into the same batch — this mirrors lines 8--14 of Algorithm 1.
        #
        # For simplicity we will:
        # - compute targets_cf[batch, u_cf]
        # - compute current_q_vals for (s_batch, each u_cf, prev_op_onehot_predicted) and gather the q for taken actions
        # - compute loss = MSE(current_q_taken, target_cf_for_that_u)
        #
        # Predict opponent action logits for s_next for each hypothetical rm state (u_cf_next).
        targets = torch.zeros((batch_size, n_rm), dtype=torch.float32, device=self.device)
        # For predicted opponent action used in next-state input to Q, we use the opponent model
        # with s_next and rm_next_cf (onehot)
        with torch.no_grad():
            for u_cf in range(n_rm):
                # compute u_cf_next for each sample using RM.next_state semantics
                # We can't call RM.next_state vectorized; we'll call per-sample. RM.label_fn expects env-state objects.
                u_cf_next_list = []
                r_cf_list = []
                for i in range(batch_size):
                    # compute label(event) on the observed s_next_batch (here we pass raw array to RM.label_fn)
                    s_next_np = s_next_batch[i].cpu().numpy()
                    # rrm expects original env-state object; in this prototype we assume label_fn accepts numpy arrays
                    u_cf_next = self.rm.next_state(u_cf, s_next_np)
                    r_cf = self.rm.reward(u_cf, u_cf_next)
                    u_cf_next_list.append(u_cf_next)
                    r_cf_list.append(r_cf)
                u_cf_next_arr = torch.tensor(u_cf_next_list, dtype=torch.long, device=self.device)
                r_cf_tensor = torch.tensor(r_cf_list, dtype=torch.float32, device=self.device)

                # opponent prediction for next state given u_cf_next
                rm_next_oh = F.one_hot(u_cf_next_arr, num_classes=n_rm).float()
                # for prev opponent action when predicting next opponent action, we will use the observed op action in batch
                prev_op_next_oh = F.one_hot(op_act_batch, num_classes=self.n_op_actions).float()
                op_logits_next = self.opponent_model(s_next_batch, rm_next_oh, prev_op_next_oh)
                op_pred_next = torch.argmax(op_logits_next, dim=-1)
                op_pred_next_oh = F.one_hot(op_pred_next, num_classes=self.n_op_actions).float()

                # compute q_target(s_next, u_cf_next, op_pred_next)
                q_next = self.q_target(s_next_batch, rm_next_oh, op_pred_next_oh)  # shape [B, n_actions]
                max_q_next, _ = q_next.max(dim=1)
                target_cf = r_cf_tensor + (1.0 - done_batch) * (self.gamma * max_q_next)
                targets[:, u_cf] = target_cf

        # Now compute current Q estimates for every counterfactual rm state u_cf using the current q_net.
        # We'll compute loss that encourages q(s, u_cf, a_taken) to move toward targets[:, u_cf]
        q_losses = []
        for u_cf in range(n_rm):
            rm_cf_oh = F.one_hot(torch.full((batch_size,), u_cf, dtype=torch.long, device=self.device),
                                 num_classes=n_rm).float()
            # opponent prediction for current state given rm_cf
            op_logits_now = self.opponent_model(s_batch, rm_cf_oh, prev_op_onehot)
            op_pred_now = torch.argmax(op_logits_now, dim=-1)
            op_pred_now_oh = F.one_hot(op_pred_now, num_classes=self.n_op_actions).float()

            q_vals = self.q_net(s_batch, rm_cf_oh, op_pred_now_oh)  # [B, n_actions]
            # gather q-values for the actions actually taken in the batch
            q_taken = q_vals.gather(1, a_batch.unsqueeze(1)).squeeze(1)
            target_for_u = targets[:, u_cf].detach()
            loss_u = F.mse_loss(q_taken, target_for_u)
            q_losses.append(loss_u)

        loss_q = torch.stack(q_losses).mean()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_optimizer.step()

        # soft/hard update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return {'q_loss': loss_q.item(), 'op_loss': op_loss.item()}

# -----------------------------
# Example usage / dummy run
# -----------------------------
def _dummy_label_fn(env_state):
    #A toy label function that maps a numeric array to an event integer for RM transitions.
    # example: sum(env_state) thresholded
    s = np.array(env_state)
    ssum = float(s.sum())
    if ssum < 0.0:
        return 0
    elif ssum < 1.0:
        return 1
    else:
        return 2

def build_dummy_rm():
    # small RM with 3 states, events {0,1,2}
    n_states = 3
    delta = {
        # (current_state, event): next_state
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (1, 0): 1,
        (1, 1): 1,
        (1, 2): 2,
        (2, 0): 2,
        (2, 1): 2,
        (2, 2): 2,
    }
    sigma = {
        (0,1): 0.0,
        (1,2): 1.0,  # reward when reaching state 2 from 1
    }
    return RewardMachine(n_states, delta, sigma, _dummy_label_fn, terminal_states=[2])

def dummy_env_step(action):
    #Return next_obs, done. Here we craft a small random next-observation vector.
    obs = np.random.randn(4).astype(np.float32)
    done = random.random() < 0.05
    return obs, done

if __name__ == '__main__':
    # hyperparams for the dummy run
    obs_dim = 4
    n_actions = 5
    n_op_actions = 6
    prev_op_action_dim = n_op_actions
    rm = build_dummy_rm()
    agent = DCROMAgent(obs_dim, n_actions, rm, n_op_actions, prev_op_action_dim,
                       lr=1e-3, gamma=0.99, buffer_size=2000, batch_size=32, target_update_freq=50)

    # populate replay buffer with random transitions
    for _ in range(500):
        s = np.random.randn(obs_dim).astype(np.float32)
        u = random.randrange(rm.n_states)
        a = random.randrange(n_actions)
        s_next, done = dummy_env_step(a)
        u_next = rm.next_state(u, s_next)
        r = rm.reward(u, u_next)
        prev_op = random.randrange(n_op_actions)
        op_act = random.randrange(n_op_actions)
        agent.store_transition(s, u, a, r, s_next, u_next, done, prev_op, op_act)

    # run some training steps
    for step in range(200):
        metrics = agent.train_step()
        if metrics:
            if step % 20 == 0:
                print(f\"Step {step}: q_loss={metrics['q_loss']:.4f}, op_loss={metrics['op_loss']:.4f}")


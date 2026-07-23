from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from .base import Agent
from .networks import CrossProductQNet, PerRMStateQNet
from .opponent_model import OpponentModel
from .replay_buffer import ReplayBuffer, Experience
from .common import one_hot, one_hot_batch, one_hot_concat, epsilon_greedy

class _CrossProductBase(Agent):
    uses_counterfactual = False

    def __init__(self, obs_dim: int, n_rm_states: int, n_actions: int, extra_input_dim: int=0, opp_action_dim: int=0, device: str='cpu', lr: float=0.01, gamma: float=0.9, buffer_size: int=200000, target_update_every: int=100, seed: int=0):
        self.obs_dim = obs_dim
        self.n_rm_states = n_rm_states
        self.n_actions = n_actions
        self.extra_input_dim = extra_input_dim
        self.opp_action_dim = opp_action_dim
        self.gamma = gamma
        self.device = device
        self.target_update_every = target_update_every
        self.rng = np.random.default_rng(seed)
        self._train_steps = 0
        self.buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        eff_obs_dim = obs_dim + extra_input_dim
        self.q_eval = CrossProductQNet(eff_obs_dim, n_rm_states, n_actions, opp_action_dim=opp_action_dim).to(device)
        self.q_target = CrossProductQNet(eff_obs_dim, n_rm_states, n_actions, opp_action_dim=opp_action_dim).to(device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.optim = torch.optim.Adam(self.q_eval.parameters(), lr=lr)

    def _augment(self, obs_vec: np.ndarray) -> np.ndarray:
        return obs_vec

    def _predicted_opp_action_onehot(self, obs_vec, u_index):
        return None

    def _opp_actions_to_onehot(self, opp_actions: Dict[str, int]) -> np.ndarray:
        raise NotImplementedError

    def _predict_next_opp_onehot_batch(self, s_next: torch.Tensor, u_next_oh: torch.Tensor, opp_action_batch: torch.Tensor):
        return None

    def act(self, obs_vec, u_index, epsilon) -> int:
        s = torch.tensor(self._augment(obs_vec), dtype=torch.float32, device=self.device).unsqueeze(0)
        u_oh = one_hot_batch(torch.tensor([u_index]), self.n_rm_states).to(self.device)
        opp_oh_np = self._predicted_opp_action_onehot(obs_vec, u_index)
        opp_oh = None if opp_oh_np is None else torch.tensor(opp_oh_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_eval(s, u_oh, opp_oh).squeeze(0)
        return epsilon_greedy(q, epsilon, self.rng)

    def remember(self, s, u_idx, a, r, s_next, u_next_idx, done, cf_targets=None, u_to_index=None, opp_action=None) -> None:
        if self.opp_action_dim > 0:
            assert opp_action is not None, f"{self.variant} has an opponent model and requires the ACTUAL opponent action(s) taken at this step (Line 8's 'a', from the rollout) -- pass opp_action."
            opp_action_oh = self._opp_actions_to_onehot(opp_action)
        else:
            opp_action_oh = None
        self.buffer.push(Experience(state=self._augment(s), u_index=u_idx, action=a, reward=r, next_state=self._augment(s_next), next_u_index=u_next_idx, done=done, opp_action=opp_action_oh))

    def train(self, batch_size: int=64) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        s = torch.tensor(np.stack([e.state for e in batch]), dtype=torch.float32, device=self.device)
        u_idx = torch.tensor([e.u_index for e in batch], dtype=torch.long, device=self.device)
        a = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        r = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        s_next = torch.tensor(np.stack([e.next_state for e in batch]), dtype=torch.float32, device=self.device)
        u_next_idx = torch.tensor([e.next_u_index for e in batch], dtype=torch.long, device=self.device)
        done = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        u_oh = one_hot_batch(u_idx, self.n_rm_states)
        u_next_oh = one_hot_batch(u_next_idx, self.n_rm_states)
        if self.opp_action_dim > 0:
            opp_action = torch.tensor(np.stack([e.opp_action for e in batch]), dtype=torch.float32, device=self.device)
        else:
            opp_action = None
        q_sa = self.q_eval(s, u_oh, opp_action).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_opp_action = self._predict_next_opp_onehot_batch(s_next, u_next_oh, opp_action)
            q_next = self.q_target(s_next, u_next_oh, next_opp_action).max(dim=1).values
            y = r + self.gamma * (1 - done) * q_next
        loss = F.mse_loss(q_sa, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
        return loss.item()

class DQNBaseline(_CrossProductBase):
    variant = 'DQN'
    has_opponent_model = False

class DQNOMBaseline(_CrossProductBase):
    variant = 'DQN-OM'
    has_opponent_model = True

    def __init__(self, *args, opponent_ids: List[str], n_opponent_actions: int, **kwargs):
        self.opponent_ids = opponent_ids
        self.n_opponent_actions = n_opponent_actions
        opp_action_dim = len(opponent_ids) * n_opponent_actions
        super().__init__(*args, opp_action_dim=opp_action_dim, **kwargs)
        self.opponent_models: Dict[str, OpponentModel] = {oid: OpponentModel(self.obs_dim, self.n_rm_states, n_opponent_actions).to(self.device) for oid in opponent_ids}
        self.opponent_optims = {oid: torch.optim.Adam(m.parameters(), lr=0.01) for oid, m in self.opponent_models.items()}
        self._prev_opponent_actions = {oid: 0 for oid in opponent_ids}

    def _opp_actions_to_onehot(self, opp_actions: Dict[str, int]) -> np.ndarray:
        indices = [opp_actions.get(oid, 0) for oid in self.opponent_ids]
        return one_hot_concat(indices, self.n_opponent_actions)

    def _predicted_opp_action_onehot(self, obs_vec, u_index):
        predicted = self.predict_opponent_actions(obs_vec, u_index)
        return self._opp_actions_to_onehot(predicted)

    def _predict_next_opp_onehot_batch(self, s_next, u_next_oh, opp_action_batch):
        parts = []
        for i, oid in enumerate(self.opponent_ids):
            model = self.opponent_models[oid]
            prev_a_oh = opp_action_batch[:, i * self.n_opponent_actions:(i + 1) * self.n_opponent_actions]
            logits = model.forward(s_next, u_next_oh, prev_a_oh)
            parts.append(one_hot_batch(logits.argmax(dim=-1), self.n_opponent_actions))
        return torch.cat(parts, dim=-1)

    def predict_opponent_actions(self, obs_vec, u_index) -> Dict[str, int]:
        state_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_oh = torch.tensor(one_hot(u_index, self.n_rm_states), device=self.device).unsqueeze(0)
        preds = {}
        for oid, model in self.opponent_models.items():
            prev_a_oh = torch.tensor(one_hot(self._prev_opponent_actions[oid], self.n_opponent_actions), device=self.device).unsqueeze(0)
            preds[oid] = model.predict_action(state_t, u_oh, prev_a_oh, sample=True).item()
        return preds

    def observe_opponents(self, obs_vec, u_index, joint_actions, prev_opponent_actions) -> None:
        state_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_oh = torch.tensor(one_hot(u_index, self.n_rm_states), device=self.device).unsqueeze(0)
        for oid, model in self.opponent_models.items():
            if oid not in joint_actions:
                continue
            prev_a_oh = torch.tensor(one_hot(prev_opponent_actions.get(oid, 0), self.n_opponent_actions), device=self.device).unsqueeze(0)
            target = torch.tensor([joint_actions[oid]], dtype=torch.long, device=self.device)
            loss = model.loss(state_t, u_oh, prev_a_oh, target)
            opt = self.opponent_optims[oid]
            opt.zero_grad()
            loss.backward()
            opt.step()
        self._prev_opponent_actions.update({oid: joint_actions[oid] for oid in self.opponent_ids if oid in joint_actions})

class MADQNBaseline(_CrossProductBase):
    variant = 'MA-DQN'
    has_opponent_model = False

    def __init__(self, *args, opponent_ids: List[str], n_opponent_actions: int, **kwargs):
        self.opponent_ids = opponent_ids
        self.n_opponent_actions = n_opponent_actions
        extra_dim = len(opponent_ids) * n_opponent_actions
        super().__init__(*args, extra_input_dim=extra_dim, **kwargs)
        self._prev_opponent_actions = {oid: 0 for oid in opponent_ids}

    def _augment(self, obs_vec: np.ndarray) -> np.ndarray:
        extra = np.concatenate([one_hot(self._prev_opponent_actions[oid], self.n_opponent_actions) for oid in self.opponent_ids]) if self.opponent_ids else np.zeros(0, dtype=np.float32)
        return np.concatenate([obs_vec, extra]).astype(np.float32)

    def observe_opponents(self, obs_vec, u_index, joint_actions, prev_opponent_actions) -> None:
        self._prev_opponent_actions.update({oid: joint_actions[oid] for oid in self.opponent_ids if oid in joint_actions})

class DQRMBaseline(Agent):
    variant = 'DQRM'
    uses_counterfactual = True
    has_opponent_model = False

    def __init__(self, obs_dim: int, n_rm_states: int, n_actions: int, device: str='cpu', lr: float=0.01, gamma: float=0.9, buffer_size: int=200000, target_update_every: int=100, seed: int=0):
        self.obs_dim, self.n_rm_states, self.n_actions = (obs_dim, n_rm_states, n_actions)
        self.gamma, self.device = (gamma, device)
        self.target_update_every = target_update_every
        self.rng = np.random.default_rng(seed)
        self._train_steps = 0
        self.buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        self.q_eval = PerRMStateQNet(obs_dim, n_rm_states, n_actions).to(device)
        self.q_target = PerRMStateQNet(obs_dim, n_rm_states, n_actions).to(device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.optim = torch.optim.Adam(self.q_eval.parameters(), lr=lr)

    def act(self, obs_vec, u_index, epsilon) -> int:
        s = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_eval(s, u_index).squeeze(0)
        return epsilon_greedy(q, epsilon, self.rng)

    def remember(self, s, u_idx, a, r, s_next, u_next_idx, done, cf_targets=None, u_to_index=None) -> None:
        assert cf_targets is not None and u_to_index is not None
        exps = []
        for u, (u_next, reward) in cf_targets.items():
            exps.append(Experience(state=s, u_index=u_to_index(u), action=a, reward=reward, next_state=s_next, next_u_index=u_to_index(u_next), done=False))
        self.buffer.push_many(exps)

    def train(self, batch_size: int=64) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        s = torch.tensor(np.stack([e.state for e in batch]), dtype=torch.float32, device=self.device)
        u_idx = torch.tensor([e.u_index for e in batch], dtype=torch.long, device=self.device)
        a = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        r = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        s_next = torch.tensor(np.stack([e.next_state for e in batch]), dtype=torch.float32, device=self.device)
        u_next_idx = torch.tensor([e.next_u_index for e in batch], dtype=torch.long, device=self.device)
        done = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        q_sa = self.q_eval.forward_batched(s, u_idx).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target.forward_batched(s_next, u_next_idx).max(dim=1).values
            y = r + self.gamma * (1 - done) * q_next
        loss = F.mse_loss(q_sa, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
        return loss.item()

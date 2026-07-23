from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from .base import Agent
from .networks import CrossProductQNet, PerRMStateQNet
from .opponent_model import OpponentModel
from .replay_buffer import ReplayBuffer, Experience
from .common import one_hot, one_hot_batch, one_hot_concat, epsilon_greedy

class RMAgent(Agent):
    variant = 'base'
    uses_counterfactual = True
    has_opponent_model = True

    def __init__(self, agent_id: str, obs_dim: int, n_rm_states: int, n_actions: int, opponent_ids: List[str], n_opponent_actions: int, device: str='cpu', lr: float=0.01, gamma: float=0.9, buffer_size: int=200000, target_update_every: int=100, seed: int=0):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.n_rm_states = n_rm_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.target_update_every = target_update_every
        self.rng = np.random.default_rng(seed)
        self._train_steps = 0
        self._q_lr = lr
        self.buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        self.opponent_ids = opponent_ids
        self.n_opponent_actions = n_opponent_actions
        self.opp_action_dim = len(opponent_ids) * n_opponent_actions
        self.opponent_models: Dict[str, OpponentModel] = {oid: OpponentModel(obs_dim, n_rm_states, n_opponent_actions).to(device) for oid in opponent_ids}
        self.opponent_optims = {oid: torch.optim.Adam(m.parameters(), lr=lr) for oid, m in self.opponent_models.items()}
        self._prev_opponent_actions = {oid: 0 for oid in opponent_ids}

    def _opp_actions_to_onehot(self, opp_actions: Dict[str, int]) -> np.ndarray:
        indices = [opp_actions.get(oid, 0) for oid in self.opponent_ids]
        return one_hot_concat(indices, self.n_opponent_actions)

    def _q_online(self, state_t: torch.Tensor, u_idx, opp_action_oh=None) -> torch.Tensor:
        raise NotImplementedError

    def _q_target(self, state_t: torch.Tensor, u_idx, opp_action_oh=None) -> torch.Tensor:
        raise NotImplementedError

    def _optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def _maybe_update_target(self):
        raise NotImplementedError

    def act(self, obs_vec: np.ndarray, u_index: int, epsilon: float) -> int:
        state_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.opponent_ids:
            predicted = self.predict_opponent_actions(obs_vec, u_index)
            opp_oh = torch.tensor(self._opp_actions_to_onehot(predicted), dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            opp_oh = None
        with torch.no_grad():
            q = self._q_online(state_t, u_index, opp_oh).squeeze(0)
        return epsilon_greedy(q, epsilon, self.rng)

    def predict_opponent_actions(self, obs_vec: np.ndarray, u_index: int) -> Dict[str, int]:
        state_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_oh = torch.tensor(one_hot(u_index, self.n_rm_states), device=self.device).unsqueeze(0)
        preds = {}
        for oid, model in self.opponent_models.items():
            prev_a_oh = torch.tensor(one_hot(self._prev_opponent_actions[oid], self.n_opponent_actions), device=self.device).unsqueeze(0)
            a = model.predict_action(state_t, u_oh, prev_a_oh, sample=True).item()
            preds[oid] = a
        return preds

    def update_prev_opponent_actions(self, actions: Dict[str, int]):
        for oid in self.opponent_ids:
            if oid in actions:
                self._prev_opponent_actions[oid] = actions[oid]

    def _predict_next_opp_onehot_batch(self, s_next: torch.Tensor, u_next_oh: torch.Tensor, opp_action_batch: torch.Tensor):
        if not self.opponent_ids or self.opp_action_dim == 0:
            return None
        parts = []
        for i, oid in enumerate(self.opponent_ids):
            model = self.opponent_models[oid]
            prev_a_oh = opp_action_batch[:, i * self.n_opponent_actions:(i + 1) * self.n_opponent_actions]
            logits = model.forward(s_next, u_next_oh, prev_a_oh)
            pred_idx = logits.argmax(dim=-1)
            parts.append(one_hot_batch(pred_idx, self.n_opponent_actions))
        return torch.cat(parts, dim=-1)

    def store_counterfactual(self, s: np.ndarray, a: int, s_next: np.ndarray, cf_targets: Dict, u_to_index, opp_action_oh: np.ndarray) -> int:
        exps = []
        for u, (u_next, r) in cf_targets.items():
            done = False
            exps.append(Experience(state=s, u_index=u_to_index(u), action=a, reward=r, next_state=s_next, next_u_index=u_to_index(u_next), done=done, opp_action=opp_action_oh))
        self.buffer.push_many(exps)
        return len(exps)

    def remember(self, s, u_idx, a, r, s_next, u_next_idx, done, cf_targets=None, u_to_index=None, opp_action=None) -> None:
        assert cf_targets is not None and u_to_index is not None, f'{self.variant} requires cf_targets/u_to_index (uses_counterfactual=True)'
        if self.opp_action_dim > 0:
            assert opp_action is not None, f"{self.variant} has an opponent model and requires the ACTUAL opponent action(s) taken at this step (Line 8's 'a', from the rollout) -- pass opp_action."
            opp_action_oh = self._opp_actions_to_onehot(opp_action)
        else:
            opp_action_oh = np.zeros(0, dtype=np.float32)
        self.store_counterfactual(s, a, s_next, cf_targets, u_to_index, opp_action_oh)

    def observe_opponents(self, obs_vec, u_index, joint_actions, prev_opponent_actions) -> None:
        self.train_opponent_models(obs_vec, u_index, joint_actions, prev_opponent_actions)
        self.update_prev_opponent_actions(joint_actions)

    def train(self, batch_size: int=64):
        return self.train_q(batch_size=batch_size)

    def train_opponent_models(self, s: np.ndarray, u_index: int, actions: Dict[str, int], prev_actions: Dict[str, int]):
        state_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_oh = torch.tensor(one_hot(u_index, self.n_rm_states), device=self.device).unsqueeze(0)
        for oid, model in self.opponent_models.items():
            if oid not in actions:
                continue
            prev_a_oh = torch.tensor(one_hot(prev_actions.get(oid, 0), self.n_opponent_actions), device=self.device).unsqueeze(0)
            target = torch.tensor([actions[oid]], dtype=torch.long, device=self.device)
            loss = model.loss(state_t, u_oh, prev_a_oh, target)
            opt = self.opponent_optims[oid]
            opt.zero_grad()
            loss.backward()
            opt.step()

    def train_q(self, batch_size: int=64):
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        loss_val = self._train_step(batch)
        self._train_steps += 1
        self._maybe_update_target()
        return loss_val

    def _batch_tensors(self, batch):
        s = torch.tensor(np.stack([e.state for e in batch]), dtype=torch.float32, device=self.device)
        u_idx = torch.tensor([e.u_index for e in batch], dtype=torch.long, device=self.device)
        a = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        r = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        s_next = torch.tensor(np.stack([e.next_state for e in batch]), dtype=torch.float32, device=self.device)
        u_next_idx = torch.tensor([e.next_u_index for e in batch], dtype=torch.long, device=self.device)
        done = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        if self.opp_action_dim > 0:
            opp_action = torch.tensor(np.stack([e.opp_action for e in batch]), dtype=torch.float32, device=self.device)
        else:
            opp_action = None
        return (s, u_idx, a, r, s_next, u_next_idx, done, opp_action)

class DCROMAgent(RMAgent):
    variant = 'DCROM'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_eval = CrossProductQNet(self.obs_dim, self.n_rm_states, self.n_actions, opp_action_dim=self.opp_action_dim).to(self.device)
        self.q_target = CrossProductQNet(self.obs_dim, self.n_rm_states, self.n_actions, opp_action_dim=self.opp_action_dim).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.optim = torch.optim.Adam(self.q_eval.parameters(), lr=self._q_lr)

    def _q_online(self, state_t, u_idx, opp_action_oh=None):
        u_oh = one_hot_batch(torch.tensor([u_idx]), self.n_rm_states).to(self.device)
        return self.q_eval(state_t, u_oh, opp_action_oh)

    def _train_step(self, batch):
        s, u_idx, a, r, s_next, u_next_idx, done, opp_action = self._batch_tensors(batch)
        u_oh = one_hot_batch(u_idx, self.n_rm_states)
        u_next_oh = one_hot_batch(u_next_idx, self.n_rm_states)
        q_sa = self.q_eval(s, u_oh, opp_action).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_opp_action = self._predict_next_opp_onehot_batch(s_next, u_next_oh, opp_action)
            q_next = self.q_target(s_next, u_next_oh, next_opp_action).max(dim=1).values
            y = r + self.gamma * (1 - done) * q_next
        loss = F.mse_loss(q_sa, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def _maybe_update_target(self):
        if self._train_steps % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

class DQROMAgent(RMAgent):
    variant = 'DQROM'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_eval = PerRMStateQNet(self.obs_dim, self.n_rm_states, self.n_actions, opp_action_dim=self.opp_action_dim).to(self.device)
        self.q_target = PerRMStateQNet(self.obs_dim, self.n_rm_states, self.n_actions, opp_action_dim=self.opp_action_dim).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.optim = torch.optim.Adam(self.q_eval.parameters(), lr=self._q_lr)

    def _q_online(self, state_t, u_idx, opp_action_oh=None):
        return self.q_eval(state_t, u_idx, opp_action_oh)

    def _train_step(self, batch):
        s, u_idx, a, r, s_next, u_next_idx, done, opp_action = self._batch_tensors(batch)
        q_sa = self.q_eval.forward_batched(s, u_idx, opp_action).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            u_next_oh = one_hot_batch(u_next_idx, self.n_rm_states)
            next_opp_action = self._predict_next_opp_onehot_batch(s_next, u_next_oh, opp_action)
            q_next = self.q_target.forward_batched(s_next, u_next_idx, next_opp_action).max(dim=1).values
            y = r + self.gamma * (1 - done) * q_next
        loss = F.mse_loss(q_sa, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def _maybe_update_target(self):
        if self._train_steps % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

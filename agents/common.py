from __future__ import annotations
import torch
import numpy as np

def one_hot(index: int, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v

def one_hot_batch(indices: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices.long(), num_classes=dim).float()

def one_hot_concat(action_indices, dim: int) -> np.ndarray:
    if len(action_indices) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([one_hot(int(a), dim) for a in action_indices]).astype(np.float32)

def epsilon_greedy(q_values: torch.Tensor, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q_values.shape[0]))
    return int(torch.argmax(q_values).item())

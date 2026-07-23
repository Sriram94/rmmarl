from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0)]
PLACE_BOMB, STAY = (4, 5)
N_ACTIONS = 6
RIGID, WOOD, EMPTY = (1, 2, 0)

@dataclass
class Bomb:
    pos: Tuple[int, int]
    owner: int
    timer: int
    blast_range: int

@dataclass
class PommermanState:
    grid: np.ndarray
    agent_pos: np.ndarray
    agent_alive: np.ndarray
    agent_bomb_count: np.ndarray
    agent_max_bombs: np.ndarray
    agent_blast_range: np.ndarray
    bombs: List[Bomb]
    powerups: Dict[Tuple[int, int], str]
    coin_pos: Optional[Tuple[int, int]]
    coin_captured_by_team: Optional[int]
    t: int
    captured_by_team: Optional[int] = None
    just_died: List[int] = field(default_factory=list)

class PommermanEnv:

    def __init__(self, n_per_team: int=1, grid_size: int=15, min_steps: int=800, max_steps_range: int=200, bomb_timer: int=8, blast_range: int=2, wood_density: float=0.3, seed: int=None):
        assert n_per_team in (1, 2)
        self.n_per_team = n_per_team
        self.n_agents = 2 * n_per_team
        self.grid_size = grid_size
        self.min_steps = min_steps
        self.max_steps_range = max_steps_range
        self.max_steps = min_steps
        self.bomb_timer = bomb_timer
        self.blast_range = blast_range
        self.wood_density = wood_density
        self.rng = np.random.default_rng(seed)
        team0 = [f'agent_0_{i}' for i in range(n_per_team)]
        team1 = [f'agent_1_{i}' for i in range(n_per_team)]
        self.agent_ids = team0 + team1
        self.team_of = {aid: 0 for aid in team0}
        self.team_of.update({aid: 1 for aid in team1})
        self.state: PommermanState = None

    def _team_alive(self, state, team: int) -> bool:
        return any((state.agent_alive[i] for i, aid in enumerate(self.agent_ids) if self.team_of[aid] == team))

    def _build_grid(self) -> np.ndarray:
        g = np.full((self.grid_size, self.grid_size), EMPTY, dtype=np.int8)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = RIGID
        g[2:-2:2, 2:-2:2] = RIGID
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                if g[x, y] == EMPTY and self.rng.random() < self.wood_density:
                    g[x, y] = WOOD
        return g

    def reset(self) -> PommermanState:
        self.max_steps = int(self.rng.integers(self.min_steps, self.min_steps + self.max_steps_range + 1))
        grid = self._build_grid()
        corners = [(1, 1), (1, self.grid_size - 2), (self.grid_size - 2, 1), (self.grid_size - 2, self.grid_size - 2)]
        positions = []
        for i in range(self.n_agents):
            cx, cy = corners[i % 4]
            grid[cx, cy] = EMPTY
            positions.append([cx, cy])
        agent_pos = np.array(positions, dtype=np.int32)
        coin_pos = None
        for _ in range(200):
            cx = self.rng.integers(1, self.grid_size - 1)
            cy = self.rng.integers(1, self.grid_size - 1)
            if grid[cx, cy] == EMPTY and (not any((np.array_equal([cx, cy], p) for p in positions))):
                coin_pos = (int(cx), int(cy))
                break
        self.state = PommermanState(grid=grid, agent_pos=agent_pos, agent_alive=np.ones(self.n_agents, dtype=bool), agent_bomb_count=np.ones(self.n_agents, dtype=np.int32), agent_max_bombs=np.ones(self.n_agents, dtype=np.int32), agent_blast_range=np.full(self.n_agents, self.blast_range, dtype=np.int32), bombs=[], powerups={}, coin_pos=coin_pos, coin_captured_by_team=None, t=0)
        return self.state

    def observe(self, agent_id: str, state: PommermanState) -> np.ndarray:
        idx = self.agent_ids.index(agent_id)
        own_pos = state.agent_pos[idx].astype(np.float32)
        alive = np.array([1.0 if state.agent_alive[idx] else 0.0], dtype=np.float32)
        others_rel = []
        for j in range(self.n_agents):
            if j == idx:
                continue
            rel = (state.agent_pos[j] - state.agent_pos[idx]).astype(np.float32)
            same_team = 1.0 if self.team_of[self.agent_ids[j]] == self.team_of[agent_id] else 0.0
            others_rel.append(np.concatenate([rel, [same_team], [1.0 if state.agent_alive[j] else 0.0]]))
        others_rel = np.concatenate(others_rel) if others_rel else np.zeros(0, dtype=np.float32)
        if state.coin_pos is not None:
            coin_rel = (np.array(state.coin_pos) - state.agent_pos[idx]).astype(np.float32)
            coin_present = np.array([1.0], dtype=np.float32)
        else:
            coin_rel = np.zeros(2, dtype=np.float32)
            coin_present = np.array([0.0], dtype=np.float32)
        bomb_map = np.zeros_like(state.grid, dtype=np.float32)
        for b in state.bombs:
            bomb_map[b.pos[0], b.pos[1]] = 1.0
        powerup_map = np.zeros_like(state.grid, dtype=np.float32)
        for (px, py), kind in state.powerups.items():
            powerup_map[px, py] = 1.0 if kind == 'extra_bomb' else 2.0
        full_map = state.grid.astype(np.float32) + bomb_map * 3.0 + powerup_map * 5.0
        bomb_count = np.array([state.agent_bomb_count[idx]], dtype=np.float32)
        blast_range = np.array([state.agent_blast_range[idx]], dtype=np.float32)
        return np.concatenate([own_pos, alive, others_rel, coin_rel, coin_present, full_map.flatten(), bomb_count, blast_range]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        others = (self.n_agents - 1) * 4
        return 2 + 1 + others + 2 + 1 + self.grid_size * self.grid_size + 1 + 1

    def step(self, actions: Dict[str, int]):
        s = self.state
        s.captured_by_team = None
        s.just_died = []
        for idx, aid in enumerate(self.agent_ids):
            if not s.agent_alive[idx]:
                continue
            a = actions[aid]
            if a == PLACE_BOMB:
                if s.agent_bomb_count[idx] > 0 and (not any((b.pos == tuple(s.agent_pos[idx]) for b in s.bombs))):
                    s.bombs.append(Bomb(pos=tuple(s.agent_pos[idx]), owner=idx, timer=self.bomb_timer, blast_range=int(s.agent_blast_range[idx])))
                    s.agent_bomb_count[idx] -= 1
            elif a != STAY:
                dx, dy = ACTION_DELTAS[a]
                nx, ny = (s.agent_pos[idx][0] + dx, s.agent_pos[idx][1] + dy)
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (s.grid[nx, ny] == EMPTY) and (not any((b.pos == (nx, ny) for b in s.bombs))):
                    s.agent_pos[idx] = [nx, ny]
                    picked = s.powerups.pop((nx, ny), None)
                    if picked == 'extra_bomb':
                        s.agent_max_bombs[idx] += 1
                        s.agent_bomb_count[idx] += 1
                    elif picked == 'extra_range':
                        s.agent_blast_range[idx] += 1
        if s.coin_pos is not None:
            for idx, aid in enumerate(self.agent_ids):
                if s.agent_alive[idx] and tuple(s.agent_pos[idx]) == s.coin_pos:
                    team = self.team_of[aid]
                    s.coin_captured_by_team = team
                    s.captured_by_team = team
                    s.coin_pos = None
                    break
        exploding = [b for b in s.bombs if b.timer <= 1]
        s.bombs = [b for b in s.bombs if b.timer > 1]
        for b in s.bombs:
            b.timer -= 1
        killed = set()
        for b in exploding:
            s.agent_bomb_count[b.owner] = min(s.agent_max_bombs[b.owner], s.agent_bomb_count[b.owner] + 1)
            blast_cells = [b.pos]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for r in range(1, b.blast_range + 1):
                    x, y = (b.pos[0] + dx * r, b.pos[1] + dy * r)
                    if not (0 <= x < self.grid_size and 0 <= y < self.grid_size) or s.grid[x, y] == RIGID:
                        break
                    blast_cells.append((x, y))
                    if s.grid[x, y] == WOOD:
                        s.grid[x, y] = EMPTY
                        if self.rng.random() < 0.3:
                            s.powerups[x, y] = self.rng.choice(['extra_bomb', 'extra_range'])
                        break
                    s.powerups.pop((x, y), None)
            for idx in range(self.n_agents):
                if s.agent_alive[idx] and tuple(s.agent_pos[idx]) in blast_cells:
                    killed.add(idx)
        for idx in killed:
            if s.agent_alive[idx]:
                s.agent_alive[idx] = False
                s.just_died.append(idx)
        s.t += 1
        team0_alive = self._team_alive(s, 0)
        team1_alive = self._team_alive(s, 1)
        env_done = not team0_alive or not team1_alive or s.t >= self.max_steps
        info = {'captured_by_team': s.captured_by_team, 'just_died': list(s.just_died), 'team0_alive': team0_alive, 'team1_alive': team1_alive, 'timeout': s.t >= self.max_steps}
        return (s, env_done, info)

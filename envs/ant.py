from __future__ import annotations
import numpy as np
import mujoco
from dataclasses import dataclass
from pathlib import Path
ASSET_PATH = str(Path(__file__).parent / 'assets' / 'two_agent_ant.xml')
FRONT_ACTUATORS = [0, 1, 2, 3]
BACK_ACTUATORS = [4, 5, 6, 7]
ACTION_DIM_PER_AGENT = 4
POINT_A_X, POINT_B_X = (-3.0, 3.0)
WAYPOINT_RADIUS = 0.5

@dataclass
class AntState:
    qpos: np.ndarray
    qvel: np.ndarray
    torso_x: float
    t: int
    reached_point: str = None
    fell: bool = False

class TwoAgentAntEnv:

    def __init__(self, max_steps: int=2000, frame_skip: int=5, torque_penalty_coef: float=0.005, healthy_z_range=(0.2, 1.5), seed: int=None):
        self.model = mujoco.MjModel.from_xml_path(ASSET_PATH)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.torque_penalty_coef = torque_penalty_coef
        self.healthy_z_range = healthy_z_range
        self.rng = np.random.default_rng(seed)
        self.agent_ids = ['front', 'back']
        self.state: AntState = None

    def reset(self) -> AntState:
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[:7] = [0, 0, 0.75, 1, 0, 0, 0]
        qpos[7:] += self.rng.uniform(-0.05, 0.05, size=qpos[7:].shape)
        qvel[:] += self.rng.uniform(-0.05, 0.05, size=qvel.shape)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self.state = AntState(qpos=qpos.copy(), qvel=qvel.copy(), torso_x=float(qpos[0]), t=0)
        self._next_target = 'B'
        return self.state

    def observe(self, agent_id: str, state: AntState) -> np.ndarray:
        qpos_rel = np.concatenate([state.qpos[2:3], state.qpos[3:]])
        which = np.array([1.0, 0.0] if agent_id == 'front' else [0.0, 1.0], dtype=np.float32)
        dist_to_a = np.array([POINT_A_X - state.torso_x], dtype=np.float32)
        dist_to_b = np.array([POINT_B_X - state.torso_x], dtype=np.float32)
        return np.concatenate([qpos_rel, state.qvel, which, dist_to_a, dist_to_b]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return self.model.nq - 2 + self.model.nv + 2 + 2

    @property
    def action_dim(self) -> int:
        return ACTION_DIM_PER_AGENT

    def step(self, actions: dict):
        front_a = np.clip(actions['front'], -1.0, 1.0)
        back_a = np.clip(actions['back'], -1.0, 1.0)
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[FRONT_ACTUATORS] = front_a
        ctrl[BACK_ACTUATORS] = back_a
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        s = self.state
        s.qpos = self.data.qpos.copy()
        s.qvel = self.data.qvel.copy()
        s.torso_x = float(s.qpos[0])
        s.t += 1
        reached = None
        if self._next_target == 'B' and abs(s.torso_x - POINT_B_X) < WAYPOINT_RADIUS:
            reached = 'B'
            self._next_target = 'A'
        elif self._next_target == 'A' and abs(s.torso_x - POINT_A_X) < WAYPOINT_RADIUS:
            reached = 'A'
            self._next_target = 'B'
        s.reached_point = reached
        torso_z = float(s.qpos[2])
        s.fell = not self.healthy_z_range[0] <= torso_z <= self.healthy_z_range[1]
        env_done = s.fell or s.t >= self.max_steps
        info = {'reached_point': reached, 'fell': s.fell, 'torso_x': s.torso_x}
        return (s, env_done, info)

    def torque_penalty(self, agent_id: str, action: np.ndarray) -> float:
        return -self.torque_penalty_coef * float(np.sum(np.square(action)))

"""MuJoCo backend — the reference simulator.

CPU, pip-installable (``pip install mujoco``), runs everywhere including CI.
Other adapters are validated against MuJoCo's behavior. Torques are applied
through the actuator ``ctrl`` vector; base-frame velocities and projected
gravity are derived from the free-joint state via the shared helpers so they
match the other backends exactly.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotCfg
from .base import (
    CAM_AZIMUTH,
    CAM_DISTANCE,
    CAM_ELEVATION,
    CAM_LOOKAT,
    CAPTURE_H,
    CAPTURE_W,
    Simulator,
)
from .state import RobotState, quat_to_projected_gravity, world_velocities_to_base


class MujocoSimulator(Simulator):
    name = "mujoco"

    def __init__(self) -> None:
        self._mj = None
        self.model = None
        self.data = None
        self._joint_qpos_adr: np.ndarray | None = None  # qpos index per canonical joint
        self._joint_qvel_adr: np.ndarray | None = None  # qvel (dof) index per joint
        self._actuator_ids: np.ndarray | None = None
        self.robot_cfg: RobotCfg | None = None
        self._renderer = None
        self._cam = None

    @staticmethod
    def is_available() -> bool:
        import importlib.util

        return importlib.util.find_spec("mujoco") is not None

    def load(
        self, robot_cfg: RobotCfg, *, render: bool = False, capture: bool = False, seed: int = 0
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.robot_cfg = robot_cfg
        self.model = mujoco.MjModel.from_xml_path(robot_cfg.resolve(robot_cfg.mjcf_path))
        self.data = mujoco.MjData(self.model)

        if capture:
            self._renderer = mujoco.Renderer(self.model, height=CAPTURE_H, width=CAPTURE_W)
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = CAM_LOOKAT
            self._cam.distance = CAM_DISTANCE
            self._cam.azimuth = CAM_AZIMUTH
            self._cam.elevation = CAM_ELEVATION

        # Map canonical joint names -> qpos/qvel addresses and actuators. Doing
        # this once is what enforces the canonical joint order on every read.
        qpos_adr, qvel_adr, act_ids = [], [], []
        for jname in robot_cfg.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"joint '{jname}' not found in MJCF")
            qpos_adr.append(self.model.jnt_qposadr[jid])
            qvel_adr.append(self.model.jnt_dofadr[jid])
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            if aid < 0:
                raise ValueError(f"actuator '{jname}' not found in MJCF")
            act_ids.append(aid)
        self._joint_qpos_adr = np.array(qpos_adr, dtype=int)
        self._joint_qvel_adr = np.array(qvel_adr, dtype=int)
        self._actuator_ids = np.array(act_ids, dtype=int)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def total_mass(self) -> float:
        return float(np.sum(self.model.body_mass))

    def reset(self, init=None) -> RobotState:
        mujoco = self._mj
        mujoco.mj_resetData(self.model, self.data)
        height = init.base_height if init else self.robot_cfg.base_height_init
        quat = init.base_quat if init else (1.0, 0.0, 0.0, 0.0)
        joint_pos = init.joint_pos if init else self.robot_cfg.default_joint_pos
        # Free joint qpos layout: [x y z qw qx qy qz]; qvel: [vx vy vz wx wy wz].
        self.data.qpos[0:3] = [0.0, 0.0, height]
        self.data.qpos[3:7] = quat
        for adr, q in zip(self._joint_qpos_adr, joint_pos, strict=True):
            self.data.qpos[adr] = float(q)
        self.data.qvel[:] = 0.0
        if init is not None:
            self.data.qvel[0:3] = init.base_lin_vel
            self.data.qvel[3:6] = init.base_ang_vel
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def apply_torques(self, tau: np.ndarray) -> None:
        tau = np.asarray(tau, dtype=np.float64).ravel()
        self.data.ctrl[self._actuator_ids] = tau

    def step(self) -> None:
        self._mj.mj_step(self.model, self.data)

    def get_state(self) -> RobotState:
        qpos, qvel = self.data.qpos, self.data.qvel
        base_pos = np.array(qpos[0:3], dtype=np.float32)
        quat = np.array(qpos[3:7], dtype=np.float32)  # (w, x, y, z)

        # qvel[0:3] linear, qvel[3:6] angular are in the WORLD frame for a free
        # joint; rotate into the base frame to match what policies expect.
        base_lin_vel, base_ang_vel = world_velocities_to_base(
            quat, np.array(qvel[0:3], dtype=np.float32), np.array(qvel[3:6], dtype=np.float32)
        )

        joint_pos = qpos[self._joint_qpos_adr].astype(np.float32)
        joint_vel = qvel[self._joint_qvel_adr].astype(np.float32)
        return RobotState(
            base_pos=base_pos,
            base_quat=quat,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            projected_gravity=quat_to_projected_gravity(quat),
            sim_time=float(self.data.time),
        )

    def render(self) -> np.ndarray | None:
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

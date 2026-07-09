"""Genesis backend (https://github.com/Genesis-Embodied-AI/Genesis).

Genesis is a GPU-accelerated simulator and is **not runnable on a CPU-only
host** (this environment, standard CI). The adapter is written against the real
Genesis API and is exercised on a GPU host; here it is import-guarded so the
registry simply reports it unavailable rather than crashing.

Genesis exposes a batched env API; we use ``n_envs=1`` and squeeze the leading
batch dimension to fit the single-robot :class:`RobotState` contract. Genesis
loads the same quad12 MJCF as MuJoCo, so the morphology is identical.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotCfg
from .base import CAPTURE_H, CAPTURE_W, Simulator
from .state import RobotState, quat_to_projected_gravity, world_velocities_to_base


class GenesisSimulator(Simulator):
    name = "genesis"

    def __init__(self) -> None:
        self._gs = None
        self.scene = None
        self.robot = None
        self._dof_idx: list[int] = []
        self._dt = 0.005
        self.robot_cfg: RobotCfg | None = None
        self._tau = None
        self._initialized = False
        self._cam = None

    @staticmethod
    def is_available() -> bool:
        # Require both the package and a CUDA device. Importing genesis is heavy
        # and may init the GPU, so we only probe specs here (no side effects).
        import importlib.util

        if importlib.util.find_spec("genesis") is None:
            return False
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def load(
        self, robot_cfg: RobotCfg, *, render: bool = False, capture: bool = False, seed: int = 0
    ) -> None:
        import genesis as gs

        self._gs = gs
        self.robot_cfg = robot_cfg
        # gs.init is process-global and raises on a second call, so guard on
        # Genesis's own flag rather than per-adapter-instance state.
        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu, seed=seed)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self._dt),
            show_viewer=render,
        )
        self.scene.add_entity(gs.morphs.Plane())
        # Genesis re-parses the MJCF via from_xml_string, which drops the file's
        # directory as the base for relative meshdir paths — chdir around load.
        import os
        from pathlib import Path

        mjcf = robot_cfg.resolve(robot_cfg.mjcf_path)
        cwd = os.getcwd()
        os.chdir(Path(mjcf).parent)
        try:
            self.robot = self.scene.add_entity(gs.morphs.MJCF(file=mjcf))
        finally:
            os.chdir(cwd)
        if capture:
            self._cam = self.scene.add_camera(
                res=(CAPTURE_W, CAPTURE_H),
                pos=(1.6, -2.0, 0.85),
                lookat=(0.0, 0.0, 0.45),
                fov=45,
                GUI=False,
            )
        self.scene.build(n_envs=1)

        # Map canonical joint names -> local DOF indices (enforce joint order).
        self._dof_idx = [self.robot.get_joint(name).dof_idx_local for name in robot_cfg.joint_names]
        self._tau = np.zeros(robot_cfg.n_dof, dtype=np.float32)

        # Genesis keeps the MJCF's static <worldbody> as the entity's base link,
        # so entity.get_pos()/set_pos() address a fixed link. Locate the actual
        # floating base: the link driven by the free joint.
        free = next(j for j in self.robot.joints if j.n_dofs == 6)
        self._base_link = free.link
        self._free_qs = list(free.q_idx_local)  # 7 qpos: xyz + wxyz quat

    @property
    def dt(self) -> float:
        return self._dt

    def total_mass(self) -> float:
        try:
            return float(self.robot.get_mass())
        except Exception:
            return 1.0

    def reset(self, init=None) -> RobotState:
        import torch

        joint_pos = init.joint_pos if init else self.robot_cfg.default_joint_pos
        height = init.base_height if init else self.robot_cfg.base_height_init
        quat = list(init.base_quat) if init else [1.0, 0.0, 0.0, 0.0]
        q = torch.tensor(np.asarray(joint_pos), dtype=torch.float32).unsqueeze(0)
        self.robot.set_dofs_position(q, self._dof_idx, envs_idx=[0])
        base_q = torch.tensor([[0.0, 0.0, height, *quat]], dtype=torch.float32)
        self.robot.set_qpos(base_q, self._free_qs, envs_idx=[0])
        self.robot.zero_all_dofs_velocity()

        # Genesis's convex collision hulls sit thicker than MuJoCo's mesh
        # contacts, so the shared base_height_init leaves the feet penetrating
        # the floor and the contact solver fires the robot upward on step one.
        # Lift the base so the lowest collision point clears the floor.
        aabb = _np(self.robot.get_AABB())
        lowest = float(aabb.reshape(-1, 3)[:, 2].min())
        clearance = 0.002
        if lowest < clearance:
            base_q[0, 2] += clearance - lowest
            self.robot.set_qpos(base_q, self._free_qs, envs_idx=[0])
            self.robot.zero_all_dofs_velocity()
        if init is not None:
            vel = torch.tensor(
                np.concatenate([init.base_lin_vel, init.base_ang_vel])[None, :],
                dtype=torch.float32,
            )
            free_dofs = list(range(6))  # free joint dofs are 0..5 locally
            self.robot.set_dofs_velocity(vel, free_dofs, envs_idx=[0])
        return self.get_state()

    def apply_torques(self, tau: np.ndarray) -> None:
        self._tau = np.asarray(tau, dtype=np.float32).ravel()

    def step(self) -> None:
        import torch

        t = torch.tensor(self._tau, dtype=torch.float32).unsqueeze(0)
        self.robot.control_dofs_force(t, self._dof_idx, envs_idx=[0])
        self.scene.step()

    def render(self) -> np.ndarray | None:
        if self._cam is None:
            return None
        out = self._cam.render()
        rgb = out[0] if isinstance(out, tuple) else out
        return np.asarray(rgb)[:, :, :3].astype(np.uint8)

    def get_state(self) -> RobotState:
        base_pos = _np(self._base_link.get_pos())[0]
        quat = _np(self._base_link.get_quat())[0]  # (w, x, y, z)
        lin_world = _np(self._base_link.get_vel())[0]
        ang_world = _np(self._base_link.get_ang())[0]

        base_lin_vel, base_ang_vel = world_velocities_to_base(quat, lin_world, ang_world)

        joint_pos = _np(self.robot.get_dofs_position(self._dof_idx))[0]
        joint_vel = _np(self.robot.get_dofs_velocity(self._dof_idx))[0]
        return RobotState(
            base_pos=base_pos,
            base_quat=quat,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            projected_gravity=quat_to_projected_gravity(quat),
            sim_time=0.0,
        )


def _np(x) -> np.ndarray:
    """Convert a torch tensor (Genesis returns tensors) to a 2D-ish numpy array."""
    arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.atleast_2d(arr).astype(np.float32)

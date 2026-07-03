"""Rapier backend (Rust).

CPU. The physics live in a small Rust crate (``crates/sim2sim-rapier``) built
with PyO3/maturin into the native module ``sim2sim_rapier``; install it with
``pip install sim2sim[rapier]`` or ``maturin develop -m
crates/sim2sim-rapier/Cargo.toml``. It loads the same quad12 robot from URDF so
the policy faces an identical morphology to MuJoCo/PyBullet, and it is driven by
pure joint torques matching the shared PD law applied in the runner.

The Rust side reports *native* state — quaternion as ``(x, y, z, w)`` and base
velocities in the world frame — and this adapter converts to the framework's
``(w, x, y, z)`` / base-frame conventions with the exact same shared helpers as
the other adapters, so the policy sees a bit-identical pipeline.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotCfg
from .base import Simulator
from .state import RobotState, quat_to_projected_gravity, world_velocities_to_base


class RapierSimulator(Simulator):
    name = "rapier"

    def __init__(self) -> None:
        self._sim = None
        self.robot_cfg: RobotCfg | None = None
        self._render = False

    @staticmethod
    def is_available() -> bool:
        import importlib.util

        return importlib.util.find_spec("sim2sim_rapier") is not None

    def load(self, robot_cfg: RobotCfg, *, render: bool = False, seed: int = 0) -> None:
        import sim2sim_rapier

        self.robot_cfg = robot_cfg
        self._render = render  # Rapier has no built-in viewer; kept for parity.
        self._sim = sim2sim_rapier.RapierSim()
        # Armature (reflected motor inertia) and joint damping match the MuJoCo
        # MJCF (`armature="0.01" damping="0.2"`). rapier3d-urdf ignores URDF
        # <dynamics>, so without these the joints have ~10x less effective
        # inertia and no damping, and the shared PD law goes unstable. These are
        # actuator properties, not physics-under-test, so aligning them keeps the
        # sim-to-sim comparison fair. TODO: source from RobotCfg once it carries
        # per-joint armature/damping.
        self._sim.load(
            robot_cfg.resolve(robot_cfg.urdf_path),
            list(robot_cfg.joint_names),
            float(robot_cfg.base_height_init),
            self.dt,
            0.01,  # armature (kg·m²)
            0.2,   # joint damping (N·m·s/rad)
        )

    @property
    def dt(self) -> float:
        # Fixed to match the other CPU backends (see PybulletSimulator).
        return 0.005

    def total_mass(self) -> float:
        return float(self._sim.total_mass())

    def reset(self) -> RobotState:
        self._sim.reset([float(q) for q in self.robot_cfg.default_joint_pos])
        return self.get_state()

    def apply_torques(self, tau: np.ndarray) -> None:
        self._sim.apply_torques(np.asarray(tau, dtype=np.float64).ravel().tolist())

    def step(self) -> None:
        self._sim.step()

    def get_state(self) -> RobotState:
        base_pos, quat_xyzw, lin_world, ang_world, joint_pos, joint_vel = self._sim.get_state()

        # Rust hands us (x, y, z, w); the framework uses (w, x, y, z).
        x, y, z, w = quat_xyzw
        quat = np.array([w, x, y, z], dtype=np.float32)

        base_lin_vel, base_ang_vel = world_velocities_to_base(quat, lin_world, ang_world)
        return RobotState(
            base_pos=np.asarray(base_pos, dtype=np.float32),
            base_quat=quat,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_pos=np.asarray(joint_pos, dtype=np.float32),
            joint_vel=np.asarray(joint_vel, dtype=np.float32),
            projected_gravity=quat_to_projected_gravity(quat),
            sim_time=0.0,  # rapier has no intrinsic clock; runner tracks time
        )

    def close(self) -> None:
        self._sim = None

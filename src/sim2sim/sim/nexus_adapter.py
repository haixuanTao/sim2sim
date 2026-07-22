"""Nexus backend (https://github.com/dimforge/nexus).

Nexus is dimforge's GPU-accelerated multiphysics engine — "Rapier on the GPU",
written in Rust and running compute shaders through WebGPU. Its Python bindings
are published as ``dimforge-nexus3d`` (import name ``nexus3d``).

**Status: eval-runnable, headless, on a forked build.** The stock
``dimforge-nexus3d 0.1.0`` cannot run this eval for two reasons: (1) the only
public entry point is ``NexusViewer``, a kiss3d/winit *window* that needs a
display with GPU-accelerated presentation (unusable headless); and (2) the
``Simulator`` contract needs per-joint torque input + CPU state read-back, which
the stock bindings do not expose (the only joint drive is a *velocity* motor,
and ``viewer.sync`` reads state into the renderer, not to Python).

This adapter targets a **source fork** of the bindings (built from
``github.com/dimforge/nexus``) that adds exactly the missing surface, all
windowless:

* ``NexusBackend`` — a headless ``WebGpu`` backend (no window), plus
  ``*_headless`` variants of ``preload_pipelines`` / ``finalize`` /
  ``simulate`` / ``set_rbd_gravity`` and a physics-only ``insert_mjcf_headless``.
* ``set_multibody_gen_forces_headless`` — external generalized forces (joint
  torques) injected into the multibody solver's force RHS (a small rust-gpu
  shader addition), i.e. the ``apply_torques`` contract.
* ``link_coords`` / ``dof_velocities`` / ``body_poses`` — GPU→CPU read-back of
  joint angles, joint velocities and base pose.
* ``set_rbd_dt`` — match the reference engine's physics timestep.

:meth:`is_available` probes for that surface, so on the stock wheel the backend
still reports unavailable and is skipped; with the fork installed it runs on a
GPU-only host. See the fork's ``crates/nexus_python3d`` for the added bindings.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotCfg
from .base import InitState, Simulator
from .state import (
    RobotState,
    quat_to_projected_gravity,
    world_velocities_to_base,
)

# The headless-eval surface this adapter needs. Present only on the source fork
# of the bindings; absent on the stock ``dimforge-nexus3d`` wheel. Probed by
# is_available(), so the backend auto-enables exactly when the fork is installed.
_REQUIRED_BACKEND_ATTRS = ("NexusBackend",)
_REQUIRED_STATE_METHODS = (
    "finalize_headless",  # windowless scene upload
    "insert_mjcf_headless",  # physics-only MJCF load (no renderer)
    "set_multibody_gen_forces_headless",  # joint torque input
    "link_coords",  # joint-angle read-back
    "dof_velocities",  # joint / base velocity read-back
    "body_poses",  # base pose read-back
    "set_rbd_dt",  # match the reference physics timestep
)

# Physics timestep. Matches the MuJoCo reference (200 Hz), so the runner's
# decimation (control_dt / dt) lands both engines on the same substep cadence.
_PHYSICS_DT = 0.005


class NexusSimulator(Simulator):
    name = "nexus"

    def __init__(self) -> None:
        self._nx = None  # the nexus3d module
        self.backend = None  # nexus3d.NexusBackend (headless WebGPU)
        self.pipeline = None  # nexus3d.NexusPipeline
        self.state = None  # nexus3d.NexusState
        self.timestamps = None  # nexus3d.GpuTimestamps
        self._dt = _PHYSICS_DT
        self._sim_time = 0.0
        self.robot_cfg: RobotCfg | None = None
        # Layout, discovered at load(): number of leading free-base DOFs and the
        # link rows that correspond to the actuated joints (canonical order ==
        # MJCF joint-declaration order == multibody link order).
        self._n_base_dofs = 0
        self._total_dofs = 0
        self._mjcf_path = ""
        self._total_mass = 1.0
        # Canonical (robot_cfg.joint_names, alphabetical) -> tree-order index.
        # Nexus link rows and generalized DOFs follow MJCF body-tree order,
        # which differs from the canonical order on tree-shaped robots.
        self._tree_from_canon: np.ndarray | None = None

    @staticmethod
    def is_available() -> bool:
        import importlib.util

        if importlib.util.find_spec("nexus3d") is None:
            return False
        # Importing nexus3d is light — GPU init is deferred to NexusBackend(), so
        # this has no side effects. Gate on the headless-eval surface existing, so
        # the backend auto-enables when the forked bindings are installed and
        # stays skipped on the stock wheel.
        try:
            import nexus3d as nx
        except Exception:
            return False
        if not all(hasattr(nx, a) for a in _REQUIRED_BACKEND_ATTRS):
            return False
        return all(hasattr(nx.NexusState, m) for m in _REQUIRED_STATE_METHODS)

    # -- setup -------------------------------------------------------------

    def load(
        self, robot_cfg: RobotCfg, *, render: bool = False, capture: bool = False, seed: int = 0
    ) -> None:
        import nexus3d as nx

        self._nx = nx
        self.robot_cfg = robot_cfg
        self._mjcf_path = robot_cfg.resolve(robot_cfg.mjcf_path)
        self._total_mass = _estimate_total_mass(self._mjcf_path)

        # Headless GPU backend (no window). Compiles all pipelines up-front.
        self.backend = nx.NexusBackend()
        self.pipeline = nx.NexusPipeline()
        self.pipeline.preload_pipelines_headless(self.backend)
        self.timestamps = nx.GpuTimestamps.headless(self.backend, 2048)

        tree = _tree_joint_names(self._mjcf_path)
        self._tree_from_canon = np.array(
            [tree.index(n) for n in robot_cfg.joint_names], dtype=int
        )
        self._build_state()

    def _build_state(self) -> None:
        """(Re)build the GPU scene from the MJCF at its neutral pose. Used by both
        load() and reset() — Nexus has no cheap 'teleport to arbitrary pose', so a
        fresh build is the clean way to return to a known initial state."""
        nx = self._nx
        state = nx.NexusState()
        state.set_rbd_dt(self._dt)
        # The eval MJCFs place the floating root at z=0 and rely on the backend
        # to set the initial base height; nexus has no post-load teleport, so
        # load a patched copy with the root body raised to base_height_init.
        with _spawn_height_mjcf(self._mjcf_path, self.robot_cfg.base_height_init) as path:
            state.insert_mjcf_headless(path)
        state.finalize_headless(self.backend)
        state.set_rbd_gravity_headless(self.backend, nx.Vec3(0.0, 0.0, -9.81))
        self.state = state

        c = state.counts()
        self._total_dofs = c.multibody_dofs
        n_dof = self.robot_cfg.n_dof
        # Floating base: leading DOFs are the free-root joint, the rest are the
        # actuated joints in link order. (n_dof actuated -> total - n_dof base.)
        self._n_base_dofs = max(0, self._total_dofs - n_dof)
        self._sim_time = 0.0
        # Locate the free base's row in body_poses: rigid-body order is not
        # root-first in this engine, so find the body sitting at the spawn pose
        # (identity rotation, base_height_init) right after the scene build.
        poses = np.asarray(state.body_poses(self.backend), dtype=np.float32)
        target = np.array([0.0, 0.0, self.robot_cfg.base_height_init], dtype=np.float32)
        pos_err = np.linalg.norm(poses[:, 0:3] - target, axis=1)
        rot_err = 1.0 - np.abs(poses[:, 6])  # |qw| ~ 1 at identity
        self._base_row = int(np.argmin(pos_err + rot_err))

    @property
    def dt(self) -> float:
        return self._dt

    def total_mass(self) -> float:
        return self._total_mass

    # -- episode -----------------------------------------------------------

    def reset(self, init: InitState | None = None) -> RobotState:
        # Rebuild to the MJCF neutral pose and clear any external forces. Per-
        # episode init noise (joint_pos / base_height / velocities) is not applied
        # — Nexus's reduced-coordinate multibody has no host-side setter for
        # arbitrary joint/base state, so we start from the canonical pose. The PD
        # controller then drives the joints to their targets, as in every backend.
        self._build_state()
        # Zero external forces for the fresh state.
        self.state.set_multibody_gen_forces_headless(
            self.backend, 0, [0.0] * self._total_dofs
        )
        return self.get_state()

    def apply_torques(self, tau: np.ndarray) -> None:
        tau = np.asarray(tau, dtype=np.float32).ravel()
        # Generalized-force vector: zeros for the free-base DOFs, then the joint
        # torques in canonical order (== link order). Persists across substeps
        # until the next apply_torques, matching the runner's control loop.
        gen = np.zeros(self._total_dofs, dtype=np.float32)
        gen[self._n_base_dofs + self._tree_from_canon] = tau
        self.state.set_multibody_gen_forces_headless(self.backend, 0, gen.tolist())

    def step(self) -> None:
        self.pipeline.simulate_headless(self.backend, self.state, self.timestamps)
        self._sim_time += self._dt

    def get_state(self) -> RobotState:
        n_dof = self.robot_cfg.n_dof

        poses = np.asarray(self.state.body_poses(self.backend), dtype=np.float32)
        # Base row located at build time. Row = [tx,ty,tz, qx,qy,qz,qw].
        base_pos = poses[self._base_row, 0:3].astype(np.float32)
        qx, qy, qz, qw = (float(v) for v in poses[self._base_row, 3:7])
        quat = np.array([qw, qx, qy, qz], dtype=np.float32)  # -> (w, x, y, z)

        dof_vel = np.asarray(self.state.dof_velocities(self.backend), dtype=np.float32)
        # Free-base spatial velocity is the leading 6 generalized velocities
        # (linear 0:3, angular 3:6), world frame; the actuated joint velocities
        # follow in link order.
        base_lin_world = dof_vel[0:3]
        base_ang_world = dof_vel[3:6]
        joint_vel = dof_vel[self._n_base_dofs + self._tree_from_canon].astype(np.float32)
        base_lin_vel, base_ang_vel = world_velocities_to_base(
            quat, base_lin_world, base_ang_world
        )

        # Joint angles live in the link workspace `coords`: for a 1-DOF revolute
        # link the angle is the single non-zero angular entry (columns 3..6, in
        # the joint's local frame — column 3 for these hinges). Rows 1..n_dof are
        # the actuated links in canonical order.
        coords = np.asarray(self.state.link_coords(self.backend), dtype=np.float32)
        joint_ang = coords[1 + self._tree_from_canon, 3:6]
        joint_pos = joint_ang.sum(axis=1).astype(np.float32)  # one non-zero axis per hinge

        return RobotState(
            base_pos=base_pos,
            base_quat=quat,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            projected_gravity=quat_to_projected_gravity(quat),
            sim_time=float(self._sim_time),
        )

    def close(self) -> None:
        self.state = None
        self.pipeline = None
        self.backend = None
        self.timestamps = None


def _tree_joint_names(mjcf_path: str) -> list[str]:
    """Hinge-joint names in MJCF body-tree order — the multibody link/DOF order
    nexus uses (link row = 1 + tree index; gen-force DOF = base DOFs + index)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    return [
        model.joint(j).name
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]


import contextlib
import os
import xml.etree.ElementTree as ET


@contextlib.contextmanager
def _spawn_height_mjcf(mjcf_path: str, height: float):
    """Yield a path to a copy of the MJCF with the free-root body raised to
    ``height``. The copy lives next to the original so relative ``<include>``
    paths keep resolving; it is removed on exit."""

    def find_free_root(tree):
        for body in tree.getroot().iter("body"):
            if body.find("freejoint") is not None or any(
                j.get("type") == "free" for j in body.findall("joint")
            ):
                return body
        return None

    def raise_root(body):
        pos = [float(v) for v in (body.get("pos") or "0 0 0").split()]
        pos[2] = height
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")

    def tmp_name(path):
        return os.path.join(
            os.path.dirname(path), f".nexus_spawn_{os.getpid()}_{os.path.basename(path)}"
        )

    base_dir = os.path.dirname(mjcf_path)
    scene = ET.parse(mjcf_path)
    tmp_files = []
    root_body = find_free_root(scene)
    if root_body is not None:
        raise_root(root_body)
    else:
        # The free root may live in an <include>d file: patch a copy of that
        # file and repoint the include in the scene copy.
        for inc in scene.getroot().iter("include"):
            inc_path = os.path.join(base_dir, inc.get("file", ""))
            if not os.path.isfile(inc_path):
                continue
            sub = ET.parse(inc_path)
            body = find_free_root(sub)
            if body is None:
                continue
            raise_root(body)
            sub_out = tmp_name(inc_path)
            sub.write(sub_out)
            tmp_files.append(sub_out)
            inc.set("file", os.path.basename(sub_out))
            root_body = body
            break
    if root_body is None:
        # No free root anywhere; the robot spawns at the MJCF pose.
        yield mjcf_path
        return
    out = tmp_name(mjcf_path)
    scene.write(out)
    tmp_files.append(out)
    try:
        yield out
    finally:
        for f in tmp_files:
            with contextlib.suppress(OSError):
                os.remove(f)


def _estimate_total_mass(mjcf_path: str) -> float:
    """Total robot mass (kg) for cost-of-transport. Nexus doesn't expose body
    masses to Python, so read them from the same MJCF via MuJoCo (always
    installed as the CPU reference backend). Falls back to 1.0 if unavailable."""
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(mjcf_path)
        return float(np.sum(model.body_mass))
    except Exception:
        return 1.0

"""Nexus backend (https://github.com/dimforge/nexus).

Nexus is dimforge's GPU-accelerated multiphysics engine — "Rapier on the GPU",
written in Rust and running compute shaders through WebGPU. Its Python bindings
are published as ``dimforge-nexus3d`` (import name ``nexus3d``).

**Status: integrated but not yet eval-runnable.** ``dimforge-nexus3d`` installs
and imports fine, and the world-setup surface below (``NexusViewer`` /
``NexusPipeline`` / ``NexusState``, ``insert_mjcf``, ``set_rbd_gravity``,
``pipeline.simulate``) is real and correct. But the sim2sim :class:`Simulator`
contract needs two things the current bindings do **not** expose:

1. **Per-joint torque input.** The only joint drive is
   ``state.set_multibody_motor_velocity(...)`` (a *velocity* motor). There is no
   ``set_multibody_joint_force`` / generalized-force entry point.
2. **CPU state read-back.** ``NexusState`` exposes no getters for base pose /
   velocity or per-joint position / velocity, and ``insert_mjcf`` returns only a
   :class:`MjcfSceneInfo` (``loaded`` / ``z_up``), not per-joint handles.
   ``viewer.sync(...)`` reads GPU state back *into the renderer* for display
   only; ``NexusViewer`` is a *windowed* viewer with no off-screen read-back.

So :meth:`is_available` probes for the missing methods and reports the backend
unavailable until dimforge adds them (at which point it auto-enables). The
control / read-back methods below raise :class:`NotImplementedError` with a
precise message rather than calling API that does not exist. When the bindings
grow force control + state getters, fill those three methods in.
"""

from __future__ import annotations

import numpy as np

from ..config import RobotCfg
from .base import Simulator
from .state import RobotState

# Methods NexusState must expose before an eval can run: per-joint torque input
# plus base/joint state read-back. Probed by is_available(); absent as of the
# pinned dimforge-nexus3d release, so the registry skips this backend for now.
_REQUIRED_STATE_METHODS = (
    "set_multibody_joint_force",  # torque input (only motor *velocity* exists today)
    "multibody_root_pose",  # base pose read-back
    "multibody_root_linvel",  # base linear velocity read-back
    "multibody_root_angvel",  # base angular velocity read-back
    "multibody_joint_position",  # per-joint position read-back
    "multibody_joint_velocity",  # per-joint velocity read-back
)

_UNSUPPORTED_MSG = (
    "nexus3d exposes neither per-joint torque input (set_multibody_joint_force) "
    "nor CPU state read-back (multibody_root_pose / multibody_joint_position, ...), "
    "so the sim2sim Simulator contract cannot be satisfied yet. is_available() "
    "returns False for this reason; this method will be implemented once the "
    "dimforge-nexus3d bindings add force control and state getters."
)


class NexusSimulator(Simulator):
    name = "nexus"

    def __init__(self) -> None:
        self._nx = None  # the nexus3d module
        self.viewer = None  # nexus3d.NexusViewer
        self.pipeline = None  # nexus3d.NexusPipeline
        self.state = None  # nexus3d.NexusState
        self.timestamps = None  # nexus3d.GpuTimestamps
        self._info = None  # nexus3d.MjcfSceneInfo from insert_mjcf
        self._dt = 0.005
        self.robot_cfg: RobotCfg | None = None

    @staticmethod
    def is_available() -> bool:
        import importlib.util

        if importlib.util.find_spec("nexus3d") is None:
            return False
        # Importing nexus3d is light — GPU backend init is deferred to
        # NexusViewer(), so this has no side effects. Nexus manages its own GPU
        # via WebGPU (no CUDA/torch probe). Gate on the control/read-back API
        # actually existing, so the backend auto-enables when dimforge ships it.
        try:
            import nexus3d as nx
        except Exception:
            return False
        return all(hasattr(nx.NexusState, m) for m in _REQUIRED_STATE_METHODS)

    def load(self, robot_cfg: RobotCfg, *, render: bool = False, seed: int = 0) -> None:
        # World setup below uses the real, working nexus3d API and doubles as a
        # reference for loading an MJCF into Nexus. The eval loop cannot proceed
        # past this without joint handles + torque control (see module docstring),
        # so we set the world up and then raise with a precise explanation.
        import nexus3d as nx

        self._nx = nx
        self.robot_cfg = robot_cfg

        self.viewer = nx.NexusViewer()
        self.pipeline = nx.NexusPipeline()
        self.pipeline.preload_pipelines(self.viewer)
        self.state = nx.NexusState()

        self.viewer.set_up_axis(nx.Vec3.Z)  # MJCF is Z-up
        self._info = self.state.insert_mjcf(self.viewer, robot_cfg.resolve(robot_cfg.mjcf_path))
        self.state.finalize(self.viewer)
        self.state.set_rbd_gravity(self.viewer, nx.Vec3(0.0, 0.0, -9.81))
        self.timestamps = nx.GpuTimestamps(self.viewer, 2048)

        raise NotImplementedError(
            "Nexus world loaded (MJCF + gravity), but per-joint control/read-back "
            f"is unavailable. {_UNSUPPORTED_MSG}"
        )

    @property
    def dt(self) -> float:
        return self._dt

    def reset(self) -> RobotState:
        raise NotImplementedError(_UNSUPPORTED_MSG)

    def apply_torques(self, tau: np.ndarray) -> None:
        raise NotImplementedError(_UNSUPPORTED_MSG)

    def step(self) -> None:
        # The step itself is real (pipeline.simulate advances one GPU frame); it
        # is unreachable only because apply_torques/get_state cannot run yet.
        self.pipeline.simulate(self.viewer, self.state, self.timestamps)

    def get_state(self) -> RobotState:
        raise NotImplementedError(_UNSUPPORTED_MSG)

    def close(self) -> None:
        self.viewer = None
        self.pipeline = None
        self.state = None
        self.timestamps = None

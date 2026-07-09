"""The simulator adapter contract.

Adapters are deliberately *thin*: load a model, accept joint torques, advance
one physics step, and report a neutral :class:`RobotState`. The episode loop,
control law, and observation assembly all live above this interface (in the
runner / control / obs modules), which is what guarantees the policy sees an
identical pipeline no matter which backend is underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import RobotCfg
from .state import RobotState

# Shared default for off-screen video capture, so every backend frames the robot
# identically (a ~0.6 m biped). Adapters translate this to their renderer's API.
CAPTURE_W, CAPTURE_H = 640, 480
CAM_LOOKAT = (0.0, 0.0, 0.45)
CAM_DISTANCE = 2.6
CAM_AZIMUTH = 120.0
CAM_ELEVATION = -8.0


class Simulator(ABC):
    name: str = "abstract"

    @abstractmethod
    def load(
        self, robot_cfg: RobotCfg, *, render: bool = False, capture: bool = False, seed: int = 0
    ) -> None:
        """Construct the physics world and load the robot. Called once.

        ``render`` opens the backend's live GUI window; ``capture`` sets up an
        off-screen renderer so :meth:`render` returns RGB frames for video.
        """

    def render(self) -> np.ndarray | None:
        """Return the current frame as an ``(H, W, 3)`` uint8 RGB array.

        Returns ``None`` when the backend has no capture renderer configured
        (i.e. it was not loaded with ``capture=True``, or it doesn't support
        off-screen rendering). This is the generic hook the recording driver
        uses, so a new backend gains video support just by overriding it.
        """
        return None

    @abstractmethod
    def reset(self) -> RobotState:
        """Reset the robot to its initial standing pose; return the first state."""

    @abstractmethod
    def apply_torques(self, tau: np.ndarray) -> None:
        """Set the joint torques to apply on the next :meth:`step`."""

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by exactly :attr:`dt` seconds."""

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return the current robot state in canonical joint order."""

    @property
    @abstractmethod
    def dt(self) -> float:
        """Physics timestep in seconds."""

    def total_mass(self) -> float:
        """Total robot mass (kg), used for cost-of-transport. Override per backend."""
        return 1.0

    def close(self) -> None:  # noqa: B027 - intentional optional cleanup hook
        pass

    @staticmethod
    def is_available() -> bool:
        """Whether this backend can run here (deps importable, hardware present).

        Must NOT initialize a GPU or load heavy modules with side effects — it is
        called by the registry just to decide which simulators to run.
        """
        return False

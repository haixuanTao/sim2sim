"""Cube-drop smoke demo — PyBullet backend.

Same scenario as the MuJoCo demo: a rigid cube dropped from ~1.5 m with a small
tilt and spin, settling on the floor over 5 s. Runs fully head-less in DIRECT
mode and renders with the built-in software (TinyRenderer) camera, so no GPU or
display is required.

Run:  python examples/cube_drop/pybullet_cube.py
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v2 as imageio
import pybullet as p
import pybullet_data

DURATION_S = 5.0
FPS = 30
W, H = 640, 480
TIMESTEP = 1.0 / 240.0


def main() -> None:
    out = Path(__file__).parent / "cube_pybullet.mp4"

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(TIMESTEP)
    p.loadURDF("plane.urdf")

    half = 0.15
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[0.90, 0.32, 0.22, 1]
    )
    cube = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0, 0, 1.5],
        baseOrientation=p.getQuaternionFromEuler([0.21, 0.38, 0.09]),
    )
    p.resetBaseVelocity(cube, angularVelocity=[1.5, -1.0, 0.5])

    view = p.computeViewMatrix(
        cameraEyePosition=[2.8, -2.8, 2.0],
        cameraTargetPosition=[0, 0, 0.4],
        cameraUpVector=[0, 0, 1],
    )
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=W / H, nearVal=0.1, farVal=20)

    n_frames = int(DURATION_S * FPS)
    steps_per_frame = max(1, round((1.0 / FPS) / TIMESTEP))

    frames = []
    t0 = time.perf_counter()
    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            p.stepSimulation()
        _, _, rgba, _, _ = p.getCameraImage(
            W, H, view, proj, renderer=p.ER_TINY_RENDERER
        )
        frames.append(_to_rgb(rgba))
    gen = time.perf_counter() - t0

    p.disconnect(cid)
    imageio.mimsave(out, frames, fps=FPS)
    print(f"wrote {out}  ({len(frames)} frames, {DURATION_S:.0f}s @ {FPS}fps)")
    print(f"[fps] pybullet: {len(frames)} frames in {gen:.2f}s = {len(frames) / gen:.1f} gen-fps")


def _to_rgb(rgba):
    import numpy as np

    arr = np.reshape(np.asarray(rgba, dtype=np.uint8), (H, W, 4))
    return arr[:, :, :3]


if __name__ == "__main__":
    main()

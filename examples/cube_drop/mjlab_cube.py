"""Cube-drop smoke demo — mjlab backend (MuJoCo-Warp, GPU).

Same scenario as the other demos: a rigid cube dropped from ~1.5 m with a small
tilt, settling on the floor over 5 s. mjlab runs the physics on the GPU via
MuJoCo-Warp; we read the base pose back each frame and render with the host
MuJoCo renderer (EGL, off-screen) — the same host-model + GPU-sim split the
mjlab adapter uses.

Run:  MUJOCO_GL=egl python examples/cube_drop/mjlab_cube.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

DURATION_S = 5.0
FPS = 30
W, H = 640, 480
DT = 0.005

# Cube starts tilted at 1.5 m so it topples on landing — no GPU-state writes
# needed, we only read qpos back for rendering.
MJCF = """
<mujoco>
  <option timestep="0.005"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="1 -1 3" dir="-1 1 -3" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.88 1"/>
    <body name="cube" pos="0 0 1.5" euler="12 22 5">
      <freejoint/>
      <geom name="box" type="box" size="0.15 0.15 0.15" rgba="0.90 0.32 0.22 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _row0_np(arr) -> np.ndarray:
    row = arr[0]
    if hasattr(row, "detach"):
        return row.detach().cpu().numpy()
    if hasattr(row, "numpy"):
        return row.numpy()
    return np.asarray(row)


def main() -> None:
    from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

    model = mujoco.MjModel.from_xml_string(MJCF)
    cfg = SimulationCfg(mujoco=MujocoCfg(timestep=DT, gravity=(0.0, 0.0, -9.81)))
    sim = Simulation(num_envs=1, cfg=cfg, model=model, device="cuda:0")
    sim.reset()
    sim.forward()

    host_data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.4]
    cam.distance = 4.0
    cam.azimuth = 135.0
    cam.elevation = -20.0

    n_frames = int(DURATION_S * FPS)
    steps_per_frame = max(1, round((1.0 / FPS) / DT))

    frames = []
    with mujoco.Renderer(model, height=H, width=W) as renderer:
        t0 = time.perf_counter()
        for _ in range(n_frames):
            for _ in range(steps_per_frame):
                sim.step()
            host_data.qpos[:] = _row0_np(sim.data.qpos)
            mujoco.mj_forward(model, host_data)
            renderer.update_scene(host_data, camera=cam)
            frames.append(renderer.render())
        gen = time.perf_counter() - t0

    out = Path(__file__).parent / "cube_mjlab.mp4"
    imageio.mimsave(out, frames, fps=FPS)
    print(f"wrote {out}  ({len(frames)} frames, {DURATION_S:.0f}s @ {FPS}fps)")
    print(f"[fps] mjlab: {len(frames)} frames in {gen:.2f}s = {len(frames) / gen:.1f} gen-fps")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

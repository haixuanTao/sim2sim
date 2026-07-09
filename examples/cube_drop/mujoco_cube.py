"""Cube-drop smoke demo — MuJoCo backend.

A single rigid cube is dropped from ~1.5 m with a small initial tilt and spin,
falls under gravity, and settles on the floor over 5 s. Rendered off-screen
(EGL) to an MP4. This is a raw-physics sanity check of the engine itself, not a
sim2sim robot rollout, so it deliberately does NOT go through the adapter/PD
pipeline.

Run:  MUJOCO_GL=egl python examples/cube_drop/mujoco_cube.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco

DURATION_S = 5.0
FPS = 30
W, H = 640, 480

MJCF = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
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


def main() -> None:
    out = Path(__file__).parent / "cube_mujoco.mp4"
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    # give it a little tumble so the fall is visually obvious
    data.qvel[3:6] = [1.5, -1.0, 0.5]  # angular velocity of the free joint

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.4]
    cam.distance = 4.0
    cam.azimuth = 135.0
    cam.elevation = -20.0

    n_frames = int(DURATION_S * FPS)
    steps_per_frame = max(1, round((1.0 / FPS) / model.opt.timestep))

    frames = []
    t_phys = t_render = 0.0
    with mujoco.Renderer(model, height=H, width=W) as renderer:
        t0 = time.perf_counter()
        for _ in range(n_frames):
            t = time.perf_counter()
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            t_phys += time.perf_counter() - t
            t = time.perf_counter()
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
            t_render += time.perf_counter() - t
        gen = time.perf_counter() - t0

    imageio.mimsave(out, frames, fps=FPS)
    print(f"wrote {out}  ({len(frames)} frames, {DURATION_S:.0f}s @ {FPS}fps)")
    print(f"[fps] mujoco: {len(frames)} frames in {gen:.2f}s = {len(frames) / gen:.1f} gen-fps")
    n = len(frames)
    print(f"[segments] mujoco: physics={1e3 * t_phys / n:.2f}ms render={1e3 * t_render / n:.2f}ms")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

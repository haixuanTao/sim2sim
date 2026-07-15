"""Cube-drop smoke demo — mjlab backend (MuJoCo-Warp, GPU).

Same scenario as the other demos: a rigid cube dropped from ~1.5 m with a small
tilt, settling on the floor over 5 s. mjlab runs the physics on the GPU via
MuJoCo-Warp; we read the base pose back each frame and render with the host
MuJoCo renderer (EGL, off-screen) — the same host-model + GPU-sim split the
mjlab adapter uses.

Run:  MUJOCO_GL=egl python examples/cube_drop/mjlab_cube.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

_PX1 = np.empty((1, 1, 3), np.uint8)
_RECT1 = mujoco.MjrRect(0, 0, 1, 1)


def render_noread(renderer: mujoco.Renderer) -> None:
    """Draw without the full-frame GPU->CPU readback; a 1x1 readPixels still
    flushes the GL pipeline so the draw is honestly timed (same idea as the
    Nexus/Genesis scripts' --no-capture)."""
    if renderer._gl_context:
        renderer._gl_context.make_current()
    mujoco.mjr_render(renderer._rect, renderer._scene, renderer._mjr_context)
    mujoco.mjr_readPixels(_PX1, None, _RECT1, renderer._mjr_context)

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-capture", action="store_true",
                    help="skip the frame readback (and the MP4): benchmark the sim+render loop with frames staying on the GPU")
    args = ap.parse_args()

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
    t_phys = t_sync = t_render = 0.0
    with mujoco.Renderer(model, height=H, width=W) as renderer:
        t0 = time.perf_counter()
        for _ in range(n_frames):
            t = time.perf_counter()
            for _ in range(steps_per_frame):
                sim.step()
            t_phys += time.perf_counter() - t
            t = time.perf_counter()
            host_data.qpos[:] = _row0_np(sim.data.qpos)
            mujoco.mj_forward(model, host_data)
            t_sync += time.perf_counter() - t
            t = time.perf_counter()
            renderer.update_scene(host_data, camera=cam)
            if args.no_capture:
                render_noread(renderer)
            else:
                frames.append(renderer.render())
            t_render += time.perf_counter() - t
        gen = time.perf_counter() - t0

    if args.no_capture:
        print(f"[fps-nocapture] mjlab: {n_frames} frames in {gen:.2f}s = {n_frames / gen:.1f} gen-fps")
        n = n_frames
        print(f"[segments] mjlab: physics={1e3 * t_phys / n:.2f}ms sync={1e3 * t_sync / n:.2f}ms "
              f"render={1e3 * t_render / n:.2f}ms")
        return
    out = Path(__file__).parent / "cube_mjlab.mp4"
    imageio.mimsave(out, frames, fps=FPS)
    print(f"wrote {out}  ({len(frames)} frames, {DURATION_S:.0f}s @ {FPS}fps)")
    print(f"[fps] mjlab: {len(frames)} frames in {gen:.2f}s = {len(frames) / gen:.1f} gen-fps")
    n = len(frames)
    print(f"[segments] mjlab: physics={1e3 * t_phys / n:.2f}ms sync={1e3 * t_sync / n:.2f}ms "
          f"render={1e3 * t_render / n:.2f}ms")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

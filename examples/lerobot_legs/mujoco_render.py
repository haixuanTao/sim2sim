"""LeRobot-legs robot demo — MuJoCo, real bipedal-platform asset.

Loads the real LeRobot bipedal platform (full STL meshes + built-in position
actuators, kp 10-100) from lerobot-humanoid-design, lifts the base so the feet
start on the floor, and holds the zero pose with the model's own servos
(``ctrl = 0``), rendered off-screen (EGL) to an MP4. Same asset as
``nexus_render.py`` so the two backends are directly comparable.

Run:  MUJOCO_GL=egl python examples/lerobot_legs/mujoco_render.py
"""

from __future__ import annotations

import argparse
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

XML = (
    Path.home()
    / "Documents/work/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms/mjcf/sim_scene_safe.xml"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-capture", action="store_true",
                    help="skip the frame readback (and the MP4): benchmark the sim+render loop with frames staying on the GPU")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)

    # Canonical standing init from the RL training config (RL_policy/*/config.yaml):
    # base at z = 0.72 with all joints zero. The zero pose is NOT statically
    # stable (CoM ahead of the weak kp=20 ankles, which saturate their +-0.18
    # limits) — bias hip and ankle pitch to bring the CoM over the feet. Right-
    # leg joint axes are mirrored, so the right side takes the opposite sign
    # (grid-searched: this holds z~=0.69 indefinitely).
    data.qpos[2] = 0.72
    data.ctrl[:] = 0.0
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if "hipy" in name or "ankley" in name:
            data.ctrl[i] = 0.15 if "right" in name else -0.15
    mujoco.mj_forward(model, data)
    z0 = data.qpos[2]

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, z0 * 0.6]
    cam.distance, cam.azimuth, cam.elevation = 1.8, 135, -15

    n_frames = int(DURATION_S * FPS)
    spf = max(1, round(1.0 / (FPS * model.opt.timestep)))
    frames = []
    t_phys = t_render = 0.0
    t0 = time.perf_counter()
    for _ in range(n_frames):
        t = time.perf_counter()
        for _ in range(spf):
            mujoco.mj_step(model, data)
        t_phys += time.perf_counter() - t
        t = time.perf_counter()
        renderer.update_scene(data, cam)
        if args.no_capture:
            render_noread(renderer)
        else:
            frames.append(renderer.render())
        t_render += time.perf_counter() - t
    gen_s = time.perf_counter() - t0

    if args.no_capture:
        print(f"[fps-nocapture] lerobot/mujoco: {n_frames} frames in {gen_s:.2f}s = {n_frames / gen_s:.1f} gen-fps")
        n = n_frames
        print(f"[segments] lerobot/mujoco: physics={1e3 * t_phys / n:.2f}ms render={1e3 * t_render / n:.2f}ms")
        print(f"final base height: {data.qpos[2]:.3f} m (started {z0:.3f})")
        return
    out = Path(__file__).parent / "lerobot_mujoco_real.mp4"
    imageio.mimsave(out, frames, fps=FPS)
    print(f"wrote {out}  ({len(frames)} frames @ {FPS}fps)")
    print(f"[fps] lerobot/mujoco: {len(frames)} frames in {gen_s:.2f}s = {len(frames) / gen_s:.1f} gen-fps")
    n = len(frames)
    print(f"[segments] lerobot/mujoco: physics={1e3 * t_phys / n:.2f}ms render={1e3 * t_render / n:.2f}ms")
    print(f"final base height: {data.qpos[2]:.3f} m (started {z0:.3f})")


if __name__ == "__main__":
    main()

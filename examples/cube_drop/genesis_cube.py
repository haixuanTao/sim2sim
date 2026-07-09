"""Cube-drop smoke demo — Genesis backend (GPU).

Same scenario as the MuJoCo demo: a rigid cube dropped from ~1.5 m
with a small tilt, settling on the floor over 5 s. Genesis is GPU-accelerated;
we render off-screen with a headless camera and write an MP4. Falls back to the
CPU backend if the GPU can't be initialised.

Run:  python examples/cube_drop/genesis_cube.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

DURATION_S = 5.0
FPS = 30
W, H = 640, 480
DT = 1.0 / 120.0


def patch_out_readback() -> None:
    """Skip the GPU->CPU glReadPixels in Genesis's rasterizer (frames stay on
    the GPU; the MSAA-resolve blit still runs). Same idea as the Nexus
    scripts' --no-capture: benchmark sim+render without the readback."""
    from genesis.ext.pyrender import jit_render

    dummies: dict[tuple, np.ndarray] = {}

    def no_read(self, width, height, rgba):
        key = (height, width, 4 if rgba else 3)
        if key not in dummies:
            dummies[key] = np.zeros(key, np.uint8)
        return dummies[key]

    jit_render.JITRenderer.read_color_buf = no_read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-capture", action="store_true",
                    help="skip the frame readback (and the MP4): benchmark the sim+render loop with frames staying on the GPU")
    args = ap.parse_args()

    import genesis as gs

    if args.no_capture:
        patch_out_readback()

    try:
        gs.init(backend=gs.gpu)
    except Exception as e:  # brand-new GPU / torch-CUDA mismatch -> CPU fallback
        print(f"[genesis] GPU init failed ({e}); falling back to CPU backend")
        gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(0.0, 0.0, 1.5), euler=(12.0, 22.0, 5.0))
    )
    cam = scene.add_camera(
        res=(W, H), pos=(2.8, -2.8, 2.0), lookat=(0.0, 0.0, 0.4), fov=45, GUI=False
    )
    scene.build()

    n_frames = int(DURATION_S * FPS)
    steps_per_frame = max(1, round((1.0 / FPS) / DT))

    # Warmup outside the timers: the first step/render JIT-compile taichi and
    # render kernels (~2 s), which otherwise lands in the measured segments.
    for _ in range(5):
        scene.step()
        cam.render()

    frames = []
    t_phys = t_render = 0.0
    t0 = time.perf_counter()
    for _ in range(n_frames):
        t = time.perf_counter()
        for _ in range(steps_per_frame):
            scene.step()
        t_phys += time.perf_counter() - t
        t = time.perf_counter()
        out = cam.render()
        rgb = out[0] if isinstance(out, tuple) else out
        frames.append(rgb[:, :, :3])
        t_render += time.perf_counter() - t
    gen = time.perf_counter() - t0

    n = len(frames)
    if args.no_capture:
        print(f"[fps-nocapture] genesis: {n} frames in {gen:.2f}s = {n / gen:.1f} gen-fps")
        return
    path = Path(__file__).parent / "cube_genesis.mp4"
    imageio.mimsave(path, frames, fps=FPS)
    print(f"wrote {path}  ({len(frames)} frames, {DURATION_S:.0f}s @ {FPS}fps)")
    print(f"[fps] genesis: {len(frames)} frames in {gen:.2f}s = {len(frames) / gen:.1f} gen-fps")
    print(f"[segments] genesis: physics={1e3 * t_phys / n:.2f}ms render={1e3 * t_render / n:.2f}ms")


if __name__ == "__main__":
    main()

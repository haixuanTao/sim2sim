"""Cube-drop smoke demo — Genesis backend (GPU).

Same scenario as the MuJoCo demo: a rigid cube dropped from ~1.5 m
with a small tilt, settling on the floor over 5 s. Genesis is GPU-accelerated;
we render off-screen with a headless camera and write an MP4. Falls back to the
CPU backend if the GPU can't be initialised.

Run:  python examples/cube_drop/genesis_cube.py
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v2 as imageio

DURATION_S = 5.0
FPS = 30
W, H = 640, 480
DT = 1.0 / 120.0


def main() -> None:
    import genesis as gs

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

    frames = []
    t0 = time.perf_counter()
    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            scene.step()
        out = cam.render()
        rgb = out[0] if isinstance(out, tuple) else out
        frames.append(rgb[:, :, :3])
    gen = time.perf_counter() - t0

    path = Path(__file__).parent / "cube_genesis.mp4"
    imageio.mimsave(path, frames, fps=FPS)
    print(f"wrote {path}  ({len(frames)} frames, {DURATION_S:.0f}s @ {FPS}fps)")
    print(f"[fps] genesis: {len(frames)} frames in {gen:.2f}s = {len(frames) / gen:.1f} gen-fps")


if __name__ == "__main__":
    main()

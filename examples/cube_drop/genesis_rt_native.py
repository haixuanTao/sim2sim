"""Cube drop rendered by Genesis's OWN ray tracer (LuisaRender), no Mitsuba.

Genesis is the only pip-adjacent engine with a built-in path tracer, but it
ships without it: ``gs.renderers.RayTracer`` needs ``LuisaRenderPy`` compiled
from source inside a Genesis clone (ext/LuisaRender). See the README for the
build recipe; this script assumes a Genesis clone (editable install) whose
``ext/LuisaRender/build/bin`` exists.

Run:  <genesis-venv>/bin/python examples/cube_drop/genesis_rt_native.py --outdir /tmp/gs_frames
Then: ffmpeg -y -framerate 30 -i /tmp/gs_frames/f_%03d.png -pix_fmt yuv420p cube_rt_genesis_native.mp4
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

N_FRAMES = 150
FRAME_DT = 1.0 / 30.0
HALF = 0.15
START_Z = 1.5
EULER_DEG = (12.0, 22.0, 5.0)
W, H, SPP = 480, 368, 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import genesis as gs
    from PIL import Image

    gs.init(backend=gs.gpu)
    dt = 1.0 / 120.0
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt),
        renderer=gs.renderers.RayTracer(
            tracing_depth=10,
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(color=(0.8, 0.85, 0.9)),
            ),
            lights=[{"pos": (2.0, -1.0, 6.0), "color": (1.0, 1.0, 1.0), "intensity": 12.0, "radius": 1.5}],
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.65, 0.65, 0.65)))
    cube = scene.add_entity(
        gs.morphs.Box(size=(2 * HALF,) * 3, pos=(0, 0, START_Z), euler=EULER_DEG),
        surface=gs.surfaces.Rough(color=(0.8, 0.25, 0.2)),
    )
    cam = scene.add_camera(
        res=(W, H), pos=(3.2, -3.2, 2.2), lookat=(0.0, 0.0, 0.35), fov=40, spp=SPP, GUI=False
    )
    scene.build()

    spf = max(1, round(FRAME_DT / dt))
    phys_s = rend_s = 0.0
    for i in range(N_FRAMES):
        t0 = time.perf_counter()
        for _ in range(spf):
            scene.step()
        t1 = time.perf_counter()
        rgb = cam.render()[0]
        t2 = time.perf_counter()
        phys_s += t1 - t0
        rend_s += t2 - t1
        Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8)).save(
            os.path.join(args.outdir, f"f_{i:03d}.png")
        )

    msg = (
        f"[rt-native] genesis: physics {spf * N_FRAMES / phys_s:,.0f} steps/s | "
        f"LuisaRender path trace {N_FRAMES / rend_s:.1f} fps "
        f"({1000 * rend_s / N_FRAMES:.0f} ms/frame @ {SPP} spp, {W}x{H})"
    )
    print(msg)
    with open(os.path.join(args.outdir, "result.txt"), "w") as f:
        f.write(msg + "\n")
    _ = cube


if __name__ == "__main__":
    main()

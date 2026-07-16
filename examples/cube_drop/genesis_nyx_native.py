"""Cube drop rendered by Genesis's Nyx path tracer (gs-nyx-plugin).

Same scene as ``genesis_cube.py`` / ``genesis_rt_native.py``, rendered by
Genesis's new official renderer plugin instead of the legacy from-source
LuisaRender path. Nyx is a camera sensor: GPU path tracing, CUDA-resident
uint8 output, wheel-installable.

Requirements (see ~/rt_build/nyx-venv): genesis git-main + torch +
``gs-nyx-plugin``; import the plugin AFTER ``gs.init()``.

Run:  ~/rt_build/nyx-venv/bin/python examples/cube_drop/genesis_nyx_native.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

DURATION_S = 5.0
FPS = 30
W, H = 480, 368  # match the other native-RT cube rows
SPP = 64
DT = 1.0 / 120.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-capture", action="store_true",
                    help="skip the GPU->CPU frame copy (and the MP4): frames stay CUDA-resident")
    args = ap.parse_args()

    import genesis as gs

    gs.init(backend=gs.gpu)
    # Import AFTER gs.init(): the plugin's module body touches gs.qd_float.
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(0.0, 0.0, 1.5), euler=(12.0, 22.0, 5.0))
    )
    cam = scene.add_sensor(NyxCameraOptions(
        res=(W, H), pos=(2.8, -2.8, 2.0), lookat=(0.0, 0.0, 0.4),
        spp=SPP, denoise=True))
    scene.build()

    n_frames = int(DURATION_S * FPS)
    spf = max(1, round((1.0 / FPS) / DT))

    for _ in range(5):  # kernel JIT + first-render allocations
        scene.step()
        cam.read()

    frames = []
    t_phys = t_render = t_read = 0.0
    t0 = time.perf_counter()
    for _ in range(n_frames):
        t = time.perf_counter()
        for _ in range(spf):
            scene.step()
        t_phys += time.perf_counter() - t
        t = time.perf_counter()
        rgb = cam.read().rgb
        t_render += time.perf_counter() - t
        if not args.no_capture:
            t = time.perf_counter()
            a = rgb.cpu().numpy() if rgb.ndim == 3 else rgb[0].cpu().numpy()
            t_read += time.perf_counter() - t
            frames.append(a)
    gen_s = time.perf_counter() - t0

    n = n_frames
    if args.no_capture:
        print(f"[fps-nocapture] genesis-nyx: {n} frames in {gen_s:.2f}s = {n / gen_s:.1f} gen-fps",
              flush=True)
    else:
        import imageio.v2 as imageio
        out = Path(__file__).parent / "cube_rt_genesis_nyx.mp4"
        imageio.mimsave(out, frames, fps=FPS)
        print(f"wrote {out}  ({len(frames)} frames @ {FPS}fps)", flush=True)
        print(f"[rt-native] genesis-nyx: {n} frames in {gen_s:.2f}s = "
              f"{n / gen_s:.1f} fps ({1000 * gen_s / n:.0f} ms/frame @ {SPP} spp, {W}x{H})",
              flush=True)
    print(f"[segments] genesis-nyx: physics={1e3 * t_phys / n:.2f}ms "
          f"render={1e3 * t_render / n:.2f}ms readback={1e3 * t_read / n:.2f}ms", flush=True)


if __name__ == "__main__":
    main()

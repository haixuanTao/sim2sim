"""Measure how Genesis-Nyx's frame cost scales with resolution.

Why this exists: the cube ``rt_native`` row is measured at 480x368 because that
is the resolution every other native-RT row uses, and the panel is only
meaningful if the rows are comparable. But at 480x368 the frame is so small
that fixed per-frame overhead -- not ray tracing -- dominates, so that row
*understates* Nyx. This sweep is the evidence: cost per sample falls sharply as
the frame grows, which is the signature of an overhead-bound measurement.

Scene matches genesis_nyx_native.py (same Rough surfaces, same area light), so
the 480x368 point here lines up with the published row.

Run:  ~/rt_build/nyx-venv/bin/python examples/cube_drop/tools/nyx_res_sweep.py
      [--out nyx_res_sweep.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SPP = 64
DT = 1.0 / 120.0
WARMUP = 5
FRAMES = 30
RESOLUTIONS = [(480, 368), (960, 736), (1280, 960), (1920, 1440), (2560, 1920)]
LIGHTS = [{"pos": (2.0, -1.0, 6.0), "color": (1.0, 1.0, 1.0),
           "intensity": 12.0, "radius": 1.5}]


def measure(gs, NyxCameraOptions, res: tuple[int, int]) -> dict:
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=False)
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.65, 0.65, 0.65)))
    scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(0.0, 0.0, 1.5), euler=(12.0, 22.0, 5.0)),
        surface=gs.surfaces.Rough(color=(0.8, 0.25, 0.2)),
    )
    cam = scene.add_sensor(NyxCameraOptions(
        res=res, pos=(2.8, -2.8, 2.0), lookat=(0.0, 0.0, 0.4),
        spp=SPP, denoise=True, lights=LIGHTS))
    scene.build()
    for _ in range(WARMUP):
        scene.step()
        cam.read()

    t_render = 0.0
    for _ in range(FRAMES):
        scene.step()
        t = time.perf_counter()
        rgb = cam.read().rgb
        _ = rgb[0, 0].cpu()  # force the GPU to finish before stopping the clock
        t_render += time.perf_counter() - t

    ms = 1e3 * t_render / FRAMES
    px = res[0] * res[1]
    return {"resolution": f"{res[0]}x{res[1]}", "pixels": px, "spp": SPP,
            "render_ms": round(ms, 2), "render_fps": round(1000.0 / ms, 1),
            "ns_per_sample": round(ms * 1e6 / (px * SPP), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "nyx_res_sweep.json"))
    args = ap.parse_args()

    import genesis as gs

    rows = []
    for res in RESOLUTIONS:
        gs.init(backend=gs.gpu, logging_level="error")
        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions  # after gs.init()

        row = measure(gs, NyxCameraOptions, res)
        rows.append(row)
        print(f"{row['resolution']:>10}  {row['render_ms']:6.2f} ms  "
              f"{row['render_fps']:6.1f} fps  {row['ns_per_sample']:.3f} ns/sample", flush=True)
        gs.destroy()

    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()

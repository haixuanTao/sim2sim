"""Measure how much the *lighting choice* costs, in the one RT engine that runs here.

Why this exists: the three native-RT rows do not light the same scene.

    genesis (LuisaRender)  area light radius 1.5 + emissive env  -> soft shadows
    isaac                  DomeLight + DistantLight              -> dome env sampling
    nexus                  add_directional_light                 -> hard shadows
    genesis_nyx            area light radius 1.5 (matched to LuisaRender)

That makes the rt_native column indicative rather than exact, and the honest
question is how much the disagreement is worth. LuisaRender/Isaac/Nexus cannot
be re-measured on this box (no from-source LuisaRender build, no isaac-venv, no
nexus3d), so this sweeps the lighting regimes on Nyx instead and puts a number
on the effect.

Answer, as of 2026-07-16 on the RTX 5090: ~10-15% at a tracing-bound resolution
and inside the noise at the row's 480x368 -- so lighting choice does not explain
cross-engine gaps of 24x/58x. Soft vs hard shadows costs Nyx nothing measurable.

Run:  ~/rt_build/nyx-venv/bin/python examples/cube_drop/tools/nyx_light_sweep.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SPP = 64
DT = 1.0 / 120.0
FRAMES = 30
# The row's resolution (overhead-bound; see nyx_res_sweep.py) and one where ray
# tracing actually dominates, because the answer differs between them.
RESOLUTIONS = [(480, 368), (1920, 1440)]
BASE = {"pos": (2.0, -1.0, 6.0), "color": (1.0, 1.0, 1.0), "intensity": 12.0}
REGIMES = {
    "unlit": None,
    "point (radius 0, hard shadow)": [{**BASE, "radius": 0.0}],
    "area (radius 1.5, soft shadow)": [{**BASE, "radius": 1.5}],
    "area (radius 4.0, very soft)": [{**BASE, "radius": 4.0}],
    "3 area lights (radius 1.5)": [
        {**BASE, "radius": 1.5},
        {"pos": (-3.0, 2.0, 4.0), "color": (0.6, 0.7, 1.0), "intensity": 8.0, "radius": 1.5},
        {"pos": (0.0, -4.0, 2.0), "color": (1.0, 0.8, 0.6), "intensity": 6.0, "radius": 1.5},
    ],
}


def measure(gs, NyxCameraOptions, res, lights) -> float:
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=False)
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.65, 0.65, 0.65)))
    scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(0.0, 0.0, 1.5), euler=(12.0, 22.0, 5.0)),
        surface=gs.surfaces.Rough(color=(0.8, 0.25, 0.2)),
    )
    kw = dict(res=res, pos=(2.8, -2.8, 2.0), lookat=(0.0, 0.0, 0.4), spp=SPP, denoise=True)
    if lights:
        kw["lights"] = lights
    cam = scene.add_sensor(NyxCameraOptions(**kw))
    scene.build()
    for _ in range(5):
        scene.step()
        cam.read()

    total = 0.0
    for _ in range(FRAMES):
        scene.step()
        t = time.perf_counter()
        rgb = cam.read().rgb
        _ = rgb[0, 0].cpu()  # force GPU completion inside the timed window
        total += time.perf_counter() - t
    return 1e3 * total / FRAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "nyx_light_sweep.json"))
    args = ap.parse_args()

    import genesis as gs

    out = []
    for res in RESOLUTIONS:
        for tag, lights in REGIMES.items():
            gs.init(backend=gs.gpu, logging_level="error")
            from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions  # after gs.init()

            ms = measure(gs, NyxCameraOptions, res, lights)
            out.append({"resolution": f"{res[0]}x{res[1]}", "regime": tag, "spp": SPP,
                        "render_ms": round(ms, 2), "render_fps": round(1000.0 / ms, 1)})
            print(f"{res[0]:>4}x{res[1]:<4}  {tag:32s} {ms:6.2f} ms", flush=True)
            gs.destroy()

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()

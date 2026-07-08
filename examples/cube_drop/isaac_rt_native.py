"""Cube drop rendered by Isaac Sim's OWN RTX path tracer (native RT, no Mitsuba).

Unlike rt_record.py + rt_render.py (shared external tracer), this uses the
Omniverse RTX renderer in PathTracing mode — the one engine in the lineup whose
native renderer is a ray tracer. Physics is PhysX; the camera sensor grabs the
path-traced frames.

Run (Isaac needs its own venv, and Kit swallows stdout — results go to a file):
  ~/rt_build/isaac-venv/bin/python examples/cube_drop/isaac_rt_native.py --outdir /tmp/isaac_frames
Then encode:
  ffmpeg -y -framerate 30 -i /tmp/isaac_frames/f_%03d.png -pix_fmt yuv420p cube_rt_isaac_native.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

N_FRAMES = 150
FRAME_DT = 1.0 / 30.0
HALF = 0.15
START_Z = 1.5
EULER_DEG = (12.0, 22.0, 5.0)
W, H, SPP = 480, 368, 64


def euler_to_quat_wxyz(deg):
    r, p, y = (math.radians(d) for d in deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": W, "height": H})

    import carb
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    from PIL import Image
    from pxr import UsdLux
    import omni.usd

    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", SPP)
    s.set("/rtx/pathtracing/totalSpp", SPP)

    dt = 1.0 / 240.0
    world = World(physics_dt=dt, rendering_dt=FRAME_DT, stage_units_in_meters=1.0)
    GroundPlane(prim_path="/World/ground", z_position=0.0)
    cube = DynamicCuboid(
        prim_path="/World/cube",
        position=np.array([0.0, 0.0, START_Z]),
        orientation=euler_to_quat_wxyz(EULER_DEG),
        size=2 * HALF,
        color=np.array([0.8, 0.25, 0.2]),
        mass=1.0,
    )
    world.scene.add(cube)

    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(0.53)

    cam = Camera(prim_path="/World/cam", resolution=(W, H))
    world.reset()
    cam.initialize()
    set_camera_view(
        eye=[3.2, -3.2, 2.2], target=[0.0, 0.0, 0.35], camera_prim_path="/World/cam"
    )

    # warm up renderer / fill sensor pipeline
    for _ in range(8):
        world.render()

    spf = max(1, round(FRAME_DT / dt))
    phys_s = rend_s = 0.0
    for i in range(N_FRAMES):
        t0 = time.perf_counter()
        for _ in range(spf):
            world.step(render=False)
        t1 = time.perf_counter()
        world.render()  # one render tick = SPP path-traced samples
        rgba = cam.get_rgba()
        t2 = time.perf_counter()
        phys_s += t1 - t0
        rend_s += t2 - t1
        Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(
            os.path.join(args.outdir, f"f_{i:03d}.png")
        )

    with open(os.path.join(args.outdir, "result.txt"), "w") as f:
        f.write(
            f"[rt-native] isaac: physics {spf * N_FRAMES / phys_s:,.0f} steps/s | "
            f"RTX path trace {N_FRAMES / rend_s:.1f} fps "
            f"({1000 * rend_s / N_FRAMES:.0f} ms/frame @ {SPP} spp, {W}x{H})\n"
        )
    app.close()


if __name__ == "__main__":
    main()

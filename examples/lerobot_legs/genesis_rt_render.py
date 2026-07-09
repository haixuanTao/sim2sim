"""LeRobot-legs robot demo — Genesis's OWN ray tracer (LuisaRender).

Same robot, gains, and grid-searched standing bias as ``genesis_render.py``,
rendered with ``gs.renderers.RayTracer`` instead of the raster camera.
Needs the LuisaRenderPy build (see cube_drop README / ~/rt_build). 32 spp at
640x480 to match the Isaac LeRobot row.

Run:  .venv/bin/python examples/lerobot_legs/genesis_rt_render.py --outdir /tmp/gs_lerobot
Then: ffmpeg -y -framerate 30 -i /tmp/gs_lerobot/f_%03d.png -pix_fmt yuv420p lerobot_genesis_rt.mp4
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

DURATION_S = 5.0
FPS = 30
W, H, SPP = 640, 480, 32
DT = 0.005

ROBOT_XML = (
    Path.home()
    / "Documents/work/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms/mjcf/robot.xml"
)

GAINS = {
    "hipz": (30, 3),
    "hipx": (40, 3),
    "hipy": (60, 4),
    "knee": (60, 4),
    "ankley": (20, 1.5),
    "anklex": (20, 1.5),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import genesis as gs
    from PIL import Image

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
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
    robot = scene.add_entity(gs.morphs.MJCF(file=str(ROBOT_XML), pos=(0.0, 0.0, 0.72)))
    cam = scene.add_camera(
        res=(W, H), pos=(1.3, -1.3, 0.85), lookat=(0.0, 0.0, 0.4), fov=45, spp=SPP, GUI=False
    )
    scene.build()

    # Same per-DOF gains and grid-searched bias as genesis_render.py.
    dofs, kps, kvs, targets = [], [], [], []
    for side in ("left", "right"):
        for jname, (kp, kv) in GAINS.items():
            joint = robot.get_joint(f"{jname}_{side}")
            dofs.append(joint.dof_idx_local)
            kps.append(kp)
            kvs.append(kv)
            bias = 0.25 if side == "right" else -0.25
            targets.append(bias if jname == "hipy" else 0.0)
    robot.set_dofs_kp(np.array(kps, dtype=np.float32), dofs)
    robot.set_dofs_kv(np.array(kvs, dtype=np.float32), dofs)
    target = np.array(targets, dtype=np.float32)

    n_frames = int(DURATION_S * FPS)
    spf = max(1, round((1.0 / FPS) / DT))
    phys_s = rend_s = 0.0
    for i in range(n_frames):
        t0 = time.perf_counter()
        for _ in range(spf):
            robot.control_dofs_position(target, dofs)
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
        f"[rt-native] genesis-lerobot: physics {spf * n_frames / phys_s:,.0f} steps/s | "
        f"LuisaRender path trace {n_frames / rend_s:.1f} fps "
        f"({1000 * rend_s / n_frames:.0f} ms/frame @ {SPP} spp, {W}x{H})"
    )
    print(msg)
    with open(os.path.join(args.outdir, "result.txt"), "w") as f:
        f.write(msg + "\n")
    print(f"final base height: {float(robot.get_pos()[2]):.3f} m")


if __name__ == "__main__":
    main()

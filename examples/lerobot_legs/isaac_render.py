"""LeRobot-legs robot demo — Isaac Sim backend, real bipedal-platform asset.

Imports the platform's URDF (PhysX RTX render, PathTracing mode like the cube
demo — the realtime renderer produces noise-only frames on this driver).
Import quirks handled here, found the hard way:

- The URDF importer crashes with "Used null prim" when one link references the
  same mesh file twice (screws/motors); we symlink every mesh reference to a
  unique name first.
- Drive gains set through ``ArticulationController.set_gains`` land in PhysX
  ~57x weaker than the MJCF kp/kv (deg/rad convention); we scale by 180/pi so
  the effective stiffness matches the MuJoCo/Genesis demos.
- The importer leaves joint limits at +-180 deg and drops frictionloss; both
  are re-applied from the MJCF values.

Unlike the MuJoCo/Genesis demos the base is PINNED (fix_base): the URDF zero
pose rests on plantar-flexed ankles (tiptoes), and no static hip/ankle bias
keeps the robot upright under PhysX's contact model (grid-searched hipy/ankley
over +-0.4 rad, both axis conventions, 1x-10x gains — every combination tips
over). The servos still hold the pose; only the free-fall balance is omitted.

Run (Isaac needs its own venv, and Kit swallows stdout — results go to a file):
  ~/rt_build/isaac-venv/bin/python examples/lerobot_legs/isaac_render.py --outdir /tmp/isaac_lerobot
Then encode:
  ffmpeg -y -framerate 30 -i /tmp/isaac_lerobot/f_%03d.png -pix_fmt yuv420p lerobot_isaac.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path

import numpy as np

DURATION_S = 5.0
FPS = 30
W, H, SPP = 640, 480, 32
DT = 1.0 / 200.0
R2D = 180.0 / math.pi

ASSET_DIR = (
    Path.home() / "Documents/work/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms"
)
URDF = ASSET_DIR / "urdf/robot.urdf"
STAND_Z = 0.7145  # root height with feet on the floor (from the import bbox)

# kp/kv per joint family, from the MJCF <actuator> block
GAINS = {
    "hipz": (30, 3),
    "hipx": (40, 3),
    "hipy": (60, 4),
    "knee": (60, 4),
    "ankley": (20, 1.5),
    "anklex": (20, 1.5),
}
# joint limits (rad) and frictionloss (N*m) from the MJCF robot.xml
LIM = {"hipz": (-0.349, 0.349), "hipx": (-0.349, 0.349), "hipy": (-1.047, 1.047), "knee": (-0.524, 0.524)}
LIM_SIDE = {
    ("ankley", "right"): (-0.175, 0.349),
    ("ankley", "left"): (-0.349, 0.175),
    ("anklex", "right"): (-0.175, 0.175),
    ("anklex", "left"): (-0.175, 0.175),
}
FRIC = {"hipz": 1.35, "hipx": 1.16, "hipy": 1.31, "knee": 1.0, "ankley": 0.17, "anklex": 0.26}


def make_unique_mesh_urdf(outdir: Path) -> Path:
    """Symlink every mesh reference to a unique name (importer chokes on dupes)."""
    uniq = outdir / "uniq_assets"
    uniq.mkdir(parents=True, exist_ok=True)
    counter = [0]

    def sub(m: re.Match) -> str:
        link = uniq / f"m{counter[0]:03d}.stl"
        counter[0] += 1
        link.unlink(missing_ok=True)
        link.symlink_to(ASSET_DIR / "urdf/assets" / m.group(1))
        return f'<mesh filename="{link}"/>'

    patched = outdir / "robot_uniq.urdf"
    patched.write_text(
        re.sub(r'<mesh filename="package://assets/([^"]*)"\s*/?>', sub, URDF.read_text())
    )
    return patched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    patched = make_unique_mesh_urdf(outdir)

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": W, "height": H})

    import carb
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")

    from isaacsim.core.api import World
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    from PIL import Image
    from pxr import PhysxSchema, UsdLux, UsdPhysics
    import omni.usd

    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", SPP)
    s.set("/rtx/pathtracing/totalSpp", SPP)

    # The importer anchors the fixed base at the URDF origin and wins over any
    # later transform, so the floor goes DOWN to meet the feet instead.
    world = World(physics_dt=DT, rendering_dt=1.0 / FPS, stage_units_in_meters=1.0)
    GroundPlane(prim_path="/World/ground", z_position=-STAND_Z, color=np.array([0.25, 0.25, 0.28]))

    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.fix_base = True  # see module docstring
    cfg.import_inertia_tensor = True
    ok, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(patched), import_config=cfg,
        get_articulation_root=True,
    )
    if not ok or not prim_path:
        raise RuntimeError(f"URDF import failed (ok={ok}, prim_path={prim_path})")

    stage = omni.usd.get_context().get_stage()
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint):
            base, side = p.GetName().rsplit("_", 1)
            if base not in GAINS:
                continue
            lo, hi = LIM_SIDE.get((base, side), LIM.get(base))
            j = UsdPhysics.RevoluteJoint(p)
            j.CreateLowerLimitAttr(0.0).Set(lo * R2D)
            j.CreateUpperLimitAttr(0.0).Set(hi * R2D)
            PhysxSchema.PhysxJointAPI.Apply(p).CreateJointFrictionAttr(0.0).Set(FRIC[base])

    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(300.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(1000.0)
    sun.CreateAngleAttr(0.53)

    robot = SingleArticulation(prim_path)
    world.scene.add(robot)
    world.reset()

    names = list(robot.dof_names)
    kps = np.array([GAINS[nm.rsplit("_", 1)[0]][0] for nm in names]) * R2D
    kds = np.array([GAINS[nm.rsplit("_", 1)[0]][1] for nm in names]) * R2D
    robot.get_articulation_controller().set_gains(kps=kps, kds=kds)
    robot.apply_action(ArticulationAction(joint_positions=np.zeros(len(names))))

    cam = Camera(prim_path="/World/cam", resolution=(W, H))
    cam.initialize()
    set_camera_view(
        eye=[3.0, -3.0, 1.1 - STAND_Z],
        target=[0.0, 0.0, 0.45 - STAND_Z],
        camera_prim_path="/World/cam",
    )

    for _ in range(8):  # warm up render pipeline
        world.render()

    n_frames = int(DURATION_S * FPS)
    spf = max(1, round((1.0 / FPS) / DT))
    phys_s = rend_s = 0.0
    for i in range(n_frames):
        t0 = time.perf_counter()
        for _ in range(spf):
            world.step(render=False)
        t1 = time.perf_counter()
        world.render()
        rgba = cam.get_rgba()
        t2 = time.perf_counter()
        phys_s += t1 - t0
        rend_s += t2 - t1
        Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(
            outdir / f"f_{i:03d}.png"
        )

    err = float(np.abs(robot.get_joint_positions()).max())
    (outdir / "result.txt").write_text(
        f"[fps] lerobot/isaac: physics {spf * n_frames / phys_s:,.0f} steps/s | "
        f"RTX path trace {n_frames / rend_s:.1f} fps ({1000 * rend_s / n_frames:.0f} ms/frame @ {SPP} spp, {W}x{H})\n"
        f"max joint deviation from zero pose: {err:.3f} rad\n"
    )
    app.close()


if __name__ == "__main__":
    main()

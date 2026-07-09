"""LeRobot-legs robot demo — Genesis backend, real bipedal-platform asset.

Loads the same LeRobot bipedal platform MJCF as ``mujoco_render.py`` /
``nexus_render.py`` (robot.xml with a freejoint), adds a ground plane, and
holds the standing pose with Genesis's per-DOF PD control using the same gains
as the MJCF <actuator> block (kp 20-60) and the same hip/ankle-pitch bias.
Renders off-screen to an MP4.

Run:  python examples/lerobot_legs/genesis_render.py
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

DURATION_S = 5.0
FPS = 30
W, H = 640, 480
DT = 0.005  # matches sim_scene_safe.xml

ROBOT_XML = (
    Path.home()
    / "Documents/work/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms/mjcf/robot.xml"
)

# (joint suffix -> kp, kv) copied from sim_scene_safe.xml <actuator>
GAINS = {
    "hipz": (30, 3),
    "hipx": (40, 3),
    "hipy": (60, 4),
    "knee": (60, 4),
    "ankley": (20, 1.5),
    "anklex": (20, 1.5),
}


def main() -> None:
    import genesis as gs

    try:
        gs.init(backend=gs.gpu)
    except Exception as e:
        print(f"[genesis] GPU init failed ({e}); falling back to CPU backend")
        gs.init(backend=gs.cpu)

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(ROBOT_XML), pos=(0.0, 0.0, 0.72)))
    cam = scene.add_camera(
        res=(W, H), pos=(1.3, -1.3, 0.85), lookat=(0.0, 0.0, 0.4), fov=45, GUI=False
    )
    scene.build()

    # Per-DOF PD gains from the MJCF <actuator> block. The MuJoCo bias
    # (hipy/ankley +-0.15) is not stable under Genesis's contact model —
    # grid-searched here: hipy +-0.25 with zero ankle bias holds z~=0.70.
    names, dofs, kps, kvs, targets = [], [], [], [], []
    for side in ("left", "right"):
        for jname, (kp, kv) in GAINS.items():
            name = f"{jname}_{side}"
            joint = robot.get_joint(name)
            names.append(name)
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
    frames = []
    t0 = time.perf_counter()
    for _ in range(n_frames):
        for _ in range(spf):
            robot.control_dofs_position(target, dofs)
            scene.step()
        out = cam.render()
        rgb = out[0] if isinstance(out, tuple) else out
        frames.append(rgb[:, :, :3])
    gen_s = time.perf_counter() - t0

    out_path = Path(__file__).parent / "lerobot_genesis.mp4"
    imageio.mimsave(out_path, frames, fps=FPS)
    z = float(robot.get_pos()[2])
    print(f"wrote {out_path}  ({len(frames)} frames @ {FPS}fps)")
    print(f"[fps] lerobot/genesis: {len(frames)} frames in {gen_s:.2f}s = {len(frames) / gen_s:.1f} gen-fps")
    print(f"final base height: {z:.3f} m")


if __name__ == "__main__":
    main()

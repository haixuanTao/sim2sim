"""LeRobot-legs robot demo — Genesis with the Nyx path tracer (gs-nyx-plugin).

Same scene as ``genesis_render.py`` (real bipedal-platform MJCF, per-DOF PD
stance hold) but rendered by Genesis's new official renderer plugin instead of
the legacy from-source LuisaRender path: Nyx is a camera *sensor* that path
traces on the GPU and returns CUDA-resident uint8 tensors (batched across envs
for batched scenes). Wheel-installable — no source build, no Taichi/NVRTC
fallback.

Requirements (see ~/rt_build/nyx-venv): genesis git-main (the released 1.2.2
lacks ``gs.qd_float``), torch, ``gs-nyx-plugin``; the plugin must be imported
AFTER ``gs.init()``.

Run:  ~/rt_build/nyx-venv/bin/python examples/lerobot_legs/genesis_nyx_render.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

DURATION_S = 5.0
FPS = 30
W, H = 640, 480
SPP = 32  # matches the other LeRobot RT rows
DT = 0.005
# Same area light as genesis_rt_render.py's RayTracer(lights=...): a radius, so
# it casts a soft shadow rather than a hard one.
LIGHTS = [{"pos": (2.0, -1.0, 6.0), "color": (1.0, 1.0, 1.0),
           "intensity": 12.0, "radius": 1.5}]

ROBOT_XML = (
    Path.home()
    / "Documents/work/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms/mjcf/robot.xml"
)

# (joint suffix -> kp, kv) copied from sim_scene_safe.xml <actuator>.
GAINS = {
    "hipz": (30, 3), "hipx": (40, 3), "hipy": (60, 4),
    "knee": (60, 4), "ankley": (20, 1.5), "anklex": (20, 1.5),
}


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
    # Surface + light mirror genesis_rt_render.py's LuisaRender scene, so the two
    # lerobot_rt rows are the same render, not just the same geometry. Without
    # LIGHTS the sensor warns and every camera ray terminates on a constant
    # ambient env: no shadow rays, no GI -- a path trace of nothing.
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.65, 0.65, 0.65)))
    robot = scene.add_entity(gs.morphs.MJCF(file=str(ROBOT_XML), pos=(0.0, 0.0, 0.72)))
    cam = scene.add_sensor(NyxCameraOptions(
        res=(W, H), pos=(1.3, -1.3, 0.85), lookat=(0.0, 0.0, 0.4),
        spp=SPP, denoise=True, lights=LIGHTS))
    scene.build()

    dofs, kps, kvs, targets = [], [], [], []
    for side in ("left", "right"):
        for jname, (kp, kv) in GAINS.items():
            joint = robot.get_joint(f"{jname}_{side}")
            idx = getattr(joint, "dofs_idx_local", None)
            if idx is None:
                idx = joint.dof_idx_local
            dofs.append(idx if np.isscalar(idx) else idx[0])
            kps.append(kp)
            kvs.append(kv)
            bias = 0.25 if side == "right" else -0.25
            targets.append(bias if jname == "hipy" else 0.0)
    robot.set_dofs_kp(np.array(kps, dtype=np.float32), dofs)
    robot.set_dofs_kv(np.array(kvs, dtype=np.float32), dofs)
    target = np.array(targets, dtype=np.float32)

    n_frames = int(DURATION_S * FPS)
    spf = max(1, round((1.0 / FPS) / DT))

    # Warmup (kernel JIT + first render allocations) outside the timers.
    for _ in range(5):
        robot.control_dofs_position(target, dofs)
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
        rgb = cam.read().rgb  # CUDA-resident uint8
        t_render += time.perf_counter() - t
        if not args.no_capture:
            t = time.perf_counter()
            a = rgb.cpu().numpy() if rgb.ndim == 3 else rgb[0].cpu().numpy()
            t_read += time.perf_counter() - t
            frames.append(a)
    gen_s = time.perf_counter() - t0

    n = n_frames
    tag = "lerobot/genesis-nyx"
    if args.no_capture:
        print(f"[fps-nocapture] {tag}: {n} frames in {gen_s:.2f}s = {n / gen_s:.1f} gen-fps",
              flush=True)
    else:
        import imageio.v2 as imageio
        out = Path(__file__).parent / "lerobot_genesis_nyx.mp4"
        imageio.mimsave(out, frames, fps=FPS)
        print(f"wrote {out}  ({len(frames)} frames @ {FPS}fps)", flush=True)
        print(f"[fps] {tag}: {n} frames in {gen_s:.2f}s = {n / gen_s:.1f} gen-fps", flush=True)
    print(f"[segments] {tag}: physics={1e3 * t_phys / n:.2f}ms render={1e3 * t_render / n:.2f}ms "
          f"readback={1e3 * t_read / n:.2f}ms", flush=True)


if __name__ == "__main__":
    main()

"""Cube-drop rendered by Nexus/kiss3d's GPU path tracer (wgpu, kiss3d 0.45).

Uses the ``NexusViewer.raytrace_frame()`` / ``render()`` API added in the
feat/python-rt-render change of dimforge/nexus: each video frame is path
traced by calling ``raytrace_frame()`` repeatedly so samples accumulate, then
read back with ``render()`` as an ``(H, W, 3)`` uint8 numpy array.

Run:  python examples/cube_drop/nexus_rt_native.py
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v2 as imageio
import nexus3d as nx

FPS = 30
N_FRAMES = 150
ACCUM_FRAMES = 8  # raytrace_frame() calls per video frame (sample accumulation)
SAMPLES_PER_FRAME = 8  # spp added per raytrace_frame() call
W, H = 480, 368  # match the genesis/isaac native-RT demos


def main() -> None:
    viewer = nx.NexusViewer(W, H)
    pipeline = nx.NexusPipeline()
    pipeline.preload_pipelines(viewer)
    state = nx.NexusState()

    viewer.set_up_axis(nx.Vec3.Y)

    cube_body = (
        nx.RigidBodyBuilder.dynamic()
        .translation(nx.Vec3(0.0, 3.0, 0.0))
        .rotation(nx.Vec3(0.3, 0.4, 0.2))
        .angvel(nx.Vec3(1.5, -1.0, 0.5))
        .build()
    )
    # Rapier defaults to perfectly inelastic contact (restitution 0), unlike the
    # soft-contact engines (MuJoCo/Genesis) that return a little energy — match
    # their visible micro-bounce.
    cube_col = nx.ColliderBuilder.cuboid(0.15, 0.15, 0.15).restitution(0.3).build()
    cube_h = state.insert_rigid_body(cube_body, cube_col)
    viewer.insert_shape_with_color(
        cube_h, cube_col.shared_shape(), nx.Pose.IDENTITY, nx.Vec4(0.90, 0.32, 0.22, 1.0)
    )

    ground_body = nx.RigidBodyBuilder.fixed().translation(nx.Vec3(0.0, -0.5, 0.0)).build()
    ground_col = nx.ColliderBuilder.cuboid(6.0, 0.5, 6.0).restitution(0.3).build()
    ground_h = state.insert_rigid_body(ground_body, ground_col)
    viewer.insert_shape(ground_h, ground_col.shared_shape(), nx.Pose.IDENTITY)

    state.finalize(viewer)
    state.set_rbd_gravity(viewer, nx.Vec3(0.0, -9.81, 0.0))
    # Each solver step advances 1/60 s (verified via body_pose()); at 30 fps
    # video we need two steps per frame or the clip plays in 0.5x slow motion.
    state.set_rbd_steps_per_frame(2)
    viewer.add_directional_light(nx.Vec3(1.0, -3.0, 2.0))
    viewer.set_camera(nx.Vec3(6.0, 4.0, 6.0), nx.Vec3(0.0, 0.6, 0.0))

    viewer.set_raytracer_samples_per_frame(SAMPLES_PER_FRAME)
    viewer.set_raytracer_max_bounces(6)
    viewer.set_raytracer_denoise(True)

    ts = nx.GpuTimestamps(viewer, 2048)

    frames = []
    rend_s = 0.0
    while len(frames) < N_FRAMES:
        pipeline.simulate(viewer, state, ts)
        viewer.sync(state, ts)
        t0 = time.perf_counter()
        ok = all(viewer.raytrace_frame() for _ in range(ACCUM_FRAMES))
        rend_s += time.perf_counter() - t0
        if not ok:
            break
        frames.append(viewer.render())  # (H, W, 3) uint8 numpy array

    out = Path(__file__).parent / "cube_rt_nexus_native.mp4"
    imageio.mimsave(out, frames, fps=FPS)
    spp = ACCUM_FRAMES * SAMPLES_PER_FRAME
    print(
        f"wrote {out} ({len(frames)} frames @ {FPS}fps) | "
        f"path trace {len(frames) / rend_s:.1f} fps ({1000 * rend_s / max(len(frames), 1):.0f} ms/frame @ {spp} spp)"
    )


if __name__ == "__main__":
    main()

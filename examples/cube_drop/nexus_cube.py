"""Cube-drop smoke demo — Nexus (dimforge, Rapier-on-GPU / WebGPU).

Uses ``NexusViewer.render()`` to read each rendered frame back as an
``(H, W, 3)`` uint8 numpy array — exactly like ``mujoco.Renderer.render()`` or
a Genesis ``camera.render()`` — and encodes them to an MP4.

Requires a build of ``dimforge-nexus3d`` that includes ``render()`` (added in
the feat/viewer-render-export change; not in the released wheel yet). Nexus needs a
GPU (WebGPU) and opens a viewer window, but frames come from the engine's own
framebuffer rather than any screen capture.

Run:  python examples/cube_drop/nexus_cube.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import nexus3d as nx

FPS = 30
N_FRAMES = 180  # frames captured after the drop starts
W, H = 640, 480  # match the other backends' raster demos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true",
                    help="step physics with the Rapier CPU backend (needs a wheel built with --features cpu)")
    args = ap.parse_args()

    viewer = nx.NexusViewer(W, H)
    viewer.set_draw_ui(False)  # keep the egui panel out of captured frames
    if args.cpu:
        viewer.with_cpu()
        viewer.init_backend()
    pipeline = nx.NexusPipeline()
    pipeline.preload_pipelines(viewer)  # ~1 min: compiles the GPU pipelines
    state = nx.NexusState()

    viewer.set_up_axis(nx.Vec3.Y)  # Rapier is Y-up: gravity -Y, ground normal +Y

    # Falling cube (half-extent 0.15) dropped from 3 m along +Y, with an initial
    # tilt + spin so it tumbles and lands on an edge (like the other backends'
    # demos) rather than perfectly flat.
    cube_body = (
        nx.RigidBodyBuilder.dynamic()
        .translation(nx.Vec3(0.0, 3.0, 0.0))
        .rotation(nx.Vec3(0.3, 0.4, 0.2))  # scaled-axis tilt (radians)
        .angvel(nx.Vec3(1.5, -1.0, 0.5))  # slight tumble
        .build()
    )
    # Rapier defaults to perfectly inelastic contact (restitution 0), unlike the
    # soft-contact engines (MuJoCo/Genesis) that return a little energy — match
    # their visible micro-bounce.
    cube_col = nx.ColliderBuilder.cuboid(0.15, 0.15, 0.15).restitution(0.3).build()
    cube_shape = cube_col.shared_shape()
    cube_h = state.insert_rigid_body(cube_body, cube_col)
    viewer.insert_shape_with_color(
        cube_h, cube_shape, nx.Pose.IDENTITY, nx.Vec4(0.90, 0.32, 0.22, 1.0)
    )

    # Ground: large fixed slab, thin in Y, top surface at y=0.
    ground_body = nx.RigidBodyBuilder.fixed().translation(nx.Vec3(0.0, -0.5, 0.0)).build()
    ground_col = nx.ColliderBuilder.cuboid(6.0, 0.5, 6.0).restitution(0.3).build()
    ground_shape = ground_col.shared_shape()
    ground_h = state.insert_rigid_body(ground_body, ground_col)
    viewer.insert_shape(ground_h, ground_shape, nx.Pose.IDENTITY)

    state.finalize(viewer)
    state.set_rbd_gravity(viewer, nx.Vec3(0.0, -9.81, 0.0))
    viewer.add_directional_light(nx.Vec3(1.0, -3.0, 2.0))
    viewer.set_camera(nx.Vec3(6.0, 4.0, 6.0), nx.Vec3(0.0, 0.6, 0.0))

    ts = nx.GpuTimestamps(viewer, 2048)

    frames = []
    t0 = time.perf_counter()
    while len(frames) < N_FRAMES:
        if not viewer.render_frame():  # draw one frame + pump events
            break
        pipeline.simulate(viewer, state, ts)  # advance the physics
        viewer.sync(state, ts)  # GPU -> renderer
        frames.append(viewer.render())  # framebuffer -> (H, W, 3) uint8
    gen_s = time.perf_counter() - t0

    tag = "nexus-cpu" if args.cpu else "nexus"
    out = Path(__file__).parent / f"cube_{tag.replace('-', '_')}.mp4"
    imageio.mimsave(out, frames, fps=FPS)
    print(f"wrote {out}  ({len(frames)} frames @ {FPS}fps)")
    print(f"[fps] {tag}: {len(frames)} frames in {gen_s:.2f}s = {len(frames) / gen_s:.1f} gen-fps")


if __name__ == "__main__":
    main()

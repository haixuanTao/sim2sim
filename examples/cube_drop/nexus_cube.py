"""Cube-drop smoke demo — Nexus (dimforge, Rapier-on-GPU / WebGPU).

Uses ``NexusViewer.snap_rgb()`` to read each rendered frame back as an
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
    ap.add_argument("--cuda", action="store_true",
                    help="step physics with the native CUDA (cuda-oxide) backend (needs a wheel built with --features cuda)")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="like --cuda, but capture the per-frame solver steps into a CUDA graph after warmup and replay it (one cuGraphLaunch per frame)")
    ap.add_argument("--no-capture", action="store_true",
                    help="skip the frame readback (and the MP4): benchmark the sim+render loop with frames staying on the GPU")
    args = ap.parse_args()

    # Headless: no window, no swapchain — capture is not throttled by the
    # display's vsync (a windowed viewer caps the loop at ~30 fps).
    viewer = nx.NexusViewer(W, H, headless=True)
    viewer.set_draw_ui(False)  # keep the egui panel out of captured frames
    if args.cpu:
        viewer.with_cpu()
        viewer.init_backend()
    elif args.cuda or args.cuda_graph:
        viewer.with_cuda()
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
    # Each solver step advances 1/60 s (verified via body_pose()); at 30 fps
    # video we need two steps per frame or the clip plays in 0.5x slow motion.
    state.set_rbd_steps_per_frame(2)
    # Side-angled sun (Y-up): the old (1,-3,2) travelled with the camera's view
    # direction, so shadows fell hidden behind objects — RT and raster looked
    # identical. Perpendicular horizontal component makes shadows visible.
    viewer.add_directional_light(nx.Vec3(-2.0, -3.0, 1.0))
    viewer.set_camera(nx.Vec3(6.0, 4.0, 6.0), nx.Vec3(0.0, 0.6, 0.0))

    ts = nx.GpuTimestamps(viewer, 2048)

    # CUDA-graph mode: run a few plain frames so buffer sizes stabilize, then
    # capture the per-frame solver-step sequence once; the loop replays it with
    # a single cuGraphLaunch per frame (no per-dispatch host encode).
    graphed = False
    if args.cuda_graph:
        for _ in range(5):
            viewer.render_frame()
            pipeline.simulate(viewer, state, None)
            viewer.sync(state, None)
        graphed = pipeline.capture_cuda_graph(viewer, state)
        assert graphed, "CUDA graph capture failed (not on the CUDA backend?)"

    # Warmup outside the timers (like the Genesis demos): the first frame pays
    # one-off allocation/BVH/staging-buffer setup (~160 ms vs ~18 ms steady).
    for _ in range(5):
        viewer.render_frame()
        if graphed:
            pipeline.replay_cuda_graph()
        else:
            pipeline.simulate(viewer, state, ts)
        viewer.sync(state, ts)
        if not args.no_capture:
            viewer.snap_rgb_async()
    if not args.no_capture:
        viewer.snap_rgb_flush()

    frames = []
    t_phys = t_sync = t_render = t_read = 0.0
    n_loops = 0
    t0 = time.perf_counter()
    while (n_loops if args.no_capture else len(frames)) < N_FRAMES:
        t = time.perf_counter()
        ok = viewer.render_frame()  # draw one frame + pump events
        t_render += time.perf_counter() - t
        if not ok:
            break
        t = time.perf_counter()
        if graphed:
            pipeline.replay_cuda_graph()  # whole solver-step sequence, one launch
        else:
            pipeline.simulate(viewer, state, ts)  # advance the physics
        # Physics is submitted asynchronously; a state read blocks until the
        # solver finishes, so its GPU time is billed to this segment instead of
        # whichever later call happens to drain the queue.
        viewer.body_pose(state, cube_h)
        t_phys += time.perf_counter() - t
        t = time.perf_counter()
        viewer.sync(state, ts)  # GPU -> renderer
        t_sync += time.perf_counter() - t
        # Pipelined readback: returns the previous frame (None on the first
        # call) while this frame's GPU->CPU copy runs in the background.
        if not args.no_capture:
            t = time.perf_counter()
            frame = viewer.snap_rgb_async()
            t_read += time.perf_counter() - t
            if frame is not None:
                frames.append(frame)
        n_loops += 1
    if not args.no_capture:
        frame = viewer.snap_rgb_flush()  # collect the last in-flight frame
        if frame is not None and len(frames) < N_FRAMES:
            frames.append(frame)
    gen_s = time.perf_counter() - t0

    tag = ("nexus-cpu" if args.cpu else "nexus-cuda-graph" if args.cuda_graph else "nexus-cuda" if args.cuda else "nexus")
    if args.no_capture:
        print(f"[fps-nocapture] {tag}: {n_loops} frames in {gen_s:.2f}s = {n_loops / gen_s:.1f} gen-fps")
    else:
        out = Path(__file__).parent / f"cube_{tag.replace('-', '_')}.mp4"
        imageio.mimsave(out, frames, fps=FPS)
        print(f"wrote {out}  ({len(frames)} frames @ {FPS}fps)")
        print(f"[fps] {tag}: {len(frames)} frames in {gen_s:.2f}s = {len(frames) / gen_s:.1f} gen-fps")
    n = max(n_loops, 1)
    print(f"[segments] {tag}: physics={1e3 * t_phys / n:.2f}ms sync={1e3 * t_sync / n:.2f}ms "
          f"render={1e3 * t_render / n:.2f}ms readback={1e3 * t_read / n:.2f}ms")


if __name__ == "__main__":
    main()

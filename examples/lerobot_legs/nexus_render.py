"""LeRobot-legs robot demo — Nexus (dimforge, Rapier-on-GPU / WebGPU).

Loads the bundled 12-DOF LeRobot-legs biped MJCF with ``NexusState.insert_mjcf``
(which auto-registers the render shapes, floor and camera) and records an MP4
via ``NexusViewer.snap_rgb()``. The nexus bindings expose no torque control yet,
so unlike ``mujoco_stand.py`` there is no PD stance hold — the robot starts in
its standing pose and collapses passively, which still exercises the full
multibody pipeline (free-fall + joint limits + contacts) and the renderer.

With ``--rt`` each frame is path traced by kiss3d's GPU tracer instead of
rasterized (needs the feat/python-rt-render bindings).

Run:  python examples/lerobot_legs/nexus_render.py [--rt]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import nexus3d as nx

DURATION_S = 5.0
FPS = 30
W, H = 640, 480
# One raytrace_frame() call per video frame at the full spp budget: each call
# pays a denoise + tonemap pass, so batching samples into a single call is
# strictly cheaper than accumulating across several calls.
RT_ACCUM, RT_SPP = 1, 32  # 32 spp total, matching the Isaac/Genesis LeRobot RT rows

REPO = Path(__file__).resolve().parents[2]
# Real LeRobot humanoid legs (full STL meshes + position actuators); the
# bundled lerobot_legs.xml is a primitive-geometry stand-in kept as fallback.
REAL_XML = (
    Path.home()
    / "Documents/work/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms/mjcf/scene.xml"
)
XML = REAL_XML if REAL_XML.exists() else REPO / "src/sim2sim/assets/lerobot_legs/lerobot_legs.xml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", action="store_true", help="path trace instead of rasterizing")
    ap.add_argument("--cuda", action="store_true",
                    help="step physics with the native CUDA (cuda-oxide) backend")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="like --cuda, but capture the per-frame solver steps into a CUDA graph and replay it")
    ap.add_argument("--no-capture", action="store_true",
                    help="skip the frame readback (and the MP4): benchmark the sim+render loop with frames staying on the GPU")
    ap.add_argument("--no-readback", action="store_true",
                    help="additionally skip every per-frame GPU->host state read (scene sync + solver fence): "
                         "pure GPU-resident throughput, drained once at the end. The rendered scene keeps its "
                         "initial pose. Implies --no-capture.")
    args = ap.parse_args()
    if args.no_readback:
        args.no_capture = True

    # Headless: no window/swapchain, so capture is not vsync-throttled.
    viewer = nx.NexusViewer(W, H, headless=True)
    viewer.set_draw_ui(False)
    if args.cuda or args.cuda_graph:
        viewer.with_cuda()
    viewer.init_backend()
    pipeline = nx.NexusPipeline()
    pipeline.preload_pipelines(viewer)

    viewer.set_up_axis(nx.Vec3.Z)  # MJCF is Z-up
    state = nx.NexusState()
    info = state.insert_mjcf(viewer, str(XML), render_colliders=False)
    assert info.loaded, f"failed to load {XML}"
    state.finalize(viewer)
    if info.z_up:
        state.set_rbd_gravity(viewer, nx.Vec3(0.0, 0.0, -9.81))
    # Each solver step advances 1/60 s; two per 30 fps frame for real time.
    state.set_rbd_steps_per_frame(2)
    viewer.set_camera(nx.Vec3(1.3, -1.3, 0.7), nx.Vec3(0.0, 0.0, 0.25))
    # The MJCF loader's built-in light is a near-headlight from this camera
    # (shadows hidden behind the robot). Add a side sun (Z-up) so shadows are
    # actually visible in both the raster and path-traced output.
    viewer.add_directional_light(nx.Vec3(-1.0, -1.0, -2.0))

    if args.rt:
        viewer.set_raytracer_samples_per_frame(RT_SPP)
        viewer.set_raytracer_max_bounces(6)
        viewer.set_raytracer_denoise(True)
        # kiss3d defaults to half-resolution tracing while anything moves; trace
        # full resolution so the benchmark matches the Isaac/Genesis rows.
        viewer.set_raytracer_interactive_scale(1.0)

    ts = nx.GpuTimestamps(viewer, 2048)

    # CUDA-graph mode: warm up, then capture the per-frame solver steps once
    # and replay them with a single cuGraphLaunch per frame.
    graphed = False
    if args.cuda_graph:
        for _ in range(5):
            pipeline.simulate(viewer, state, None)
            viewer.sync(state, None)
        graphed = pipeline.capture_cuda_graph(viewer, state)
        assert graphed, "CUDA graph capture failed (not on the CUDA backend?)"

    n_frames = int(DURATION_S * FPS)

    # Warmup outside the timers (like the Genesis demos): the first frame pays
    # one-off allocation/BVH/staging-buffer setup.
    for _ in range(5):
        if graphed:
            pipeline.replay_cuda_graph()
        else:
            pipeline.simulate(viewer, state, ts)
        viewer.sync(state, ts)
        if args.rt:
            viewer.raytrace_frame()
        else:
            viewer.render_frame()
        if not args.no_capture:
            viewer.snap_rgb_async()
    if not args.no_capture:
        viewer.snap_rgb_flush()

    frames = []
    t_phys = t_sync = t_render = t_read = 0.0
    n_loops = 0
    t0 = time.perf_counter()
    while (n_loops if args.no_capture else len(frames)) < n_frames:
        t = time.perf_counter()
        if graphed:
            pipeline.replay_cuda_graph()
        else:
            pipeline.simulate(viewer, state, ts)
        # Physics is submitted asynchronously; a state read blocks until the
        # solver finishes, so its GPU time is billed to this segment instead of
        # whichever later call happens to drain the queue.
        if not args.no_readback:
            viewer.read_multibody_links(state)
        t_phys += time.perf_counter() - t
        t = time.perf_counter()
        if not args.no_readback:
            viewer.sync(state, ts)
        t_sync += time.perf_counter() - t
        t = time.perf_counter()
        if args.rt:
            ok = all(viewer.raytrace_frame() for _ in range(RT_ACCUM))
            # Tracing is also submitted asynchronously; on the WebGPU backend a
            # state read waits on the shared queue, billing the trace here
            # rather than to readback. (On the CUDA backend it only drains the
            # physics stream, so there the trace still lands in readback.)
            if not args.no_readback:
                viewer.read_multibody_links(state)
        else:
            ok = viewer.render_frame()
        t_render += time.perf_counter() - t
        if not ok:
            break
        # Pipelined readback: returns the previous frame (None on the first
        # call) while this frame's GPU->CPU copy runs in the background.
        if not args.no_capture:
            t = time.perf_counter()
            frame = viewer.snap_rgb_async()
            t_read += time.perf_counter() - t
            if frame is not None:
                frames.append(frame)
        n_loops += 1
    if args.no_readback:
        # Everything above only queued GPU work; block once on a state read so
        # the wall clock covers all submitted physics before it stops.
        viewer.read_multibody_links(state)
    if not args.no_capture:
        frame = viewer.snap_rgb_flush()  # collect the last in-flight frame
        if frame is not None and len(frames) < n_frames:
            frames.append(frame)
    gen_s = time.perf_counter() - t0

    backend_tag = "_cuda_graph" if args.cuda_graph else ("_cuda" if args.cuda else "")
    tag = ("nexus_rt" if args.rt else "nexus") + backend_tag
    if args.no_capture:
        print(f"[fps-nocapture] {tag}: {n_loops} frames in {gen_s:.2f}s = {n_loops / gen_s:.1f} gen-fps")
    else:
        out = Path(__file__).parent / f"lerobot_{tag}.mp4"
        imageio.mimsave(out, frames, fps=FPS)
        mode = f"path traced @ {RT_ACCUM * RT_SPP} spp" if args.rt else "rasterized"
        print(f"wrote {out}  ({len(frames)} frames @ {FPS}fps, {mode})")
        print(f"[fps] {tag}: {len(frames)} frames in {gen_s:.2f}s = {len(frames) / gen_s:.1f} gen-fps")
    n = max(n_loops, 1)
    print(f"[segments] {tag}: physics={1e3 * t_phys / n:.2f}ms sync={1e3 * t_sync / n:.2f}ms "
          f"render={1e3 * t_render / n:.2f}ms readback={1e3 * t_read / n:.2f}ms")


if __name__ == "__main__":
    main()

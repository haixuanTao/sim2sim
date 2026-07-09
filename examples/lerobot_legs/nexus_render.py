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
    args = ap.parse_args()

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
    frames = []
    t0 = time.perf_counter()
    while len(frames) < n_frames:
        if graphed:
            pipeline.replay_cuda_graph()
        else:
            pipeline.simulate(viewer, state, ts)
        viewer.sync(state, ts)
        if args.rt:
            if not all(viewer.raytrace_frame() for _ in range(RT_ACCUM)):
                break
        elif not viewer.render_frame():
            break
        # Pipelined readback: returns the previous frame (None on the first
        # call) while this frame's GPU->CPU copy runs in the background.
        frame = viewer.snap_rgb_async()
        if frame is not None:
            frames.append(frame)
    frame = viewer.snap_rgb_flush()  # collect the last in-flight frame
    if frame is not None and len(frames) < n_frames:
        frames.append(frame)
    gen_s = time.perf_counter() - t0

    backend_tag = "_cuda_graph" if args.cuda_graph else ("_cuda" if args.cuda else "")
    tag = ("nexus_rt" if args.rt else "nexus") + backend_tag
    out = Path(__file__).parent / f"lerobot_{tag}.mp4"
    imageio.mimsave(out, frames, fps=FPS)
    mode = f"path traced @ {RT_ACCUM * RT_SPP} spp" if args.rt else "rasterized"
    print(f"wrote {out}  ({len(frames)} frames @ {FPS}fps, {mode})")
    print(f"[fps] {tag}: {len(frames)} frames in {gen_s:.2f}s = {len(frames) / gen_s:.1f} gen-fps")


if __name__ == "__main__":
    main()

"""Benchmark Nexus/kiss3d rendering: rasterizer vs path tracer at several spp.

Reuses the cube-drop scene (cube resting on the ground so the tracer's sample
accumulation isn't reset by motion) and times, over N_TIMED frames each:

- physics step + sync
- rasterized render_frame()
- raytrace_frame() at various samples-per-frame
- snap_rgb() framebuffer readback

Run:  python examples/cube_drop/nexus_rt_bench.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import nexus3d as nx

N_WARMUP = 5
N_TIMED = 30
SPP_SWEEP = [1, 4, 16, 64]
W, H = 480, 368  # match the genesis/isaac native-RT demos


def timeit(fn, n_warmup=N_WARMUP, n=N_TIMED) -> float:
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> None:
    viewer = nx.NexusViewer(W, H)
    pipeline = nx.NexusPipeline()
    pipeline.preload_pipelines(viewer)
    state = nx.NexusState()

    viewer.set_up_axis(nx.Vec3.Y)

    cube_body = nx.RigidBodyBuilder.dynamic().translation(nx.Vec3(0.0, 0.15, 0.0)).build()
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
    viewer.add_directional_light(nx.Vec3(1.0, -3.0, 2.0))
    viewer.set_camera(nx.Vec3(6.0, 4.0, 6.0), nx.Vec3(0.0, 0.6, 0.0))

    ts = nx.GpuTimestamps(viewer, 2048)

    results: dict[str, dict] = {}

    def step_sync():
        pipeline.simulate(viewer, state, ts)
        viewer.sync(state, ts)

    results["physics_step_sync"] = {"ms": 1000 * timeit(step_sync)}

    results["raster_render_frame"] = {"ms": 1000 * timeit(lambda: viewer.render_frame())}

    for spp in SPP_SWEEP:
        viewer.set_raytracer_samples_per_frame(spp)
        ms = 1000 * timeit(lambda: viewer.raytrace_frame())
        results[f"raytrace_{spp}spp"] = {"ms": ms, "spp_per_s": spp * 1000 / ms}

    frame = viewer.snap_rgb()
    results["snap_rgb_readback"] = {
        "ms": 1000 * timeit(lambda: viewer.snap_rgb()),
        "shape": list(frame.shape),
    }

    print(f"\n{'=' * 60}\nNexus/kiss3d render benchmark ({frame.shape[1]}x{frame.shape[0]})")
    for name, r in results.items():
        fps = 1000 / r["ms"]
        extra = f" | {r['spp_per_s']:,.0f} spp/s" if "spp_per_s" in r else ""
        print(f"{name:24s} {r['ms']:8.2f} ms/frame  {fps:7.1f} fps{extra}")

    out = Path(__file__).parent / "nexus_rt_bench.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

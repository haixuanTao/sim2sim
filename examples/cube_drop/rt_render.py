"""Path-trace a recorded cube-drop trajectory (from rt_record.py) with Mitsuba 3.

Shared, backend-agnostic ray tracer: it renders whatever trajectory it's given,
so every simulator's cube is rendered by the *same* path tracer (GPU / OptiX via
the ``cuda_ad_rgb`` variant, CPU ``llvm_ad_rgb`` fallback). Reports the render
timing so it can be benchmarked against the physics timing from rt_record.py.

Run:  python examples/cube_drop/rt_render.py --traj traj_mujoco.npz --out cube_rt_mujoco.mp4 --label mujoco
"""

from __future__ import annotations

import argparse
import time

import imageio.v2 as imageio
import mitsuba as mi
import numpy as np

W, H, SPP, FPS = 480, 360, 96, 30


def _pick_variant() -> str:
    for v in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"):
        if v in mi.variants():
            mi.set_variant(v)
            return v
    raise RuntimeError("no mitsuba variant available")


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _scene(pos, quat, half):
    m = np.eye(4)
    m[:3, :3] = _quat_to_R(quat) * half  # isotropic scale then rotate
    m[:3, 3] = pos
    return mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 10},
            "sensor": {
                "type": "perspective",
                "fov": 40,
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[3.2, -3.2, 2.2], target=[0, 0, 0.35], up=[0, 0, 1]
                ),
                "film": {
                    "type": "hdrfilm",
                    "width": W,
                    "height": H,
                    "rfilter": {"type": "gaussian"},
                },
                "sampler": {"type": "independent", "sample_count": SPP},
            },
            "floor": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().scale(6),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.62, 0.62, 0.66]}},
            },
            "cube": {
                "type": "cube",
                "to_world": mi.ScalarTransform4f(m.tolist()),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.90, 0.32, 0.22]}},
            },
            "sky": {"type": "constant", "radiance": {"type": "rgb", "value": 0.35}},
            "sun": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().translate([3, -2, 5]).scale(1.6),
                "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 18}},
            },
        }
    )


def _to_uint8(img) -> np.ndarray:
    arr = np.array(mi.util.convert_to_bitmap(img))
    return arr[:, :, :3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="sim")
    args = ap.parse_args()

    variant = _pick_variant()
    d = np.load(args.traj)
    pos, quat, half = d["pos"], d["quat"], float(d["half"])
    phys_s = float(d["phys_s"])
    n_steps = int(d["n_steps"])

    frames = []
    t0 = time.perf_counter()
    for i in range(len(pos)):
        img = mi.render(_scene(pos[i], quat[i], half))
        frames.append(_to_uint8(img))
    render_s = time.perf_counter() - t0

    imageio.mimsave(args.out, frames, fps=FPS)
    rt_fps = len(frames) / render_s
    print(f"wrote {args.out}  ({len(frames)} frames, {W}x{H} @ {SPP} spp, mitsuba/{variant})")
    print(
        f"[rt] {args.label}: physics {n_steps / phys_s:,.0f} steps/s | "
        f"raytrace {rt_fps:.1f} fps ({1000 * render_s / len(frames):.0f} ms/frame)"
    )


if __name__ == "__main__":
    main()

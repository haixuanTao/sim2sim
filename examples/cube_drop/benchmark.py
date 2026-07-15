"""Cube-drop benchmark harness: collect per-backend perf into one JSON + webpage.

Aggregates the numbers produced by the standalone cube-drop scripts (raster
gen-fps demos, `rt_record.py` physics, `rt_render.py` shared Mitsuba tracing,
and the three native path tracers) together with the machine's hardware info
into ``benchmark_results.json``, then regenerates ``site/index.html`` via
``make_site.py``.

Already-measured numbers (from the README benchmarks) are embedded as seeds so
the default run is instant; the harness only executes what has no seed.

Run:  .venv/bin/python examples/cube_drop/benchmark.py            # fill gaps
      .venv/bin/python examples/cube_drop/benchmark.py --run none # regen only
      .venv/bin/python examples/cube_drop/benchmark.py --run all  # remeasure
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
VENV_PY = str(REPO / ".venv/bin/python")
ISAAC_PY = str(Path.home() / "rt_build/isaac-venv/bin/python")
OUT_JSON = HERE / "benchmark_results.json"

MEASURED_TODAY = f"measured {datetime.date.today().isoformat()}"

# Numbers already measured on this machine (examples/cube_drop/README.md).
KNOWN_RESULTS: list[dict] = [
    # mode "raster": full per-frame loop (physics + render + readback).
    {"backend": "mujoco", "mode": "raster", "fps": 1684, "resolution": "640x480",
     "pipeline": "CPU physics + EGL render", "video": "cube_mujoco.mp4", "video_url": "https://files.catbox.moe/yct6vx.mp4"},
    {"backend": "mjlab", "mode": "raster", "fps": 565, "resolution": "640x480",
     "pipeline": "GPU (MuJoCo-Warp) + host EGL render", "video": "cube_mjlab.mp4", "video_url": "https://files.catbox.moe/2t1e04.mp4"},
    {"backend": "genesis", "mode": "raster", "fps": 393.9, "fps_nocapture": 486.8, "resolution": "640x480",
     "pipeline": "GPU physics + render (warmed up: taichi/render JIT excluded)", "source": "measured 2026-07-09", "video": "cube_genesis.mp4", "video_url": "https://files.catbox.moe/0k4r8o.mp4"},
    {"backend": "nexus", "mode": "raster", "fps": 92.1, "fps_nocapture": 85.2, "resolution": "640x480",
     "pipeline": "GPU (WebGPU) physics + headless render + pipelined snap_rgb_async() readback (patched kiss3d; was 33 fps vsync-locked windowed)",
     "video": "cube_nexus.mp4", "video_url": "https://files.catbox.moe/q8vl72.mp4", "source": "measured 2026-07-08"},
    {"backend": "nexus_cuda", "mode": "raster", "fps": 16.4, "fps_nocapture": 18.2, "resolution": "640x480",
     "pipeline": "native CUDA (cuda-oxide) physics + WebGPU render + pipelined readback (fixed-grid dispatch: no per-dispatch stream drain; was 5.6 fps with indirect dispatch)",
     "video": "cube_nexus_cuda.mp4",
     "video_url": "https://files.catbox.moe/woyrvy.mp4", "source": "measured 2026-07-09 (idle GPU)"},
    {"backend": "nexus_cuda_graph", "mode": "raster", "fps": 115.0, "fps_nocapture": 263.9, "resolution": "640x480",
     "pipeline": "native CUDA (cuda-oxide) physics, whole solver-step sequence captured into a CUDA graph (one cuGraphLaunch/frame) + WebGPU render + pipelined readback",
     "video": "cube_nexus_cuda_graph.mp4",
     "video_url": "https://files.catbox.moe/y0wh6k.mp4", "source": "measured 2026-07-09 (idle GPU)"},
    {"backend": "nexus_cpu", "mode": "raster", "fps": 37.5, "resolution": "640x480",
     "pipeline": "CPU physics + headless render + pipelined snap_rgb_async() readback (patched kiss3d; was 26.5 fps vsync-locked windowed)",
     "video": "cube_nexus_cpu.mp4", "video_url": "https://files.catbox.moe/r7srcn.mp4", "source": "measured 2026-07-08"},
    # mode "rt_shared": physics steps/s + shared Mitsuba/OptiX tracer.
    {"backend": "mujoco", "mode": "rt_shared", "physics_steps_s": 408_000, "fps": 35.0,
     "ms_per_frame": 29, "resolution": "480x360", "spp": 96, "video": "cube_rt_mujoco.mp4", "video_url": "https://files.catbox.moe/9klizd.mp4"},
    {"backend": "mjlab", "mode": "rt_shared", "physics_steps_s": 7_000, "fps": 38.5,
     "ms_per_frame": 26, "resolution": "480x360", "spp": 96, "video": "cube_rt_mjlab.mp4", "video_url": "https://files.catbox.moe/toi5m3.mp4"},
    {"backend": "genesis", "mode": "rt_shared", "physics_steps_s": 3_300, "fps": 37.7,
     "ms_per_frame": 27, "resolution": "480x360", "spp": 96, "video": "cube_rt_genesis.mp4", "video_url": "https://files.catbox.moe/3kyw9l.mp4"},
    {"backend": "isaac", "mode": "rt_shared", "physics_steps_s": 5_400, "fps": 37.2,
     "ms_per_frame": 27, "resolution": "480x360", "spp": 96},
    # mode "rt_native": the engine's own path tracer.
    {"backend": "isaac", "mode": "rt_native", "fps": 12, "ms_per_frame": 83,
     "resolution": "480x368", "spp": 64, "pipeline": "Omniverse RTX PathTracing",
     "video": "cube_rt_isaac_native.mp4", "video_url": "https://files.catbox.moe/zr5sma.mp4"},
    {"backend": "genesis", "mode": "rt_native", "physics_steps_s": 1_887, "fps": 14.3,
     "ms_per_frame": 70, "resolution": "480x368", "spp": 64,
     "pipeline": "LuisaRender (CUDA)", "video": "cube_rt_genesis_native.mp4", "video_url": "https://files.catbox.moe/qgub98.mp4"},
    # LeRobot bipedal platform (real STL-mesh asset, 12 DOF, 13 bodies).
    {"backend": "mujoco", "mode": "lerobot_raster", "fps": 536, "resolution": "640x480",
     "pipeline": "CPU physics (PD stance hold) + EGL render",
     "video": "lerobot_mujoco_real.mp4",
     "video_url": "https://files.catbox.moe/ery9of.mp4", "source": "measured 2026-07-08"},
    {"backend": "nexus", "mode": "lerobot_raster", "fps": 6.7, "fps_nocapture": 6.6, "resolution": "640x480",
     "pipeline": "GPU (WebGPU) multibody, passive (no torque API) + headless pipelined readback (physics-bound)",
     "video": "lerobot_nexus.mp4",
     "video_url": "https://files.catbox.moe/j8bzx1.mp4", "source": "measured 2026-07-08"},
    {"backend": "genesis", "mode": "lerobot_raster", "fps": 115.7, "fps_nocapture": 117.9, "resolution": "640x480",
     "pipeline": "GPU physics (PD stance hold, grid-searched hipy bias) + render (warmed up: taichi/render JIT excluded)",
     "video": "../lerobot_legs/lerobot_genesis.mp4",
     "video_url": "https://files.catbox.moe/3d5tbq.mp4", "source": "measured 2026-07-09"},
    {"backend": "nexus_cuda_graph", "mode": "lerobot_raster", "fps": 29.6, "fps_nocapture": 35.1, "resolution": "640x480",
     "pipeline": "native CUDA (cuda-oxide) multibody physics, CUDA-graph replay (one cuGraphLaunch/frame) + WebGPU render + pipelined readback; passive collapse (no torque hold in this demo)",
     "video": "../lerobot_legs/lerobot_nexus_cuda_graph.mp4",
     "video_url": "https://files.catbox.moe/8o4zn0.mp4", "source": "measured 2026-07-09 (idle GPU)"},
    {"backend": "nexus_cuda_graph", "mode": "lerobot_rt", "fps": 4.9, "resolution": "640x480", "spp": 32,
     "pipeline": "native CUDA (cuda-oxide) multibody physics via CUDA-graph replay + kiss3d wgpu path tracer (full-res, TLAS-only update)",
     "video": "../lerobot_legs/lerobot_nexus_rt_cuda_graph.mp4",
     "video_url": "https://files.catbox.moe/fv1o9a.mp4", "source": "measured 2026-07-09 (idle GPU)"},
    {"backend": "genesis", "mode": "lerobot_rt", "fps": 2.7, "ms_per_frame": 365,
     "resolution": "640x480", "spp": 32, "physics_steps_s": 758,
     "pipeline": "GPU physics (PD stance hold) + LuisaRender (CUDA) path tracer",
     "video": "../lerobot_legs/lerobot_genesis_rt.mp4",
     "video_url": "https://files.catbox.moe/6028t5.mp4", "source": "measured 2026-07-09 (idle GPU)"},
    {"backend": "isaac", "mode": "lerobot_rt", "fps": 70.1, "ms_per_frame": 14,
     "resolution": "640x480", "spp": 32, "physics_steps_s": 3351,
     "pipeline": "PhysX (PD hold, pinned base) + Omniverse RTX PathTracing",
     "video": "../lerobot_legs/lerobot_isaac.mp4",
     "video_url": "https://files.catbox.moe/rvxlc4.mp4", "source": "measured 2026-07-09 (idle GPU)"},
    {"backend": "nexus", "mode": "lerobot_rt", "fps": 3.2, "resolution": "640x480", "spp": 32,
     "pipeline": "kiss3d wgpu path tracer (HW ray query), headless, full-res (patched kiss3d: TLAS-only update for rigid motion + single 32-spp call; was 1.0 fps re-baking all meshes, traced at half res)",
     "video": "../lerobot_legs/lerobot_nexus_rt.mp4",
     "video_url": "https://files.catbox.moe/acuuo6.mp4", "source": "measured 2026-07-09 (idle GPU)"},
    {"backend": "nexus", "mode": "rt_native", "fps": 3.9, "ms_per_frame": 257,
     "resolution": "480x368", "spp": 64,
     "pipeline": "kiss3d 0.45 wgpu path tracer, headless + pipelined readback (tracing-bound; whole-loop timing; TLAS-only update patch changes nothing here — 2 meshes)",
     "video": "cube_rt_nexus_native.mp4", "video_url": "https://files.catbox.moe/44cl5o.mp4", "source": "measured 2026-07-09 (idle GPU)"},
]
for _r in KNOWN_RESULTS:
    _r.setdefault("source", "readme 2026-07-08")

# Per-frame wall-clock breakdown (ms) from the [segments] lines. Nexus rows
# re-swept 2026-07-09 with drain-corrected attribution (a state read after
# the async physics step / trace bills each segment its own GPU time); their
# fps matched the idle-GPU headlines. MuJoCo/Genesis swept on battery —
# shares only; absolute ms throttled vs the headline fps.
KNOWN_SEGMENTS: dict[tuple[str, str], dict[str, float]] = {
    ("mujoco", "raster"): {"physics": 0.05, "render": 1.90},
    ("genesis", "raster"): {"physics": 1.32, "render": 1.22},
    ("nexus", "raster"): {"physics": 11.70, "sync": 0.16, "render": 0.25, "readback": 5.10},
    ("nexus_cuda", "raster"): {"physics": 57.42, "sync": 0.02, "render": 0.25, "readback": 5.04},
    ("nexus_cuda_graph", "raster"): {"physics": 3.53, "sync": 0.01, "render": 0.18, "readback": 5.32},
    ("nexus", "rt_native"): {"physics": 9.38, "sync": 0.31, "render": 241.68, "readback": 3.53},
    ("mujoco", "lerobot_raster"): {"physics": 0.22, "render": 6.68},
    ("genesis", "lerobot_raster"): {"physics": 5.19, "render": 3.46},
    ("nexus", "lerobot_raster"): {"physics": 136.10, "sync": 3.79, "render": 4.40, "readback": 4.40},
    ("nexus_cuda_graph", "lerobot_raster"): {"physics": 25.89, "sync": 0.02, "render": 2.68, "readback": 4.78},
    ("nexus", "lerobot_rt"): {"physics": 99.91, "sync": 0.65, "render": 205.78, "readback": 3.81},
    ("nexus_cuda_graph", "lerobot_rt"): {"physics": 26.55, "sync": 0.06, "render": 0.84, "readback": 175.66},
}
for _r in KNOWN_RESULTS:
    _seg = KNOWN_SEGMENTS.get((_r["backend"], _r["mode"]))
    if _seg and "segments" not in _r:
        _r["segments"] = dict(_seg)
        _r["segments_source"] = "measured 2026-07-09"

# How to (re)measure each backend x mode: (interpreter, argv-tail, env, parser).
RE_FPS = re.compile(r"\[fps\] \w+: .* = ([\d.]+) gen-fps")
RE_PHYS = re.compile(r"\[phys\] \w+: (\d+) steps in [\d.]+s = ([\d,.]+) steps/s")
RE_RT_NATIVE = re.compile(
    r"\[rt-native\] \w+: physics ([\d,]+) steps/s \| .* ([\d.]+) fps \(([\d.]+) ms/frame"
)
RE_NEXUS_RT = re.compile(r"path trace ([\d.]+) fps \(([\d.]+) ms/frame")
RE_SEGMENTS = re.compile(r"\[segments\] [\w/-]+: (.+)")


def parse_segments(out: str, row: dict) -> None:
    """Optional per-frame timing breakdown: `[segments] tag: physics=1.23ms ...`."""
    m = RE_SEGMENTS.search(out)
    if m:
        row["segments"] = {
            k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)ms", m.group(1))
        }


def parse_fps(out: str, row: dict) -> None:
    row["fps"] = float(RE_FPS.search(out).group(1))


def parse_phys(out: str, row: dict) -> None:
    row["physics_steps_s"] = float(RE_PHYS.search(out).group(2).replace(",", ""))


def parse_rt_native(out: str, row: dict) -> None:
    m = RE_RT_NATIVE.search(out)
    row["physics_steps_s"] = float(m.group(1).replace(",", ""))
    row["fps"] = float(m.group(2))
    row["ms_per_frame"] = float(m.group(3))


def parse_nexus_rt(out: str, row: dict) -> None:
    m = RE_NEXUS_RT.search(out)
    row["fps"] = float(m.group(1))
    row["ms_per_frame"] = float(m.group(2))


def runners() -> dict[tuple[str, str], dict]:
    """(backend, mode) -> subprocess spec. Uses a temp outdir where needed."""
    egl = {"MUJOCO_GL": "egl"}
    specs: dict[tuple[str, str], dict] = {}
    for b in ("mujoco", "genesis", "mjlab", "nexus"):
        specs[(b, "raster")] = {
            "py": VENV_PY, "args": [str(HERE / f"{b}_cube.py")],
            "env": egl if b in ("mujoco", "mjlab") else {}, "parse": parse_fps,
        }
    for b in ("mujoco", "genesis", "mjlab", "isaac"):
        specs[(b, "rt_shared")] = {
            "py": ISAAC_PY if b == "isaac" else VENV_PY,
            "args": [str(HERE / "rt_record.py"), "--sim", b, "--out", "{tmp}/traj.npz"],
            "env": {}, "parse": parse_phys,
        }
    specs[("genesis", "rt_native")] = {
        "py": VENV_PY, "args": [str(HERE / "genesis_rt_native.py"), "--outdir", "{tmp}"],
        "env": {}, "parse": parse_rt_native, "result_txt": True,
    }
    specs[("isaac", "rt_native")] = {
        "py": ISAAC_PY, "args": [str(HERE / "isaac_rt_native.py"), "--outdir", "{tmp}"],
        "env": {}, "parse": parse_rt_native, "result_txt": True,
    }
    specs[("nexus", "rt_native")] = {
        "py": VENV_PY, "args": [str(HERE / "nexus_rt_native.py")],
        "env": {}, "parse": parse_nexus_rt,
    }
    return specs


def measure(backend: str, mode: str, spec: dict) -> dict:
    row: dict = {"backend": backend, "mode": mode, "source": MEASURED_TODAY}
    with tempfile.TemporaryDirectory() as tmp:
        args = [a.replace("{tmp}", tmp) for a in spec["args"]]
        env = os.environ | spec["env"]
        print(f"[bench] running {backend}/{mode}: {' '.join(args)}", flush=True)
        try:
            proc = subprocess.run(
                [spec["py"], *args], env=env, capture_output=True, text=True, timeout=1800
            )
            out = proc.stdout + proc.stderr
            if spec.get("result_txt"):  # Kit/Genesis write results to a file
                rt = Path(tmp) / "result.txt"
                if rt.exists():
                    out += rt.read_text()
            if proc.returncode != 0:
                raise RuntimeError(f"exit {proc.returncode}: {out[-400:]}")
            spec["parse"](out, row)
            parse_segments(out, row)
        except Exception as e:  # keep sweeping; record the failure
            row["error"] = str(e)[:400]
            print(f"[bench] {backend}/{mode} FAILED: {row['error']}", flush=True)
    return row


def machine_info() -> dict:
    info: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_cores": os.cpu_count(),
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        mem_kb = int(Path("/proc/meminfo").read_text().split()[1])
        info["ram_gb"] = round(mem_kb / 1024 / 1024)
    except OSError:
        pass
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().split(", ")
        info["gpu"], info["gpu_driver"], info["gpu_vram"] = gpu[0], gpu[1], gpu[2]
    except Exception:
        try:
            import torch

            info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            info["gpu"] = "none"
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["missing", "all", "none"], default="missing")
    args = ap.parse_args()

    rows = [dict(r) for r in KNOWN_RESULTS]
    have = {(r["backend"], r["mode"]) for r in rows}
    if args.run != "none":
        for key, spec in runners().items():
            if args.run == "all" or key not in have:
                row = measure(*key, spec)
                rows = [r for r in rows if (r["backend"], r["mode"]) != key] + [row]

    result = {
        "machine": machine_info(),
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }
    sweep = HERE / "nexus_rt_bench.json"
    if sweep.exists():
        result["nexus_spp_sweep"] = json.loads(sweep.read_text())
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[bench] wrote {OUT_JSON} ({len(rows)} rows)")

    subprocess.run([sys.executable, str(HERE / "make_site.py")], check=True)


if __name__ == "__main__":
    main()

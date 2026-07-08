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
     "pipeline": "CPU physics + EGL render", "video": "cube_mujoco.mp4"},
    {"backend": "mjlab", "mode": "raster", "fps": 565, "resolution": "640x480",
     "pipeline": "GPU (MuJoCo-Warp) + host EGL render", "video": "cube_mjlab.mp4"},
    {"backend": "genesis", "mode": "raster", "fps": 332, "resolution": "640x480",
     "pipeline": "GPU physics + render", "video": "cube_genesis.mp4"},
    {"backend": "pybullet", "mode": "raster", "fps": 29, "resolution": "640x480",
     "pipeline": "CPU + TinyRenderer (software raster)", "video": "cube_pybullet.mp4"},
    {"backend": "nexus", "mode": "raster", "fps": 72.7, "resolution": "640x480",
     "pipeline": "GPU (WebGPU) physics + render() readback (patched kiss3d)",
     "video": "cube_nexus.mp4", "source": "measured 2026-07-08"},
    {"backend": "nexus_cpu", "mode": "raster", "fps": 59.9, "resolution": "640x480",
     "pipeline": "Rapier CPU physics + render() readback (patched kiss3d)",
     "video": "cube_nexus_cpu.mp4", "source": "measured 2026-07-08"},
    # mode "rt_shared": physics steps/s + shared Mitsuba/OptiX tracer.
    {"backend": "mujoco", "mode": "rt_shared", "physics_steps_s": 408_000, "fps": 35.0,
     "ms_per_frame": 29, "resolution": "480x360", "spp": 96, "video": "cube_rt_mujoco.mp4"},
    {"backend": "pybullet", "mode": "rt_shared", "physics_steps_s": 317_000, "fps": 37.7,
     "ms_per_frame": 27, "resolution": "480x360", "spp": 96, "video": "cube_rt_pybullet.mp4"},
    {"backend": "mjlab", "mode": "rt_shared", "physics_steps_s": 7_000, "fps": 38.5,
     "ms_per_frame": 26, "resolution": "480x360", "spp": 96, "video": "cube_rt_mjlab.mp4"},
    {"backend": "genesis", "mode": "rt_shared", "physics_steps_s": 3_300, "fps": 37.7,
     "ms_per_frame": 27, "resolution": "480x360", "spp": 96, "video": "cube_rt_genesis.mp4"},
    {"backend": "isaac", "mode": "rt_shared", "physics_steps_s": 5_400, "fps": 37.2,
     "ms_per_frame": 27, "resolution": "480x360", "spp": 96},
    # mode "rt_native": the engine's own path tracer.
    {"backend": "isaac", "mode": "rt_native", "fps": 12, "ms_per_frame": 83,
     "resolution": "480x368", "spp": 64, "pipeline": "Omniverse RTX PathTracing",
     "video": "cube_rt_isaac_native.mp4"},
    {"backend": "genesis", "mode": "rt_native", "physics_steps_s": 1_887, "fps": 14.3,
     "ms_per_frame": 70, "resolution": "480x368", "spp": 64,
     "pipeline": "LuisaRender (CUDA)", "video": "cube_rt_genesis_native.mp4"},
    {"backend": "nexus", "mode": "rt_native", "fps": 4.3, "ms_per_frame": 234,
     "resolution": "480x368", "spp": 64,
     "pipeline": "kiss3d 0.45 wgpu path tracer (patched kiss3d; tracing-bound)",
     "video": "cube_rt_nexus_native.mp4", "source": "measured 2026-07-08"},
]
for _r in KNOWN_RESULTS:
    _r.setdefault("source", "readme 2026-07-08")

# How to (re)measure each backend x mode: (interpreter, argv-tail, env, parser).
RE_FPS = re.compile(r"\[fps\] \w+: .* = ([\d.]+) gen-fps")
RE_PHYS = re.compile(r"\[phys\] \w+: (\d+) steps in [\d.]+s = ([\d,.]+) steps/s")
RE_RT_NATIVE = re.compile(
    r"\[rt-native\] \w+: physics ([\d,]+) steps/s \| .* ([\d.]+) fps \(([\d.]+) ms/frame"
)
RE_NEXUS_RT = re.compile(r"path trace ([\d.]+) fps \(([\d.]+) ms/frame")


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
    for b in ("mujoco", "pybullet", "genesis", "mjlab", "nexus"):
        specs[(b, "raster")] = {
            "py": VENV_PY, "args": [str(HERE / f"{b}_cube.py")],
            "env": egl if b in ("mujoco", "mjlab") else {}, "parse": parse_fps,
        }
    for b in ("mujoco", "pybullet", "genesis", "mjlab", "isaac"):
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

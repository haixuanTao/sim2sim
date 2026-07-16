"""Profile RAM / VRAM / CPU / GPU while each backend renders the cube scene.

The fps rows say how fast each engine is; they say nothing about what it costs
to run. This samples the *whole process tree* of each backend's cube demo and
reports peak host RAM, peak VRAM, mean/peak CPU and mean/peak GPU utilisation.

Method notes, because each of these is easy to get wrong:

- RAM is USS (unique set size) summed over the process tree, not RSS. RSS
  double-counts shared pages across parent+children and counts shared libs the
  process did not really "use"; USS is what would be freed if the tree died,
  which is the number that matters when asking "can I run N of these".
- VRAM is read per-PID from nvidia-smi's compute-apps table, so another
  tenant's allocation is never attributed here. It is peak, not final: engines
  free buffers at exit.
- GPU utilisation is a whole-device sample -- nvidia-smi cannot attribute
  utilisation per process without MIG/accounting. The runner refuses to sample
  if any foreign compute process is on the device, since that would silently
  contaminate the number (this already bit the nexus sweep once).
- Sampling starts before the child and stops when it exits, so one-time costs
  (JIT, kernel/shader compile, USD stage load) ARE included in the peaks. That
  is deliberate: peak footprint is what sizes a machine. Steady-state means are
  reported alongside.

Run:  ~/rt_build/nyx-venv/bin/python examples/cube_drop/tools/resource_profile.py
      [--backend genesis_nyx|nexus|isaac|all] [--out ../resource_profile.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent.parent  # <repo>/examples/cube_drop/tools -> <repo>
RT = Path.home() / "rt_build"
INTERVAL = 0.2

# Each backend runs its own cube demo under its own interpreter -- they cannot
# share a venv (genesis needs git-main + torch cu128; isaac needs python 3.10).
BACKENDS = {
    "genesis_nyx": {
        "python": RT / "nyx-venv/bin/python",
        "script": REPO / "examples/cube_drop/genesis_nyx_native.py",
        "args": ["--no-capture", "--frames", "4000"],
        "label": "Genesis (Nyx)",
    },
    "nexus": {
        "python": RT / "bench-venv/bin/python",
        "script": REPO / "examples/cube_drop/nexus_rt_native.py",
        "args": ["--no-capture", "--frames", "4000"],
        "label": "Nexus",
    },
    "isaac": {
        "python": RT / "isaac-venv/bin/python",
        "script": REPO / "examples/cube_drop/isaac_rt_native.py",
        "args": ["--outdir", "/tmp/isaac_prof_frames"],
        "env": {"OMNI_KIT_ACCEPT_EULA": "YES"},
        "label": "Isaac Sim",
    },
}


def gpu_foreign_pids(exclude: set[int]) -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if p.strip().isdigit() and int(p) not in exclude]


def vram_mb_for(pids: set[int]) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    total = 0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in pids:
            total += int(parts[1])
    return total


def gpu_util() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else 0


def profile(key: str, spec: dict) -> dict:
    import psutil

    py = Path(spec["python"])
    if not py.exists():
        return {"backend": key, "label": spec["label"], "error": f"{py} missing"}

    foreign = gpu_foreign_pids(exclude=set())
    if foreign:
        return {"backend": key, "label": spec["label"],
                "error": f"GPU not idle (foreign compute pids {foreign}); refusing to sample"}

    import os
    env = {**os.environ, **spec.get("env", {})}
    proc = subprocess.Popen([str(py), str(spec["script"]), *spec["args"]],
                            cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    parent = psutil.Process(proc.pid)
    ram, vram, cpu, gpu = [], [], [], []
    t0 = time.perf_counter()
    parent.cpu_percent(None)  # prime the counter; first call always returns 0.0

    while proc.poll() is None:
        try:
            tree = [parent, *parent.children(recursive=True)]
            pids = {p.pid for p in tree}
            uss = 0
            c = 0.0
            for p in tree:
                try:
                    uss += p.memory_full_info().uss
                    c += p.cpu_percent(None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            ram.append(uss / 1e6)
            cpu.append(c)
            vram.append(vram_mb_for(pids))
            gpu.append(gpu_util())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(INTERVAL)

    out_txt = proc.stdout.read() if proc.stdout else ""
    wall = time.perf_counter() - t0

    def stat(xs, fn):
        return round(fn(xs), 1) if xs else None

    # Drop the first second of GPU/CPU samples for the "steady" means: process
    # spawn and import are not representative of the running engine.
    skip = int(1.0 / INTERVAL)
    return {
        "backend": key, "label": spec["label"], "exit": proc.returncode,
        "wall_s": round(wall, 1), "samples": len(ram),
        "ram_peak_mb": stat(ram, max), "ram_mean_mb": stat(ram, lambda x: sum(x) / len(x)),
        "vram_peak_mb": stat(vram, max), "vram_mean_mb": stat(vram, lambda x: sum(x) / len(x)),
        "cpu_peak_pct": stat(cpu, max), "cpu_mean_pct": stat(cpu[skip:] or cpu, lambda x: sum(x) / len(x)),
        "gpu_peak_pct": stat(gpu, max), "gpu_mean_pct": stat(gpu[skip:] or gpu, lambda x: sum(x) / len(x)),
        "fps_line": next((l for l in out_txt.splitlines()
                          if "fps" in l.lower() and l.startswith("[")), ""),
        # Process start -> first rendered frame in hand. Complements the `boot`
        # rows, which stop at the first physics step and never touch the render
        # pipeline -- so they miss the RTX shader cache and the first path-trace
        # accumulation entirely.
        "first_frame_s": next((float(l.split(":")[-1].split("s")[0].strip())
                               for l in out_txt.splitlines() if "[first-frame]" in l), None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="all", choices=[*BACKENDS, "all"])
    ap.add_argument("--out", default=str(HERE.parent / "resource_profile.json"))
    args = ap.parse_args()

    if not shutil.which("nvidia-smi"):
        sys.exit("nvidia-smi not found")

    keys = list(BACKENDS) if args.backend == "all" else [args.backend]
    rows = []
    for k in keys:
        print(f"[profile] {k} ...", flush=True)
        r = profile(k, BACKENDS[k])
        rows.append(r)
        if r.get("error"):
            print(f"  SKIP {r['error']}", flush=True)
        else:
            print(f"  RAM peak {r['ram_peak_mb']} MB | VRAM peak {r['vram_peak_mb']} MB | "
                  f"CPU mean {r['cpu_mean_pct']}% | GPU mean {r['gpu_mean_pct']}% "
                  f"| {r['wall_s']}s exit={r['exit']}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"[profile] wrote {args.out}")


if __name__ == "__main__":
    main()

"""Re-measure the nexus rows of the desktop-5090 panel and regenerate the site."""
import json
import re
import subprocess
import sys
import datetime
from pathlib import Path

REPO = Path.home() / "sim2sim"
PY = str(REPO / ".venv/bin/python")
JSON_PATH = REPO / "examples/cube_drop/benchmark_results.json"

RE_FPS = re.compile(r"\[fps\] [\w-]+: .* = ([\d.]+) gen-fps")
RE_NOCAP = re.compile(r"\[fps-nocapture\] [\w-]+: .* = ([\d.]+) gen-fps")
RE_RT = re.compile(r"path trace ([\d.]+) fps \((\d+) ms/frame")
RE_SEG = re.compile(r"\[segments\] [\w/-]+: (.+)")

CUBE = "examples/cube_drop"
LERO = "examples/lerobot_legs"

# (backend, mode) -> script, extra args, needs cuda warmup, has no-capture variant
TARGETS = [
    (("nexus", "raster"), f"{CUBE}/nexus_cube.py", [], False, True),
    (("nexus_cuda", "raster"), f"{CUBE}/nexus_cube.py", ["--cuda"], True, True),
    (("nexus_cuda_graph", "raster"), f"{CUBE}/nexus_cube.py", ["--cuda-graph"], True, True),
    (("nexus", "rt_native"), f"{CUBE}/nexus_rt_native.py", [], False, False),
    (("nexus", "lerobot_raster"), f"{LERO}/nexus_render.py", [], False, True),
    (("nexus_cuda_graph", "lerobot_raster"), f"{LERO}/nexus_render.py", ["--cuda-graph"], True, True),
    (("nexus", "lerobot_rt"), f"{LERO}/nexus_render.py", ["--rt"], False, False),
    (("nexus_cuda_graph", "lerobot_rt"), f"{LERO}/nexus_render.py", ["--rt", "--cuda-graph"], True, False),
]


def run(script, args):
    cmd = [PY, str(REPO / script), *args]
    print(f"[remeasure] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {out[-600:]}")
    return out


def main():
    # Co-tenant gate: any other GPU process invalidates every number
    # (a background training run once halved the whole sweep uniformly).
    procs = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    if procs:
        raise SystemExit(f"GPU not idle - refusing to measure:\n{procs}")

    data = json.loads(JSON_PATH.read_text())
    panel = next(m for m in data["machines"] if m["id"] == "desktop-5090")
    rows = {(r["backend"], r["mode"]): r for r in panel["rows"]}
    today = datetime.date.today().isoformat()

    for key, script, args, warmup, nocap in TARGETS:
        row = rows.get(key)
        if row is None:
            print(f"[remeasure] SKIP {key}: no existing row", flush=True)
            continue
        try:
            if warmup:
                run(script, args)  # cold cubin/driver-cache run, discard
            out = run(script, args)
            m = RE_FPS.search(out) or RE_RT.search(out)
            if m is None:
                raise RuntimeError(f"no fps line in output: {out[-400:]}")
            row["fps"] = float(m.group(1))
            if RE_RT.search(out):
                row["ms_per_frame"] = float(RE_RT.search(out).group(2))
            seg = RE_SEG.search(out)
            if seg:
                row["segments"] = {k: float(v) for k, v in
                                   re.findall(r"(\w+)=([\d.]+)ms", seg.group(1))}
                row["segments_source"] = f"measured {today}"
            if nocap:
                out2 = run(script, [*args, "--no-capture"])
                m2 = RE_NOCAP.search(out2)
                if m2:
                    row["fps_nocapture"] = float(m2.group(1))
            row["source"] = f"measured {today} (idle GPU, headless RTX 5090)"
            print(f"[remeasure] {key}: fps={row['fps']}"
                  f"{' nocap=' + str(row.get('fps_nocapture')) if nocap else ''}", flush=True)
        except Exception as e:
            print(f"[remeasure] {key} FAILED, keeping old row: {str(e)[:400]}", flush=True)

    panel["generated"] = datetime.datetime.now().isoformat(timespec="seconds")
    JSON_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[remeasure] wrote {JSON_PATH}", flush=True)
    subprocess.run([sys.executable, str(REPO / "examples/cube_drop/make_site.py")],
                   check=True, cwd=REPO)


if __name__ == "__main__":
    main()

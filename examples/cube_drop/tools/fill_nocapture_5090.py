"""Fill fps_nocapture for desktop-5090 rows that lack it, then regenerate the site."""
import json
import os
import re
import subprocess
import sys
import datetime
from pathlib import Path

REPO = Path.home() / "sim2sim"
PY = str(REPO / ".venv/bin/python")
JSON_PATH = REPO / "examples/cube_drop/benchmark_results.json"
RE_NOCAP = re.compile(r"\[fps-nocapture\] [\w/-]+: .* = ([\d.]+) gen-fps")

CUBE = "examples/cube_drop"
LERO = "examples/lerobot_legs"

TARGETS = [
    (("mujoco", "raster"), f"{CUBE}/mujoco_cube.py", []),
    (("mjlab", "raster"), f"{CUBE}/mjlab_cube.py", []),
    (("genesis", "raster"), f"{CUBE}/genesis_cube.py", []),
    (("mujoco", "lerobot_raster"), f"{LERO}/mujoco_render.py", []),
    (("genesis", "lerobot_raster"), f"{LERO}/genesis_render.py", []),
    (("nexus", "rt_native"), f"{CUBE}/nexus_rt_native.py", []),
    (("nexus", "lerobot_rt"), f"{LERO}/nexus_render.py", ["--rt"]),
    (("nexus_cuda_graph", "lerobot_rt"), f"{LERO}/nexus_render.py", ["--rt", "--cuda-graph"]),
]


def main():
    data = json.loads(JSON_PATH.read_text())
    panel = next(m for m in data["machines"] if m["id"] == "desktop-5090")
    rows = {(r["backend"], r["mode"]): r for r in panel["rows"]}
    today = datetime.date.today().isoformat()

    for key, script, args in TARGETS:
        row = rows.get(key)
        if row is None or row.get("fps_nocapture"):
            print(f"[fill] SKIP {key}", flush=True)
            continue
        cmd = [PY, str(REPO / script), *args, "--no-capture"]
        print(f"[fill] {' '.join(cmd)}", flush=True)
        env = os.environ | ({"MUJOCO_GL": "egl"} if key[0] in ("mujoco", "mjlab") else {})
        try:
            p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
            out = p.stdout + p.stderr
            if p.returncode != 0:
                raise RuntimeError(f"exit {p.returncode}: {out[-600:]}")
            m = RE_NOCAP.search(out)
            if m is None:
                raise RuntimeError(f"no [fps-nocapture] line: {out[-400:]}")
            row["fps_nocapture"] = float(m.group(1))
            row["source"] = (row.get("source", "").rstrip() +
                             f"; no-readback measured {today}")
            print(f"[fill] {key}: nocap={row['fps_nocapture']}", flush=True)
        except Exception as e:
            print(f"[fill] {key} FAILED: {str(e)[:400]}", flush=True)

    panel["generated"] = datetime.datetime.now().isoformat(timespec="seconds")
    JSON_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[fill] wrote {JSON_PATH}", flush=True)
    subprocess.run([sys.executable, str(REPO / "examples/cube_drop/make_site.py")],
                   check=True, cwd=REPO)


if __name__ == "__main__":
    main()

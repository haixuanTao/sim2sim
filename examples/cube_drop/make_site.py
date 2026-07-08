"""Generate the static benchmark webpage from benchmark_results.json.

Writes ``site/index.html`` with all data inlined (no fetch — the page works
from ``file://``) and videos referenced relatively (``../cube_*.mp4``), so the
directory can be committed as-is or served by GitHub Pages.

Run:  python examples/cube_drop/make_site.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = json.loads((HERE / "benchmark_results.json").read_text())
OUT = HERE / "site/index.html"

# Modes shown on the page; rt_shared rows stay in benchmark_results.json but
# are not displayed (external tracer — not an engine rendering its own scene).
MODE_LABEL = {
    "raster": "Rasterized (native renderer)",
    "rt_native": "Ray traced — engine's own path tracer",
    "lerobot_raster": "LeRobot bipedal platform — rasterized",
    "lerobot_rt": "LeRobot bipedal platform — path traced",
}
BACKEND_LABEL = {
    "mujoco": "MuJoCo", "pybullet": "PyBullet", "mjlab": "mjlab (MuJoCo-Warp)",
    "genesis": "Genesis", "nexus": "Nexus (Rapier-on-GPU)",
    "nexus_cpu": "Nexus (Rapier CPU)", "isaac": "Isaac Sim",
}


def fmt(v, suffix="") -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and v >= 100:
        v = round(v)
    return f"{v:,}{suffix}" if isinstance(v, int) else f"{v}{suffix}"


def table_rows() -> str:
    rows = sorted(
        (r for r in DATA["rows"] if r["mode"] in MODE_LABEL),
        key=lambda r: (list(MODE_LABEL).index(r["mode"]), -(r.get("fps") or 0)),
    )
    gpu = DATA["machine"].get("gpu", "none")
    out = []
    for r in rows:
        if "error" in r:
            cells = [BACKEND_LABEL.get(r["backend"], r["backend"]), MODE_LABEL[r["mode"]],
                     f"<span class='err'>failed: {html.escape(r['error'][:80])}</span>",
                     "", "", "", "", gpu, r.get("source", "")]
        else:
            cells = [
                BACKEND_LABEL.get(r["backend"], r["backend"]),
                MODE_LABEL[r["mode"]],
                fmt(r.get("fps")),
                fmt(r.get("ms_per_frame")),
                fmt(r.get("physics_steps_s")),
                r.get("resolution", "—"),
                fmt(r.get("spp")),
                gpu,
                r.get("source", ""),
            ]
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "\n".join(out)


def video_cards() -> str:
    cards = []
    for mode in MODE_LABEL:
        group = [r for r in DATA["rows"] if r["mode"] == mode
                 and (r.get("video_url") or (r.get("video") and (HERE / r["video"]).exists()))]
        if not group:
            continue
        cards.append(f"<h3>{MODE_LABEL[mode]}</h3><div class='gallery'>")
        for r in sorted(group, key=lambda r: -(r.get("fps") or 0)):
            fps = r.get("fps")
            badge = f"<span class='badge'>{fmt(fps)} fps</span>" if fps else ""
            spp = f" @ {r['spp']} spp" if r.get("spp") else ""
            cards.append(
                f"""<figure>
  <video src="{r.get('video_url') or '../' + r['video']}" controls loop muted playsinline></video>
  <figcaption><strong>{BACKEND_LABEL.get(r['backend'], r['backend'])}</strong> {badge}
  <span class="sub">{r.get('resolution', '')}{spp}</span></figcaption>
</figure>"""
            )
        cards.append("</div>")
    return "\n".join(cards)


def fps_charts() -> str:
    """One horizontal bar chart per mode (small multiples — scales differ too
    much for one axis). Single series (fps), direct-labeled, linear scale."""
    sections = []
    for mode, label in MODE_LABEL.items():
        rows = sorted(
            (r for r in DATA["rows"] if r["mode"] == mode and r.get("fps")),
            key=lambda r: -r["fps"],
        )
        if not rows:
            continue
        vmax = max(r["fps"] for r in rows)
        bars = "\n".join(
            f"""<div class="bar-row" tabindex="0" title="{BACKEND_LABEL.get(r['backend'], r['backend'])}: {fmt(r['fps'])} fps ({r.get('resolution', '')}{' @ ' + str(r['spp']) + ' spp' if r.get('spp') else ''})">
  <span class="bar-label">{BACKEND_LABEL.get(r['backend'], r['backend'])}<span class="sub"> {r.get('resolution', '')}</span></span>
  <span class="bar-track"><span class="bar" style="width:{max(100 * r['fps'] / vmax, 0.4):.1f}%"></span></span>
  <span class="bar-val">{fmt(r['fps'])} fps</span>
</div>"""
            for r in rows
        )
        sections.append(
            f"""<h3>{label}</h3>
<div class="chart" role="img" aria-label="Bar chart of render fps per backend, {label}">
{bars}
</div>"""
        )
    return "\n".join(sections)


def main() -> None:
    m = DATA["machine"]
    tiles = "".join(
        f"<div class='tile'><span class='tile-label'>{label}</span><span class='tile-val'>{html.escape(str(val))}</span></div>"
        for label, val in [
            ("GPU", m.get("gpu", "none")),
            ("Driver", m.get("gpu_driver", "—")),
            ("VRAM", m.get("gpu_vram", "—")),
            ("CPU", m.get("cpu", "—")),
            ("RAM", f"{m.get('ram_gb', '—')} GB"),
            ("OS", m.get("os", "—")),
        ]
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sim2sim cube-drop benchmark</title>
<style>
:root {{
  --surface: #fcfcfb; --surface-2: #f1f1ee; --ink: #0b0b0b; --ink-2: #52514e;
  --line: #dddcd6; --accent: #2a78d6; --err: #b3261e;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --surface: #1a1a19; --surface-2: #242422; --ink: #ffffff; --ink-2: #c3c2b7;
           --line: #3a3936; --accent: #3987e5; --err: #e66767; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0 auto; max-width: 1080px; padding: 2rem 1.25rem 4rem;
       font: 15px/1.55 system-ui, sans-serif; background: var(--surface); color: var(--ink); }}
h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; }}
h2 {{ font-size: 1.15rem; margin: 2.5rem 0 .5rem; }}
h3 {{ font-size: 1rem; margin: 1.5rem 0 .5rem; color: var(--ink-2); }}
.sub {{ color: var(--ink-2); font-size: .85rem; }}
.tiles {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0; }}
.tile {{ background: var(--surface-2); border: 1px solid var(--line); border-radius: 8px;
        padding: .5rem .8rem; display: flex; flex-direction: column; min-width: 7rem; }}
.tile-label {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .05em; color: var(--ink-2); }}
.tile-val {{ font-weight: 600; font-size: .9rem; }}
.tablewrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
th, td {{ text-align: left; padding: .45rem .7rem; border-top: 1px solid var(--line); white-space: nowrap; }}
thead th {{ border-top: 0; background: var(--surface-2); font-size: .75rem;
            text-transform: uppercase; letter-spacing: .04em; color: var(--ink-2); }}
tbody tr:hover {{ background: var(--surface-2); }}
td:nth-child(n+3):nth-child(-n+7) {{ font-variant-numeric: tabular-nums; }}
.err {{ color: var(--err); }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }}
figure {{ margin: 0; }}
video {{ width: 100%; border-radius: 8px; border: 1px solid var(--line); background: #000; }}
figcaption {{ margin-top: .3rem; font-size: .85rem; }}
figcaption .sub {{ display: block; }}
.badge {{ background: var(--accent); color: #fff; border-radius: 999px;
         padding: .05rem .55rem; font-size: .72rem; font-weight: 600; vertical-align: middle; }}
.chart {{ display: flex; flex-direction: column; gap: 6px; max-width: 680px; margin: .75rem 0 1.25rem; }}
.bar-row {{ display: grid; grid-template-columns: 14.5rem 1fr 6rem; align-items: center; gap: .6rem; }}
.bar-row:hover .bar, .bar-row:focus .bar {{ filter: brightness(1.15); }}
.bar-label {{ font-size: .8rem; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bar-track {{ background: var(--surface-2); border-radius: 4px; height: 14px; overflow: hidden; }}
.bar {{ display: block; height: 100%; background: var(--accent); border-radius: 0 4px 4px 0; }}
.bar-val {{ font-size: .8rem; font-variant-numeric: tabular-nums; }}
details {{ margin: 1rem 0; }}
summary {{ cursor: pointer; color: var(--ink-2); font-size: .9rem; }}
footer {{ margin-top: 3rem; font-size: .8rem; color: var(--ink-2); }}
code {{ background: var(--surface-2); padding: .05rem .3rem; border-radius: 4px; }}
</style>
</head>
<body>
<h1>sim2sim cube-drop benchmark <span class="sub" style="font-size:1rem; font-weight:400;">— single scene, 1 env</span></h1>
<p class="sub"><strong>Single-environment cube drop</strong>: one tilted cube dropped onto a plane in
<strong>1 env</strong>, generated by six physics engines, rasterized and path-traced natively. This measures
per-frame render/readback overhead — not the massively-parallel multi-env workloads the GPU engines
(Genesis, mjlab, Isaac, Nexus) are designed for. Generated {DATA['generated']} by
<code>examples/cube_drop/benchmark.py</code>.</p>

<div class="tiles">{tiles}</div>

<h2>Results</h2>
<p class="sub">Raster fps = full per-frame loop (physics + render + readback). Native-RT
rows are each engine path-tracing its own scene. Not apples-to-apples across resolutions.</p>
<p class="sub">Nexus numbers use a patched kiss3d: its <code>read_pixels</code> converted pixels by
indexing uncached mapped GPU memory per byte (~99 ms/frame at 640×480); converting from a cached
row copy + reusing the staging buffer cut readback to ~5 ms (18×), lifting Nexus raster capture
from 2.7 to ~40 gen-fps (at real-time physics: 2 solver steps of 1/60 s per 30 fps frame,
calibrated with the new <code>body_pose()</code> getter against analytic free fall) — see
<a href="https://github.com/dimforge/kiss3d/pull/397">kiss3d #397 (readback fix)</a>,
<a href="https://github.com/dimforge/nexus/pull/7">nexus #7 (frame export)</a> and
<a href="https://github.com/dimforge/nexus/pull/8">nexus #8 (Python ray tracing)</a>.</p>
{fps_charts()}

<details>
<summary>Full data table</summary>
<div class="tablewrap">
<table>
<thead><tr><th>Backend</th><th>Mode</th><th>Render fps</th><th>ms/frame</th>
<th>Physics steps/s</th><th>Resolution</th><th>spp</th><th>GPU</th><th>Source</th></tr></thead>
<tbody>
{table_rows()}
</tbody>
</table>
</div>
</details>

<h2>Videos</h2>
{video_cards()}

<footer>sim2sim-locomotion &middot; cube-drop smoke benchmark &middot; single-body scene: this measures
render/readback overhead, not physics scalability.</footer>
</body>
</html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"[site] wrote {OUT}")


if __name__ == "__main__":
    main()

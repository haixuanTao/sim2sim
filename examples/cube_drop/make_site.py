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
_RAW = json.loads((HERE / "benchmark_results.json").read_text())
# Accept either a single-machine file ({machine, rows, generated}) or a
# multi-machine file ({machines: [{id, label, machine, rows, generated}, ...]}).
# The page renders one selectable panel per machine.
if "machines" in _RAW:
    MACHINES = _RAW["machines"]
else:
    MACHINES = [{
        "id": "this", "label": _RAW.get("machine", {}).get("gpu", "This machine"),
        "machine": _RAW.get("machine", {}), "rows": _RAW.get("rows", []),
        "generated": _RAW.get("generated", ""),
    }]
# `DATA` is the machine currently being rendered; main() reassigns it per panel
# so the existing render helpers (which read this global) work unchanged.
DATA = MACHINES[0]
OUT = HERE / "site/index.html"

# Modes shown on the page; rt_shared rows stay in benchmark_results.json but
# are not displayed (external tracer — not an engine rendering its own scene).
MODE_LABEL = {
    "raster": "Rasterized (native renderer)",
    "rt_shared": "Ray traced — shared tracer (Mitsuba)",
    "rt_native": "Ray traced — engine's own path tracer",
    "lerobot_raster": "Rasterized (native renderer)",
    "lerobot_rt": "Ray traced — engine's own path tracer",
    "lerobot_batch": "Physics only — 2,048 parallel envs",
    "boot": "Time to first step — single env, warm JIT caches",
}
MODE_SCENE = {
    "raster": "Cube drop", "rt_shared": "Cube drop", "rt_native": "Cube drop",
    "lerobot_raster": "LeRobot legs", "lerobot_rt": "LeRobot legs",
    "lerobot_batch": "LeRobot legs × 2,048",
    "boot": "Startup",
}
# Scene sections shown on the page, in order.
SCENES = {
    "Scene 1 — Cube drop (single rigid body)": ["raster", "rt_native"],
    "Scene 2 — LeRobot bipedal platform (STL meshes, 12 DOF, PD stance hold)":
        ["lerobot_raster", "lerobot_rt"],
    "Scene 3 — LeRobot × 2,048 parallel envs (physics throughput, no rendering)":
        ["lerobot_batch"],
    "Startup — time to first physics step (LeRobot scene, single env)": ["boot"],
}
BACKEND_LABEL = {
    "mujoco": "MuJoCo", "mjlab": "mjlab (MuJoCo-Warp)",
    "genesis": "Genesis", "nexus": "Nexus",
    "nexus_cpu": "Nexus (CPU)", "nexus_cuda": "Nexus (cuda-oxide)", "nexus_cuda_graph": "Nexus (cuda-oxide + CUDA graph)", "isaac": "Isaac Sim",
}


# Per-frame time segments (fixed order + fixed colors; see site CSS vars).
SEGMENTS = ["physics", "sync", "render", "readback"]
# The page presents everything on the no-frame-readback basis, so the
# readback segment is excluded from the displayed breakdowns.
DISPLAY_SEGMENTS = ["physics", "sync", "render"]
SEG_LABEL = {
    "physics": "Physics", "sync": "Sync (GPU→host)",
    "render": "Render", "readback": "Readback",
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
        key=lambda r: (list(MODE_LABEL).index(r["mode"]),
                       -(r.get("fps_nocapture") or r.get("fps") or 0)),
    )
    gpu = DATA["machine"].get("gpu", "none")
    out = []
    for r in rows:
        segd = dict(_seg_breakdown(r))
        if r.get("fps_nocapture"):
            segd.pop("readback", None)
        raw_total = sum(segd.values())
        scale = _chart_ms(r) / raw_total if segd and r.get("fps") else 1.0
        breakdown = (" / ".join(fmt_ms(segd[k] * scale) if k in segd else "—"
                                for k in DISPLAY_SEGMENTS) if segd else "—")
        if "error" in r:
            cells = [BACKEND_LABEL.get(r["backend"], r["backend"]),
                     f"{MODE_SCENE[r['mode']]} — {MODE_LABEL[r['mode']]}",
                     f"<span class='err'>failed: {html.escape(r['error'][:80])}</span>",
                     "", "", "", "", "", gpu, r.get("source", "")]
        else:
            cells = [
                BACKEND_LABEL.get(r["backend"], r["backend"]),
                f"{MODE_SCENE[r['mode']]} — {MODE_LABEL[r['mode']]}",
                fmt(r.get("fps_nocapture")) if r.get("fps_nocapture")
                else (f"{fmt(r.get('fps'))} {'†' if r.get('readback_negligible') else '*'}"
                      if r.get("fps") else "—"),
                (f"{r['boot_s']:.2f} s" if r.get("boot_s")
                 else fmt_ms(_chart_ms(r)) if r.get("fps") else "—"),
                breakdown,
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
    for scene, modes in SCENES.items():
        cards.append(f"<h2 class='scene'>{scene}</h2>")
        cards.extend(_video_cards_for(mode) for mode in modes)
    return "\n".join(cards)


def _video_cards_for(mode: str) -> str:
    cards = []
    group = [r for r in DATA["rows"] if r["mode"] == mode
             and (r.get("video_url") or (r.get("video") and (HERE / r["video"]).exists()))]
    if group:
        cards.append(f"<h3>{MODE_LABEL[mode]}</h3><div class='gallery'>")
        for r in sorted(group, key=lambda r: -(r.get("fps_nocapture") or r.get("fps") or 0)):
            fps = r.get("fps_nocapture") or r.get("fps")
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
    for scene, modes in SCENES.items():
        sections.append(f"<h2 class='scene'>{scene}</h2>")
        sections.extend(
            (_batch_chart_for if mode == "lerobot_batch"
             else _boot_chart_for if mode == "boot"
             else _fps_chart_for)(mode, MODE_LABEL[mode])
            for mode in modes)
    return "\n".join(sections)


def _boot_chart_for(mode: str, label: str) -> str:
    """Seconds until the first physics step — lower is better, fastest first."""
    rows = sorted(
        (r for r in DATA["rows"] if r["mode"] == mode and r.get("boot_s")),
        key=lambda r: r["boot_s"],
    )
    err_rows = [r for r in DATA["rows"] if r["mode"] == mode and r.get("error")]
    if not rows and not err_rows:
        return ""
    vmax = max((r["boot_s"] for r in rows), default=1)
    bars = []
    for r in rows:
        v = r["boot_s"]
        width = max(100 * v / vmax, 0.4)
        imp = r.get("boot_imports_s", 0.0)
        ini = r.get("boot_init_s", 0.0)
        rest = max(v - imp - ini, 0.0)
        tip = (f"imports {imp:.2f} s, engine init {ini:.2f} s, "
               f"scene + first step {rest:.2f} s — "
               + html.escape(r.get("pipeline", "")))
        cold = r.get("boot_cold_s")
        extra = (f'<br><span class="sub">cold caches: {cold:.1f} s</span>' if cold else "")
        fill = "".join(
            f'<span class="bar {cls}" style="width:{width * part / v:.2f}%"></span>'
            for cls, part in (("seg-sync", imp), ("seg-physics", ini), ("seg-render", rest))
            if part > 0)
        bars.append(
            f"""<div class="bar-row" tabindex="0" title="{BACKEND_LABEL.get(r['backend'], r['backend'])}: {v:.2f} s to first step — {tip}">
  <span class="bar-label">{BACKEND_LABEL.get(r['backend'], r['backend'])}</span>
  <span class="bar-track">{fill}</span>
  <span class="bar-val">{v:.2f} s{extra}</span>
</div>"""
        )
    for r in err_rows:
        bars.append(
            f"""<div class="bar-row">
  <span class="bar-label">{BACKEND_LABEL.get(r['backend'], r['backend'])}</span>
  <span class="bar-track"></span>
  <span class="bar-val err">{html.escape(r['error'][:110])}</span>
</div>"""
        )
    return f"""<h3>{label}</h3>
<p class="sub">Wall time from process start to the first completed physics step, single env,
<strong>warm JIT caches</strong> — the cost of every script rerun / config change. Bar segments:
<span style="color:var(--seg-sync)">■</span> imports ·
<span style="color:var(--seg-physics)">■</span> engine init ·
<span style="color:var(--seg-render)">■</span> scene build + first step. Where a
<strong>cold caches</strong> figure is shown, that's the same run with the engine's kernel cache
emptied — the first-ever-run / post-upgrade cost. Hover a bar for exact segments.</p>
<div class="chart" role="img" aria-label="Bar chart of seconds until the first physics step per backend (lower is better)">
{chr(10).join(bars)}
</div>"""


def _batch_chart_for(mode: str, label: str) -> str:
    """Env-steps/s bars — higher is better (unlike the ms/frame charts)."""
    rows = sorted(
        (r for r in DATA["rows"] if r["mode"] == mode and r.get("physics_steps_s")),
        key=lambda r: -r["physics_steps_s"],
    )
    err_rows = [r for r in DATA["rows"] if r["mode"] == mode and r.get("error")]
    if not rows and not err_rows:
        return ""
    vmax = max((r["physics_steps_s"] for r in rows), default=1)
    bars = []
    for r in rows:
        v = r["physics_steps_s"]
        width = max(100 * v / vmax, 0.4)
        tip = html.escape(r.get("pipeline", ""))
        bars.append(
            f"""<div class="bar-row" tabindex="0" title="{BACKEND_LABEL.get(r['backend'], r['backend'])}: {fmt(round(v))} env-steps/s — {tip}">
  <span class="bar-label">{BACKEND_LABEL.get(r['backend'], r['backend'])}<span class="sub"> {r.get('dt_note', '')}</span></span>
  <span class="bar-track"><span class="bar seg-physics" style="width:{width:.1f}%"></span></span>
  <span class="bar-val">{fmt(round(v))} <span class="sub">env-steps/s</span></span>
</div>"""
        )
    for r in err_rows:
        bars.append(
            f"""<div class="bar-row">
  <span class="bar-label">{BACKEND_LABEL.get(r['backend'], r['backend'])}</span>
  <span class="bar-track"></span>
  <span class="bar-val err">{html.escape(r['error'][:90])}</span>
</div>"""
        )
    return f"""<h3>{label}</h3>
<p class="sub">Bars show <strong>env-steps/s — higher is better</strong> (physics only, no
rendering; steady-state window with one device sync at the end. Per-engine dt differs — an
env-step is one engine step, see the table).</p>
<div class="chart" role="img" aria-label="Bar chart of environment steps per second per backend (higher is better), {label}">
{chr(10).join(bars)}
</div>"""


def _seg_breakdown(r: dict) -> list[tuple[str, float]]:
    segs = r.get("segments") or {}
    return [(k, segs[k]) for k in SEGMENTS if segs.get(k)]


def fmt_ms(ms: float) -> str:
    return f"{ms:.2f}" if ms < 10 else f"{ms:,.0f}"


def _bar_fill(r: dict, width_pct: float, total_ms: float,
              drop_readback: bool = False) -> tuple[str, str]:
    """(inner bar HTML, tooltip suffix). Stacked by segment share when known.

    Segment ms are rescaled onto the row's headline frame time (shares were
    measured in a separate sweep), so the stack always sums to total_ms. With
    drop_readback the readback segment is excluded (GPU-only loops don't run
    it) and the rest is rescaled onto total_ms.
    """
    segs = _seg_breakdown(r)
    if drop_readback:
        segs = [(k, ms) for k, ms in segs if k != "readback"]
    if not segs:
        return (f'<span class="bar bar-flat" style="width:{width_pct:.1f}%"></span>',
                " — no per-segment breakdown")
    raw_total = sum(ms for _, ms in segs)
    spans = "".join(
        f'<span class="bar seg-{k}" style="width:{width_pct * ms / raw_total:.2f}%"></span>'
        for k, ms in segs
    )
    tip = " — " + ", ".join(
        f"{SEG_LABEL[k].lower()} {fmt_ms(total_ms * ms / raw_total)} ms"
        f" ({100 * ms / raw_total:.0f}%)"
        for k, ms in segs
    )
    return spans, tip


def _row_ms(r: dict) -> float:
    return r.get("ms_per_frame") or 1000.0 / r["fps"]


def _chart_ms(r: dict) -> float:
    """Frame time charted: the GPU-only (no frame readback) loop when measured,
    the full capture loop otherwise."""
    if r.get("fps_nocapture"):
        return 1000.0 / r["fps_nocapture"]
    return _row_ms(r)


def _fps_chart_for(mode: str, label: str) -> str:
    rows = sorted(
        (r for r in DATA["rows"] if r["mode"] == mode and r.get("fps")),
        key=_chart_ms,
    )
    if not rows:
        return ""
    vmax = max(_chart_ms(r) for r in rows)
    bars = []
    for r in rows:
        nocap = r.get("fps_nocapture")
        ms = _chart_ms(r)
        fps = nocap or r["fps"]
        width = max(100 * ms / vmax, 0.4)
        fill, tip = _bar_fill(r, width, ms, drop_readback=bool(nocap))
        sub = f"{r.get('resolution', '')}{' @ ' + str(r['spp']) + ' spp' if r.get('spp') else ''}"
        if nocap:
            extra = ""
        elif r.get("readback_negligible"):
            extra = '<br><span class="sub">incl. readback (≤1% of frame time)</span>'
        else:
            extra = '<br><span class="sub">incl. readback (not re-measured)</span>'
        bars.append(
            f"""<div class="bar-row" tabindex="0" title="{BACKEND_LABEL.get(r['backend'], r['backend'])}: {fmt_ms(ms)} ms/frame = {fmt(fps)} fps ({sub}){tip}">
  <span class="bar-label">{BACKEND_LABEL.get(r['backend'], r['backend'])}<span class="sub"> {sub}</span></span>
  <span class="bar-track">{fill}</span>
  <span class="bar-val">{fmt_ms(ms)} ms <span class="sub">{fmt(fps)} fps</span>{extra}</span>
</div>"""
        )
    return f"""<h3>{label}</h3>
<div class="chart" role="img" aria-label="Bar chart of per-frame time per backend, split by segment (lower is better), {label}">
{chr(10).join(bars)}
</div>"""


def seg_legend() -> str:
    chips = "".join(
        f'<span class="chip"><span class="swatch seg-{k}"></span>{SEG_LABEL[k]}</span>'
        for k in DISPLAY_SEGMENTS
    )
    return (f'<div class="legend">{chips}'
            '<span class="chip"><span class="swatch bar-flat"></span>'
            'no breakdown measured</span></div>')


def _machine_tiles(m: dict) -> str:
    return "".join(
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


def _panel(mach: dict) -> str:
    """Full per-machine body (hardware tiles, fps charts, table, videos).

    Reassigns the module global ``DATA`` to this machine so the existing render
    helpers, which read that global, produce this machine's numbers.
    """
    global DATA
    DATA = mach
    n = len([r for r in mach["rows"] if r["mode"] in MODE_LABEL])
    parts = [f"<div class='tiles'>{_machine_tiles(mach['machine'])}</div>"]
    parts.append(
        f"<p class='sub'>{n} shown measurement{'s' if n != 1 else ''} · "
        f"generated {mach.get('generated', '—')} · fps are the no-frame-readback loop "
        f"(frames stay on the GPU; GL renderers sync via a 1×1 readPixels); rows marked "
        f"* were not re-measured and still include readback; rows marked † include it "
        f"inherently (the renderer returns the frame as part of rendering) but it is "
        f"≤1% of the frame time.</p>"
    )
    parts.append(fps_charts())
    parts.append(
        "<details><summary>Full data table</summary>\n"
        "<div class='tablewrap'><table>\n"
        "<thead><tr><th>Backend</th><th>Mode</th><th>Render fps (no readback)</th><th>ms/frame</th>"
        "<th>Breakdown ms (physics / sync / render)</th>"
        "<th>Physics steps/s</th><th>Resolution</th><th>spp</th><th>GPU</th><th>Source</th></tr></thead>\n"
        f"<tbody>\n{table_rows()}\n</tbody>\n</table></div></details>"
    )
    vids = video_cards()
    if "<figure" in vids:
        parts.append(f"<h3>Videos</h3>\n{vids}")
    else:
        parts.append(
            "<p class='sub'>Videos for this machine are on disk "
            "(<code>cube_nexus.mp4</code>, <code>cube_rt_nexus_native.mp4</code>) but not embedded "
            "here; the numbers above are freshly measured on this hardware.</p>"
        )
    return "\n".join(parts)


def main() -> None:
    options = "".join(
        f"<option value='{m['id']}'>{html.escape(m['label'])}</option>" for m in MACHINES
    )
    panels = "".join(
        f"<section class='mpanel' data-mid='{m['id']}'{'' if i == 0 else ' hidden'}>\n{_panel(m)}\n</section>"
        for i, m in enumerate(MACHINES)
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sim2sim render benchmark</title>
<style>
:root {{
  --surface: #fcfcfb; --surface-2: #f1f1ee; --ink: #0b0b0b; --ink-2: #52514e;
  --line: #dddcd6; --accent: #2a78d6; --err: #b3261e;
  --seg-physics: #2a78d6; --seg-sync: #eda100; --seg-render: #1baf7a;
  --seg-readback: #4a3aa7; --seg-flat: #a5a49d;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --surface: #1a1a19; --surface-2: #242422; --ink: #ffffff; --ink-2: #c3c2b7;
           --line: #3a3936; --accent: #3987e5; --err: #e66767;
           --seg-physics: #3987e5; --seg-sync: #c98500; --seg-render: #199e70;
           --seg-readback: #9085e9; --seg-flat: #6a6963; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0 auto; max-width: 1080px; padding: 2rem 1.25rem 4rem;
       font: 15px/1.55 system-ui, sans-serif; background: var(--surface); color: var(--ink); }}
h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; }}
h2 {{ font-size: 1.15rem; margin: 2.5rem 0 .5rem; }}
h3 {{ font-size: 1rem; margin: 1.5rem 0 .5rem; color: var(--ink-2); }}
h2.scene {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--line); }}
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
td:nth-child(n+3):nth-child(-n+8) {{ font-variant-numeric: tabular-nums; }}
.err {{ color: var(--err); }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }}
figure {{ margin: 0; }}
video {{ width: 100%; border-radius: 8px; border: 1px solid var(--line); background: #000; }}
figcaption {{ margin-top: .3rem; font-size: .85rem; }}
figcaption .sub {{ display: block; }}
.badge {{ background: var(--accent); color: #fff; border-radius: 999px;
         padding: .05rem .55rem; font-size: .72rem; font-weight: 600; vertical-align: middle; }}
.chart {{ display: flex; flex-direction: column; gap: 6px; max-width: 680px; margin: .75rem 0 1.25rem; }}
.bar-row {{ display: grid; grid-template-columns: 14.5rem 1fr 9rem; align-items: center; gap: .6rem; }}
.bar-row:hover .bar, .bar-row:focus .bar {{ filter: brightness(1.15); }}
.bar-label {{ font-size: .8rem; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bar-track {{ background: var(--surface-2); border-radius: 4px; height: 14px; overflow: hidden;
              display: flex; }}
.bar {{ display: block; height: 100%; flex: none; }}
.bar + .bar {{ margin-left: 2px; }}
.bar:last-child {{ border-radius: 0 4px 4px 0; }}
.seg-physics {{ background: var(--seg-physics); }}
.seg-sync {{ background: var(--seg-sync); }}
.seg-render {{ background: var(--seg-render); }}
.seg-readback {{ background: var(--seg-readback); }}
.bar-flat {{ background: var(--seg-flat); }}
.bar-val {{ font-size: .8rem; font-variant-numeric: tabular-nums; }}
.legend {{ display: flex; flex-wrap: wrap; gap: .3rem 1rem; margin: .75rem 0 .25rem;
           font-size: .8rem; color: var(--ink-2); }}
.chip {{ display: inline-flex; align-items: center; gap: .35rem; }}
.swatch {{ width: 10px; height: 10px; border-radius: 3px; flex: none; }}
details {{ margin: 1rem 0; }}
summary {{ cursor: pointer; color: var(--ink-2); font-size: .9rem; }}
footer {{ margin-top: 3rem; font-size: .8rem; color: var(--ink-2); }}
code {{ background: var(--surface-2); padding: .05rem .3rem; border-radius: 4px; }}
.hwbar {{ display: flex; align-items: center; gap: .6rem; flex-wrap: wrap;
          margin: 1.25rem 0 .5rem; padding: .7rem .9rem; background: var(--surface-2);
          border: 1px solid var(--line); border-radius: 10px; }}
.hwbar label {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
                color: var(--ink-2); font-weight: 600; }}
.hwbar select {{ font: inherit; font-size: .9rem; font-weight: 600; color: var(--ink);
                 background: var(--surface); border: 1px solid var(--line);
                 border-radius: 8px; padding: .4rem 2rem .4rem .7rem; cursor: pointer;
                 appearance: none;
                 background-image: linear-gradient(45deg, transparent 50%, var(--ink-2) 50%),
                                   linear-gradient(135deg, var(--ink-2) 50%, transparent 50%);
                 background-position: calc(100% - 16px) 55%, calc(100% - 11px) 55%;
                 background-size: 5px 5px, 5px 5px; background-repeat: no-repeat; }}
.hwbar select:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.hwbar .hint {{ font-size: .8rem; color: var(--ink-2); margin-left: auto; }}
.mpanel[hidden] {{ display: none; }}
</style>
</head>
<body>
<h1>sim2sim render benchmark <span class="sub" style="font-size:1rem; font-weight:400;">— 2 scenes, 1 env</span></h1>
<p class="sub">Two single-environment scenes rendered by six physics engines, rasterized and
path-traced natively: <strong>Scene 1</strong> a tilted cube dropped onto a plane, and
<strong>Scene 2</strong> the real LeRobot bipedal platform (STL meshes, 12 DOF) holding a stance
with its built-in PD servos. This measures per-frame render/readback overhead — not the
massively-parallel multi-env workloads the GPU engines (Genesis, mjlab, Isaac, Nexus) are designed
for. Generated by <code>examples/cube_drop/benchmark.py</code> · site by
<code>make_site.py</code>.</p>

<div class="hwbar">
  <label for="hwsel">Hardware</label>
  <select id="hwsel" aria-label="Select hardware">{options}</select>
  <span class="hint">switch machines — numbers, charts, table and videos update</span>
</div>

<h2>Results</h2>
<p class="sub">Raster fps = per-frame loop (physics + render) with frames staying on the GPU —
no full-frame readback (rows not re-measured on that basis are marked). Native-RT
rows are each engine path-tracing its own scene. Not apples-to-apples across resolutions.
On the RTX&nbsp;5090 desktop the three portable backends (MuJoCo, mjlab, Genesis) were re-measured
(median of 3, machine idle), and the Nexus raster + native-RT rows were freshly measured on the 5090
from a full source build of haixuanTao's headless-render Nexus + kiss3d (offscreen WebGPU, no display).
The RT row is far faster than the laptop's because that build includes kiss3d's transform-only TLAS
fast path, which drops the per-frame BLAS re-bake (242&nbsp;ms/frame on the laptop → 2.2&nbsp;ms here).</p>
<p class="sub">Nexus numbers use a patched kiss3d: its <code>read_pixels</code> converted pixels by
indexing uncached mapped GPU memory per byte (~99 ms/frame at 640×480); converting from a cached
row copy + reusing the staging buffer cut readback to ~5 ms (18×) — see
<a href="https://github.com/dimforge/kiss3d/pull/397">kiss3d #397 (readback fix)</a>,
<a href="https://github.com/dimforge/nexus/pull/7">nexus #7 (frame export)</a> and
<a href="https://github.com/dimforge/nexus/pull/8">nexus #8 (Python ray tracing)</a>.</p>
<p class="sub"><strong>Isaac Sim on the desktop — physics yes, rendering no:</strong> the
Omniverse RTX <em>renderer</em> crashes on driver 595.71.05 (known incompatibility with the
R590/595 branch; fix promised in Isaac Sim 6.0), so Isaac's raster/RT rows exist only on the
laptop panel. Headless <em>physics</em> works via Isaac Lab's AppLauncher kit experience (which
skips the crashing RTX plugins) with all debug-vis markers disabled — that's how the desktop's
Scene-3 and Startup Isaac rows were measured. Genesis native ray tracing on the desktop runs via
a from-source LuisaRender build (CUDA backend, pip-wheel CUDA 12.9 toolchain).</p>
<p class="sub">Bars show <strong>time per frame</strong> (lower is better, fastest first), split by
where that time goes: <strong>physics</strong> (solver steps), <strong>sync</strong> (GPU-sim state
to host), <strong>render</strong> (draw / path-trace), <strong>readback</strong> (frame pixels to
CPU). Hover a bar for per-segment ms. Gray bars have no breakdown yet.</p>
{seg_legend()}

{panels}

<footer>sim2sim-locomotion &middot; cube-drop smoke benchmark &middot; single-body scene: this measures
render/readback overhead, not physics scalability.</footer>
<script>
(function() {{
  var sel = document.getElementById('hwsel');
  var panels = Array.prototype.slice.call(document.querySelectorAll('.mpanel'));
  function show(id) {{ panels.forEach(function(p) {{ p.hidden = p.dataset.mid !== id; }}); }}
  sel.addEventListener('change', function() {{ show(sel.value); }});
  show(sel.value);
}})();
</script>
</body>
</html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"[site] wrote {OUT}")


if __name__ == "__main__":
    main()

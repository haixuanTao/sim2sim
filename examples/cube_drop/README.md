# Cube-drop smoke demos

A minimal "hello world" per physics backend: a single rigid cube dropped from
~1.5–3 m with an initial tilt + spin, falling under gravity and settling on the
ground over a few seconds, rendered off-screen to an MP4.

These are **raw-engine sanity checks**, deliberately *not* routed through the
sim2sim adapter / PD-control pipeline (which expects a jointed robot) — they just
confirm each engine simulates + renders on this machine.

| Backend | Script | Hardware | Video |
|---------|--------|----------|-------|
| MuJoCo | [`mujoco_cube.py`](mujoco_cube.py) | CPU + EGL render | [mp4](https://files.catbox.moe/yct6vx.mp4) |
| Genesis | [`genesis_cube.py`](genesis_cube.py) | NVIDIA GPU (CUDA) | [mp4](https://files.catbox.moe/0k4r8o.mp4) |
| mjlab (MuJoCo-Warp) | [`mjlab_cube.py`](mjlab_cube.py) | NVIDIA GPU (CUDA) | [mp4](https://files.catbox.moe/2t1e04.mp4) |
| Nexus (Rapier-on-GPU) | [`nexus_cube.py`](nexus_cube.py) | GPU (WebGPU) | [mp4](https://files.catbox.moe/m3ywsx.mp4) |

## Benchmark harness + results page

**Live page:** https://claude.ai/code/artifact/2dd2404c-743a-44d3-bbf0-a6ab59647619
(multi-machine: RTX 5090 desktop + RTX 5080 laptop panels, no-readback basis,
plus the 2,048-env batch-physics scene; note the link is private until shared
from the artifact's share menu — the canonical copy is `site/index.html` below).

All the numbers below (plus hardware info) are aggregated into
[`benchmark_results.json`](benchmark_results.json) by
[`benchmark.py`](benchmark.py), which also regenerates the static results page
[`site/index.html`](site/index.html) (tables, the Nexus spp sweep, and all the
videos — works from `file://`, GitHub-Pages-ready):

```bash
.venv/bin/python examples/cube_drop/benchmark.py              # fill missing measurements
.venv/bin/python examples/cube_drop/benchmark.py --run all    # re-measure everything
.venv/bin/python examples/cube_drop/benchmark.py --run none   # just regenerate JSON + site
```

## Generation FPS

Frame-generation throughput measured on an **NVIDIA RTX 5080 Laptop GPU**, timing
only the per-frame loop (physics step(s) + render + readback), excluding one-time
setup (model load / GPU pipeline compile).

| Backend | gen-fps | frames / time | Resolution | Per-frame pipeline |
|---------|--------:|---------------|------------|--------------------|
| MuJoCo | **1684** | 150 / 0.09 s | 640×480 | CPU physics + EGL render |
| mjlab | **565** | 150 / 0.27 s | 640×480 | GPU (MuJoCo-Warp) + host EGL render |
| Genesis | **332** | 150 / 0.45 s | 640×480 | GPU physics + render |
| Nexus | **92.1** | 180 / 2.0 s | 640×480 | GPU (WebGPU), headless + pipelined `snap_rgb_async()` readback (patched kiss3d, 2 steps/frame for real-time playback) |
| Nexus (Rapier CPU) | **37.5** | 180 / 4.8 s | 640×480 | Rapier CPU physics, headless + pipelined `snap_rgb_async()` readback (patched kiss3d, 2 steps/frame) |

**Read these numbers with the caveats:**

- The scene is a *single cube*, so this measures **render + readback
  throughput**, not physics scalability — MuJoCo tops the list because
  CPU-stepping one body + a cheap EGL blit is nearly free, while the GPU
  engines pay fixed per-frame overhead that dominates at this trivial scale.
- **Nexus was 2.7 fps before the kiss3d readback fix.** `read_pixels` converted
  BGRA→RGB by indexing the *mapped* staging buffer per byte — mapped readback
  memory is uncached (~10 MB/s), so a 640×480 frame took ~99 ms of pure memory
  stalls. Copying each row into cached memory first (+ reusing the staging
  buffer, + waiting on the copy's submission index instead of the whole device)
  cut the readback to 5.4 ms — kiss3d PR #397.
- **And 33 fps was vsync, not compute.** The windowed viewer presents through a
  swapchain created with kiss3d's default `vsync: true` (Fifo), and the
  blocking per-frame readback prevented frame pipelining, so each loop
  iteration ate ~2 vblanks (~30 ms) regardless of GPU load. Fixed three ways in
  the local kiss3d/nexus3d builds: `NexusViewer(w, h, headless=True)` (no
  window, no swapchain — also works without a display server), `set_vsync()`
  for windowed capture, and `snap_rgb_async()`/`snap_rgb_flush()` (double-buffered
  readback that collects frame N after frame N+1 renders). Result: 33.1 → 92.1
  gen-fps (GPU physics), 26.5 → 37.5 (Rapier CPU, physics-bound).
- **Time scale calibrated with `NexusViewer.body_pose()`**: each nexus solver
  step advances 1/60 s, so 30 fps videos need `set_rbd_steps_per_frame(2)` or
  they play at 0.5× (the cube now lands at frame 23, matching analytic free
  fall's 22.9). At real-time playback: 33.1 gen-fps (WebGPU) vs 26.5 (Rapier
  CPU) — before the readback fix both drowned at ~8.8 fps.
- **Initial velocities were silently dropped** by the `from_rapier` bulk upload
  (zero-filled velocity buffer), so `.angvel(...)` cubes fell without tumbling
  until first contact — found with `body_pose()` (frozen quaternion during free
  fall) and fixed in the `fix/initial-velocities-from-rapier` branch; rotation
  now integrates at exactly |ω|·dt.

## Batch physics — 2,048 parallel envs

[`../lerobot_legs/batch_bench.py`](../lerobot_legs/batch_bench.py) steps the
LeRobot scene with 2,048 parallel environments, physics only (no rendering,
one device sync per timed window). Nexus runs one robot per environment via
the per-env `insert_mjcf` bindings ([dimforge/nexus #16](https://github.com/dimforge/nexus/pull/16))
— packing all robots into env 0 blows GPU memory quadratically (each
contact-constraint slot stores a dense M⁻¹ column sized to the whole env's
DOF count). Results are Scene 3 on the results page. Note an env-step is one
*engine* step and dt differs per engine (Nexus 16.7 ms with the substep
cascade inside; others 5 ms).

```bash
.venv/bin/python examples/lerobot_legs/batch_bench.py --sim genesis --envs 2048
```

## Startup — time to first physics step

[`../lerobot_legs/boot_bench.py`](../lerobot_legs/boot_bench.py) measures the
dev-loop boot cost: process start → one completed physics step of the LeRobot
scene (single env), split into imports / engine init / scene build + first
step. **Warm** = JIT kernel caches populated (every rerun); **cold** = kernel
cache emptied (first-ever run, or after an engine/driver upgrade). Measured on
the RTX 5090 desktop:

| Backend | warm | cold caches |
|---------|-----:|------------:|
| MuJoCo | 0.25 s | = warm (no JIT) |
| Nexus (WebGPU) | 0.90 s | = warm (shaders precompiled in the wheel) |
| Nexus (cuda-oxide + CUDA graph, incl. capture) | 1.12 s | = warm (cubins precompiled) |
| mjlab (MuJoCo-Warp) | 2.08 s | **14.4 s** (Warp kernel compile) |
| Genesis | 5.11 s | **74.2 s** (Taichi JIT: 63.7 s build + 8.5 s first step) |
| Isaac Sim | — | ~1.5 min (laptop; crashes on the desktop's driver 595) + one-time ~2 min RTX shader cache |

Cold runs redirect the kernel caches to an empty dir instead of deleting them
(`XDG_CACHE_HOME` for Genesis — it ignores `TI_OFFLINE_CACHE_FILE_PATH` and
caches under `~/.cache/genesis` — and `WARP_CACHE_PATH` for mjlab).

```bash
.venv/bin/python examples/lerobot_legs/boot_bench.py --sim genesis            # warm
XDG_CACHE_HOME=/tmp/cold .venv/bin/python examples/lerobot_legs/boot_bench.py --sim genesis  # cold
```

## Ray tracing

**Native ray-tracing support is mostly the exception.** These are physics
engines; most built-in renderers are rasterizers. Two engines have a real ray
tracer — **Isaac Sim** (its renderer *is* RTX) and **Genesis** (optional
add-on) — but neither is available out of the box on this machine (Isaac isn't
installed; Genesis's RT backend isn't built).

| Backend | Native renderer | Native ray tracing | RT backend | Status on this machine |
|---------|-----------------|--------------------|------------|------------------------|
| MuJoCo | OpenGL rasterizer (`mujoco.Renderer`) | ❌ none | — | rasterizer only |
| mjlab (MuJoCo-Warp) | host `mujoco.Renderer` (raster) | ❌ none | — | rasterizer only |
| Nexus (Rapier-on-GPU) | kiss3d (WebGPU) | ✅ kiss3d 0.45 GPU path tracer | kiss3d / khal | ✅ **exposed** via `raytrace_frame()`/`snap_rgb()` (local `feat/python-rt-render` build; PR pending) |
| Genesis | rasterizer **+ `RayTracer`** | ✅ optional | LuisaRender (LuisaCompute) | ✅ **built** from source (`~/rt_build/Genesis`, low-mem ninja build) |
| **Isaac Sim / Isaac Lab** | **Omniverse RTX renderer** | ✅ **native, default** — `RayTracedLighting` (real-time RTX) + `PathTracing` (reference) | Omniverse RTX (Kit) | ✅ **installed** (pip `isaacsim`, separate venv at `~/rt_build/isaac-venv`) |

**Isaac has the best RT story.** Its whole renderer is an RTX ray tracer —
real-time ray-traced lighting *and* a full path-traced mode, driven per
camera/render-product. It's the one engine where ray tracing is the *default*
path, not an add-on. It's installed here via the pip `isaacsim` package in a
dedicated Python 3.10 venv (`~/rt_build/isaac-venv`); headless `PathTracing`
mode verified working (`/rtx/rendermode = PathTracing`). Note the **first**
launch compiles the RTX shader cache (~2 min and can look hung/crashy);
subsequent launches start in seconds.

**Genesis is not prebuilt.** `genesis-world` (1.2.1) ships **zero** LuisaRender
files; `genesis/vis/raytracer.py` imports `LuisaRenderPy` from a *source-build*
output dir (`ext/LuisaRender/build/bin`). There is no PyPI wheel — enabling it
means cloning Genesis and compiling `ext/LuisaRender` (CMake/xmake + CUDA C++)
yourself. So out of the box, **no engine here ray-traces its own scene.**
(Both gaps have since been closed on this machine — see the native RT sections
below: Genesis's LuisaRender was compiled from source, and Nexus's path tracer
was exposed to Python on a local bindings branch.)

### Decoupled path-traced demo (`rt_record.py` + `rt_render.py`)

Because the engines can't ray-trace themselves, this demo follows the sim2sim
split of *physics vs rendering*: each backend produces the cube's **pose
trajectory** (`rt_record.py`), then **one shared GPU path tracer** — Mitsuba 3,
`cuda_ad_rgb` variant (OptiX) — renders every trajectory identically
(`rt_render.py`).

> ⚠️ This is **not** a simulator rendering its own scene — it's an external path
> tracer replayed on each engine's trajectory. Because a passive cube drop is
> near-identical across mature engines *and* the renderer is shared, the output
> videos look the same by construction. The only per-engine difference measured
> here is **physics throughput**; the render cost is constant.

Benchmark on the **RTX 5080** (150 frames, 480×360 @ 96 spp, Mitsuba/OptiX):

| Backend | Physics (steps/s) | Ray-trace render (shared) | ms/frame |
|---------|------------------:|--------------------------:|---------:|
| MuJoCo | 408,000 | 35.0 fps | 29 |
| mjlab (GPU) | 7,000 | 38.5 fps | 26 |
| Genesis (GPU) | 3,300 | 37.7 fps | 27 |
| Isaac Sim (PhysX, GPU) | 5,400 | 37.2 fps | 27 |

- **Render is ~27 ms/frame regardless of engine** (same path tracer) and dwarfs
  the physics — path-tracing one cube costs ~1000× a physics step.
- **CPU engines win the physics column** for this single-body / single-env
  workload; Genesis and mjlab are built for *thousands* of parallel envs, so one
  cube leaves the GPU idle. (Timesteps differ, so read this as per-step
  throughput, not identical work.)

```bash
python examples/cube_drop/rt_record.py --sim mujoco --out traj_mujoco.npz
python examples/cube_drop/rt_render.py --traj traj_mujoco.npz \
    --out cube_rt_mujoco.mp4 --label mujoco   # needs: mitsuba imageio imageio-ffmpeg

# Isaac runs in its own venv (pip isaacsim needs Python 3.10):
~/rt_build/isaac-venv/bin/python examples/cube_drop/rt_record.py --sim isaac --out traj_isaac.npz
```

### Native RT: Isaac rendering its own scene (`isaac_rt_native.py`)

Isaac is the one engine that can ray-trace its own scene, so it also gets a
**native** demo (`cube_rt_isaac_native.mp4`): PhysX physics + the Omniverse RTX
renderer in `PathTracing` mode, captured through an Isaac `Camera` sensor — no
external tracer involved. On the RTX 5080 at 480×368 / 64 spp it path-traces at
**~12 fps (83 ms/frame)** vs Mitsuba/OptiX's ~27 ms/frame at 96 spp — the Kit
render pipeline carries far more machinery (sensor pipeline, AOVs, denoiser
hooks) than a bare path tracer.

```bash
~/rt_build/isaac-venv/bin/python examples/cube_drop/isaac_rt_native.py --outdir /tmp/isaac_frames
ffmpeg -y -framerate 30 -i /tmp/isaac_frames/f_%03d.png -pix_fmt yuv420p \
    examples/cube_drop/cube_rt_isaac_native.mp4
```

### Native RT: Genesis rendering its own scene (`genesis_rt_native.py`)

With `ext/LuisaRender` compiled from source (CMake/Ninja + CUDA; use a low
job count — `-j24` OOMs a 30 GB machine, `-j4` works), Genesis path-traces its
own scene via `gs.renderers.RayTracer()`. On the RTX 5080 at 480×368 / 64 spp
(`cube_rt_genesis_native.mp4`):

- physics: **1,887 steps/s** | LuisaRender path trace: **14.3 fps (70 ms/frame)**

```bash
.venv/bin/python examples/cube_drop/genesis_rt_native.py --outdir /tmp/gs_frames
ffmpeg -y -framerate 30 -i /tmp/gs_frames/f_%03d.png -pix_fmt yuv420p \
    examples/cube_drop/cube_rt_genesis_native.mp4
```

### Native RT: Nexus rendering its own scene (`nexus_rt_native.py`)

kiss3d 0.45 ships a wgpu GPU path tracer; the local `feat/python-rt-render`
bindings branch exposes it as `NexusViewer.raytrace_frame()` (progressive
sample accumulation) + `snap_rgb()` → `(H, W, 3)` uint8 numpy. One catch fixed
along the way: with the tracer active, `sync` must take the CPU-readback path,
because the zero-readback kernel updates only the rasterizer's GPU instance
buffers, which the tracer's BVH never reads.

`nexus_rt_bench.py` numbers on the RTX 5080 (480×368 — same as the
genesis/isaac native demos, via `NexusViewer(width, height)`):

| Stage | ms/frame | fps |
|-------|---------:|----:|
| physics step + sync | 5.4 | 184 |
| rasterized `render_frame()` | 4.0 | 249 |
| path trace, 1 spp/call | 4.0 | 250 |
| path trace, 4 spp/call | 9.3 | 107 |
| path trace, 16 spp/call | 30.9 | 32 |
| path trace, 64 spp/call | 119 | 8.4 |
| `snap_rgb()` readback | 57.5 | 17.4 |

- End-to-end video generation at 64 spp: **3.8 fps (~265 ms/frame)**, now
  measured over the whole loop (physics + trace + pipelined readback, headless)
  — same ballpark as Isaac (12 fps) and Genesis (14.3 fps) at the same
  resolution/spp; the tracer itself is the bound.
- The former bottleneck — frame readback — is fixed: kiss3d's `read_pixels`
  converted BGRA→RGB by indexing the *mapped* staging buffer per byte, and
  mapped readback memory is uncached (~10 MB/s). Copying each row to cached
  memory first (+ staging-buffer reuse) took the readback from 57 ms → 3.5 ms
  (99 ms → 5.4 ms at 640×480) — kiss3d branch `feat/persistent-readback-staging`,
  PR pending. Raster capture went from 8.8 → 72.7 gen-fps.

```bash
.venv/bin/python examples/cube_drop/nexus_rt_native.py   # → cube_rt_nexus_native.mp4
.venv/bin/python examples/cube_drop/nexus_rt_bench.py    # → nexus_rt_bench.json
```

## Running

CPU backends run anywhere; GPU backends need an NVIDIA GPU (+ CUDA for
Genesis/mjlab, WebGPU for Nexus). Each script prints a `[fps]` line.

```bash
MUJOCO_GL=egl python examples/cube_drop/mujoco_cube.py
python examples/cube_drop/genesis_cube.py            # GPU
MUJOCO_GL=egl python examples/cube_drop/mjlab_cube.py  # GPU
python examples/cube_drop/nexus_cube.py              # GPU; needs nexus3d with snap_rgb()
```

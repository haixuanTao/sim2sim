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
| PyBullet | [`pybullet_cube.py`](pybullet_cube.py) | CPU + TinyRenderer | [mp4](https://files.catbox.moe/eyh0g5.mp4) |
| Genesis | [`genesis_cube.py`](genesis_cube.py) | NVIDIA GPU (CUDA) | [mp4](https://files.catbox.moe/0k4r8o.mp4) |
| mjlab (MuJoCo-Warp) | [`mjlab_cube.py`](mjlab_cube.py) | NVIDIA GPU (CUDA) | [mp4](https://files.catbox.moe/2t1e04.mp4) |
| Nexus (Rapier-on-GPU) | [`nexus_cube.py`](nexus_cube.py) | GPU (WebGPU) | [mp4](https://files.catbox.moe/m3ywsx.mp4) |

## Benchmark harness + results page

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
| PyBullet | **29** | 150 / 5.14 s | 640×480 | CPU + TinyRenderer (software raster) |
| Nexus | **72.7** | 180 / 2.5 s | 640×480 | GPU (WebGPU) + `render()` readback (patched kiss3d, UI hidden) |
| Nexus (Rapier CPU) | **59.9** | 180 / 3.0 s | 640×480 | Rapier CPU physics + `render()` readback (patched kiss3d) |

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
  cut the readback to 5.4 ms → 72.7 gen-fps. kiss3d branch
  `feat/persistent-readback-staging`, PR pending.
- With the readback fixed, Rapier CPU physics (59.9 fps) is measurably slower
  than the WebGPU backend (72.7 fps) — before the fix both drowned at 8.8 fps.
- PyBullet's 29 fps is bounded by its CPU software rasterizer, not physics.

## Ray tracing

**Native ray-tracing support is mostly the exception.** These are physics
engines; most built-in renderers are rasterizers. Two engines have a real ray
tracer — **Isaac Sim** (its renderer *is* RTX) and **Genesis** (optional
add-on) — but neither is available out of the box on this machine (Isaac isn't
installed; Genesis's RT backend isn't built).

| Backend | Native renderer | Native ray tracing | RT backend | Status on this machine |
|---------|-----------------|--------------------|------------|------------------------|
| MuJoCo | OpenGL rasterizer (`mujoco.Renderer`) | ❌ none | — | rasterizer only |
| PyBullet | TinyRenderer / OpenGL | ❌ none | — | rasterizer only |
| mjlab (MuJoCo-Warp) | host `mujoco.Renderer` (raster) | ❌ none | — | rasterizer only |
| Nexus (Rapier-on-GPU) | kiss3d (WebGPU) | ✅ kiss3d 0.45 GPU path tracer | kiss3d / khal | ✅ **exposed** via `raytrace_frame()`/`render()` (local `feat/python-rt-render` build; PR pending) |
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
| PyBullet | 317,000 | 37.7 fps | 27 |
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

- End-to-end video generation at 64 spp: **4.3 fps (~233 ms/frame)** — same
  ballpark as Isaac (12 fps) and Genesis (14.3 fps) at the same resolution/spp.
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
python examples/cube_drop/pybullet_cube.py
python examples/cube_drop/genesis_cube.py            # GPU
MUJOCO_GL=egl python examples/cube_drop/mjlab_cube.py  # GPU
python examples/cube_drop/nexus_cube.py              # GPU; needs nexus3d with render()
```

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
| Nexus | **2.7** | 180 / 67 s | 1200×900 | GPU (WebGPU) + `render()` readback |

**Read these numbers with the caveats:**

- **Not apples-to-apples.** Nexus renders at 1200×900 (≈3.5× the pixels) and the
  others at 640×480. The scene is a *single cube*, so this measures
  **render + readback throughput**, not physics scalability — MuJoCo tops the
  list because CPU-stepping one body + a cheap EGL blit is nearly free, while the
  GPU engines pay fixed per-frame overhead that dominates at this trivial scale.
- **Nexus's 2.7 fps is a readback limitation, not the engine.** Per-frame
  profiling: `render_frame` 2.9 ms, `simulate` 1.8 ms, `sync` 0.1 ms,
  **framebuffer `render()` 365 ms**. The GPU render is fast (~3 ms); the cost is
  a synchronous GPU→CPU framebuffer readback (`kiss3d::read_pixels`: fresh
  staging buffer + full `device.poll(wait_indefinitely)` stall + CPU BGRA→RGB
  flip, every frame). A release build (2.68 fps) ≈ debug (2.36 fps), confirming
  it's a sync-stall, not CPU compute. Nexus's `render()` itself is a new addition
  (see the upstream PR); making it real-time is a follow-up kiss3d optimization
  (persistent staging buffer + pipelined async readback).
- PyBullet's 29 fps is bounded by its CPU software rasterizer, not physics.

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

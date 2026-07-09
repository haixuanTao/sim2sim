# sim2sim-locomotion

Evaluate **one** locomotion policy across **multiple physics simulators** —
MuJoCo, mjlab (MuJoCo-Warp), Genesis, Nexus, and Isaac Lab — and compare
the results side by side. Targets the **LeRobot legged** ecosystem: train a policy in
mjlab, then sim-to-sim check it everywhere.

This is the classic **sim-to-sim** robustness check that sits between training
and sim-to-real. If a policy behaves consistently across simulators with
different contact solvers, integrators, and actuator models, it is far more
likely to transfer to real hardware. Divergence between simulators flags a
policy that has overfit to one simulator's quirks.

The policy is loaded from **ONNX**, so the *exact same* `.onnx` file is fed to
every backend — no per-simulator reimplementation of the network.

---

## Why this design works

The hard part of sim-to-sim is **not** wiring up simulators. It is guaranteeing
that every backend hands the policy a **bit-for-bit identical observation
vector** and applies actions through an **identical control law**. So those two
things are centralized and the simulators are kept deliberately thin:

```
            ┌──────────────────────── runner (episode loop) ────────────────────────┐
 RobotState │  ObservationBuilder ── obs ──▶ Policy(.onnx) ── action ──▶ PDController │ torque
   ◀────────┤        (obs/)                  (policy/)                   (control/)   ├────────▶
            └─────────────────────────────────────────────────────────────────────-┘
                         ▲                                                    │
                         │  get_state()                          apply_torques()/step()
                         └────────────────  Simulator adapter  ──────────────┘
                             (mujoco | genesis | nexus | isaaclab)
```

- A neutral [`RobotState`](src/sim2sim/sim/state.py) is the only thing adapters
  produce. Every adapter reports joints in one **canonical order** (a
  joint-order mismatch across simulators is the #1 sim-to-sim bug).
- [`ObservationBuilder`](src/sim2sim/obs/observation.py) assembles the policy
  input from `RobotState` + command in a **config-driven term order with
  per-term scales**, so a policy trained with a given observation layout is
  reproduced exactly on every backend.
- [`PDController`](src/sim2sim/control/actuation.py) maps actions to torques
  with one shared law: `tau = Kp·(default + scale·action − q) − Kd·qd`.

Adding a new simulator is one file implementing the
[`Simulator`](src/sim2sim/sim/base.py) interface.

---

## Hardware / availability matrix

| Simulator | Install | Hardware | Runs in CI / here |
|-----------|---------|----------|-------------------|
| **MuJoCo** | `pip install mujoco` | CPU | ✅ yes (reference backend) |
| **mjlab** | `pip install mjlab` | **NVIDIA GPU + CUDA** | ⚠️ GPU host only |
| **Genesis** | `pip install genesis-world` | **NVIDIA GPU + CUDA** | ⚠️ GPU host only |
| **Nexus** | `pip install dimforge-nexus3d` | **GPU (WebGPU)** | 🚧 integrated, not yet eval-runnable |
| **Isaac Lab** | NVIDIA Isaac Sim stack (out-of-band) | **NVIDIA GPU + CUDA** | ⚠️ GPU host only |

[**mjlab**](https://github.com/mujocolab/mjlab) (MuJoCo-Warp, Isaac Lab-style
API) is the simulator the **LeRobot legged** stack trains in — see
[LeRobot integration](#lerobot-legged-integration). The mjlab, Genesis, and
Isaac Lab adapters are written against their real APIs but cannot execute on a
CPU host. They are **import-guarded**: the registry reports them unavailable and
skips them instead of crashing. Their live behavior is validated on a GPU host
(see [GPU runbook](#gpu-runbook)).

[**Nexus**](https://github.com/dimforge/nexus) is dimforge's GPU (WebGPU) engine —
"Rapier on the GPU" — with `nexus3d` Python bindings. It is **registered but not
yet eval-runnable**: the current bindings expose world setup + MJCF loading but
not the per-joint torque input or CPU state read-back the [`Simulator`](src/sim2sim/sim/base.py)
contract needs (`NexusViewer` is a windowed viewer with no off-screen read-back).
`NexusSimulator.is_available()` probes for those methods and reports unavailable
until dimforge adds them, so it is skipped rather than half-run. See
[`nexus_adapter.py`](src/sim2sim/sim/nexus_adapter.py) for details.

---

## Install

```bash
# CPU backend (MuJoCo) + dev tools
pip install -e ".[all-cpu,report]"

# or individually
pip install -e ".[mujoco]"
pip install -e ".[mjlab]"       # GPU host (LeRobot legged training sim)
pip install -e ".[genesis]"     # GPU host
pip install -e ".[nexus]"       # GPU host (dimforge Nexus / WebGPU)
# Isaac Lab is installed via the NVIDIA Isaac Sim installer, not this extra.

# everything for development (tests, lint, MuJoCo, onnx)
pip install -e ".[dev]"
```

## Quickstart

```bash
# Which backends are available on this machine?
sim2sim list-sims

# Evaluate across all configured simulators (random baseline if no ONNX set)
sim2sim eval --config configs/eval.yaml --out report

# Restrict to specific simulators / policy kind
sim2sim eval --config configs/eval.yaml --sims mujoco --policy random
```

Output is a markdown table + bar-chart PNG in `report/`:

```
| Metric              | mujoco          |
|---------------------|-----------------|
| Survival rate ↑     | 1.000 ± 0.000   |
| Survival time (s) ↑ | 10.000 ± 0.000  |
| Distance (m) ↑      | 0.163 ± 0.000   |
| Lin-vel err ↓       | 0.620 ± 0.216   |
| ...                 | ...             |
```

(The numbers above are the **random baseline** on the bundled toy robot — they
illustrate the report, not a good policy. Drop in a trained ONNX policy for a
meaningful comparison.)

## Using your own policy

1. Export your locomotion policy to ONNX (single input → single output).
2. Point `configs/policy/quad12_flat.yaml` at it and make `obs_terms` match the
   **exact observation layout and scales** the policy was trained with.
3. Run `sim2sim eval`. The obs dimension is validated against the ONNX graph at
   load time, so a layout mismatch fails fast instead of producing garbage.

## Bundled robots

Two config-driven robots ship in the box, each as **both MJCF** (MuJoCo / mjlab /
Genesis / Nexus) **and URDF** (Isaac converts URDF→USD) so the **same
morphology** loads in every backend:

| Robot | Config | DOF | Notes |
|-------|--------|-----|-------|
| `lerobot_legs` | `configs/robot/lerobot_legs.yaml` | 12 | Bipedal "legs" stand-in for the **LeRobot Humanoid** (2 legs × {hip yaw/roll/pitch, knee, ankle pitch/roll}) |
| `quad12` | `configs/robot/quad12.yaml` | 12 | Quadruped (Go2-like; 4 legs × {hip, thigh, calf}) |

Both are **generic stand-ins** so the harness runs without external/proprietary
assets. For a high-fidelity study, drop the real model into `assets/` and point
the robot config at it — **no code change** is needed:

- **LeRobot Humanoid legs**: use the `lerobot-humanoid-model` MJCF/URDF.
- **Go2**: use `mujoco_menagerie`'s Go2 MJCF + a Go2 URDF.

## LeRobot legged integration

LeRobot's legged stack — [`lerobot-legged-zoo`](https://huggingface.co/blog/VirgileBatto/lerobot-humanoid)
(MJLab training envs for the LeRobot Humanoid and other legged robots),
[`mjlab`](https://github.com/mujocolab/mjlab) (MuJoCo-Warp), and
`unitree_rl_mjlab` (Unitree Go2/G1/H1) — trains locomotion policies in **mjlab**.
This project lets you take such a policy and check it **sim-to-sim**: run it in
mjlab *and* in MuJoCo/Genesis/Isaac and compare.

```bash
# Evaluate the LeRobot Humanoid legs across all backends (GPU ones auto-skip on CPU)
sim2sim eval --config configs/eval_lerobot.yaml
```

**Exporting a LeRobot / mjlab policy to ONNX.** The policy is consumed as ONNX so
it is sim-agnostic. After training (e.g. with `unitree_rl_mjlab` / rsl_rl),
export the actor MLP:

```python
import torch

policy = runner.alg.actor_critic.actor   # your trained actor (obs -> action)
obs_dim = 48                             # must match configs/policy/lerobot_legs_flat.yaml
dummy = torch.zeros(1, obs_dim)
torch.onnx.export(
    policy, dummy, "lerobot_legs_policy.onnx",
    input_names=["obs"], output_names=["action"],
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=13,
)
```

Then set `onnx_path: lerobot_legs_policy.onnx` in
`configs/policy/lerobot_legs_flat.yaml` and make `obs_terms` (order + scales)
match the observation the policy was trained on. The obs dimension is validated
against the ONNX graph at load time, so a mismatch fails fast.

## Configuration

| File | Purpose |
|------|---------|
| `configs/robot/quad12.yaml` | joint order, default pose, PD gains, action scale, asset paths |
| `configs/policy/quad12_flat.yaml` | observation term order + scales, command ranges, ONNX path |
| `configs/eval.yaml` | which sims, episodes, seeds, control rate, fall thresholds |

## Metrics

Per-episode, aggregated to mean ± std across seeds:
survival rate / time, distance travelled, linear & angular **velocity-tracking
error**, mean torque, **cost of transport**, and action smoothness.

## GPU runbook

On a host with an NVIDIA GPU + CUDA:

```bash
pip install -e ".[mjlab,report]"            # mjlab (LeRobot legged training sim)
sim2sim eval --config configs/eval_lerobot.yaml --sims mujoco,mjlab

pip install -e ".[genesis,report]"          # Genesis
sim2sim eval --config configs/eval_lerobot.yaml --sims mujoco,genesis

# Nexus (dimforge / WebGPU): pip install -e ".[nexus]" — registered but not yet
# eval-runnable (see availability matrix); skipped by the registry for now.

# Isaac Lab: install Isaac Sim + Isaac Lab per NVIDIA docs, then run with the
# Isaac Lab python:
./isaaclab.sh -p -m sim2sim.cli eval --config configs/eval_lerobot.yaml --sims isaaclab
```

GPU-only tests are marked `@pytest.mark.gpu` (none required to ship; the CPU
suite fully exercises the shared pipeline).

## Testing

```bash
pytest -m "not gpu"        # full CPU suite (parity, actuation, ONNX, smoke)
ruff check src tests
ruff format --check src tests
```

The CPU suite covers the sim-to-sim contract directly: observation parity,
deterministic actuation, the ONNX obs-dim contract, registry availability, and
end-to-end MuJoCo rollouts.

## Layout

```
src/sim2sim/
  config.py                # dataclasses + YAML loader
  sim/                     # Simulator ABC, RobotState, registry, 5 adapters:
                           #   mujoco, mjlab, genesis, nexus, isaaclab
  obs/                     # ObservationBuilder + command generator
  control/                 # shared action->torque PD law
  policy/                  # Policy protocol, ONNX policy, baselines
  eval/                    # metrics, runner, report
  cli.py                   # `sim2sim eval` / `sim2sim list-sims`
  assets/lerobot_legs/     # bundled LeRobot Humanoid legs MJCF + URDF
  assets/quad12/           # bundled quadruped MJCF + URDF
configs/  examples/  tests/
```

## License

MIT.

## TODO

- [ ] Temporal Ray Tracing 
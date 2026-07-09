# Model card — zealot biped walking policies (v6, v7)

## Summary
Velocity-tracking locomotion policies for the **LeRobot bipedal** (12 actuated DOF,
~12.7 kg, torque-PD position control @ 50 Hz). Trained with PPO on **nexus GPU
physics** (all-Rust stack, Mac Metal backend, 1024 parallel envs, ~250k samples/iter,
2000 iterations per generation). Two artifacts per version:

- `*.safetensors` — actor + critic + Adam-resumable normalizer state (training format)
- `*.onnx` — **deploy format**: obs normalizer + actor baked into one graph,
  verified to ≤1e-5 against the training-side forward. Input `obs [N,45]` float32,
  output `action [N,12]` float32.

## Architecture
Actor MLP `[45, 256, 256, 128, 12]`, ELU hidden activations, linear output
(deterministic mean action; exploration noise is train-only). Critic (safetensors
only) `[51, 512, 256, 128, 1]` with privileged base-velocity obs. Observation
normalization: running mean/var (Welford), `clip((obs-mean)/std, ±5)` — baked into
the ONNX graph.

## Observation (45-dim, in order)
| idx | block | notes |
|---|---|---|
| 0–11 | last_action | previous policy output, 2-step lag, zeroed for 2 steps post-reset |
| 12–15 | command | [vx m/s, vy m/s, yaw_rate rad/s, 0] |
| 16–27 | joint_pos | rad, relative to default pose (= 0 for this model) |
| 28–39 | joint_vel | rad/s, finite-diff @ 50 Hz, zeroed 2 steps post-reset |
| 40–42 | projected_gravity | world −Z expressed in base frame |
| 43–44 | gait clock | (sin 2πφ, cos 2πφ); φ advances 0.02/period per step, 0 at reset |

**Gait period: 0.9 s (v7), 0.7 s (v6).** Joint order: alphabetical (mjlab convention).

## Action (12-dim @ 50 Hz)
Joint-position offsets: `q_target = default_pos + action_scale * action`, driven by
per-joint torque-PD (kp/kv/effort: hips 120–240 / 6–8 / 88 N·m, knee 240/8/88,
ankles 40/2.25/15). action_scale per joint family: hipz 0.733, hipx 0.55,
hipy 0.367, knee 0.367, ankle 0.55 (see `examples/biped/sim2sim_xval.py`).

## Training distribution
Commands uniform vx ∈ ±0.5 m/s, vy ∈ ±0.3, yaw ∈ ±0.5 rad/s, resampled every 3–8 s,
25% standing (v7; 10% v6). Push perturbations every ~3.5 s ±50%: ±0.5 m/s linear,
±0.25 rad/s angular. DR over friction/restitution/PD-scale/contact softness/spawn
pose (32 templates). Obs noise on (Isaac-style amplitudes). v7 adds: CoM-centering 3.0,
stand-planted penalty −1.0, swing ratio 0.35.

## Evaluation (18 s, nexus engine unless noted)
| metric | v6 | v7 |
|---|---|---|
| vx achieved @ 0.3 m/s cmd | 0.448 | 0.539 |
| yaw achieved @ 0.4 rad/s cmd | 0.284 | 0.02 (stands) |
| falls @ 2× push stress | 0 | 0 (even standing) |
| MuJoCo sim2sim survival (12 s cap) | 2.7 s | 12 s |
| MuJoCo fwd progress @ 0.4 cmd | fell | −0.02 m/s |

## Intended use & limitations
Research checkpoints for sim evaluation and sim2sim transfer studies. **Not
deployment-ready for hardware**: no policy generation yet converts stepping into
forward propulsion under MuJoCo's mesh-foot contact model (training uses capsule
feet — root-caused via open-loop action replay). v7 does not turn in place.
Evaluate v7 with `BIPED_GAIT_PERIOD=0.9` or the gait-clock obs will be off-distribution.

## Provenance
Trained 2026-07-06 → 2026-07-08 on Apple Silicon (Metal/WebGPU via naga MSL fix),
lineage v4 → v5 (+pushes) → v6 (+full-speed cmds, wide yaw) → v7 (+deliberate gait).
Repro recipes in the release notes. Export: `examples/biped/export_onnx.py`.

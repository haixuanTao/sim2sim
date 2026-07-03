# sim2sim-rapier

Rapier (Rust) physics backend for the [sim2sim](../../README.md) locomotion
harness. Builds a native Python extension module, `sim2sim_rapier`, via
[PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).

It is intentionally a **pure native extension** — it only exposes the raw
physics. The thin Python adapter that plugs it into the harness
(`RapierSimulator`) lives in the core `sim2sim` package at
`src/sim2sim/sim/rapier_adapter.py`, so it can call the same shared state
helpers (base-frame velocity, projected gravity, canonical joint ordering) as
every other backend.

## Build

```bash
# from the repo root, into your active env:
pip install maturin
maturin develop --release -m crates/sim2sim-rapier/Cargo.toml
# or, equivalently, the extra:
pip install -e ".[rapier]"
```

Then it is auto-discovered:

```bash
sim2sim --sims rapier            # once the wheel is importable
```

## Design

- **Reduced-coordinate `Multibody`** — rapier's articulated representation, so
  joint positions/velocities read out cleanly per joint.
- **Torque control** — each control torque is applied as a world-space torque
  about the revolute axis on the child body; rapier's articulated solver
  projects it into the correct generalised force. Torques are re-applied every
  physics step (rapier clears user forces per step), matching PyBullet's
  `TORQUE_CONTROL`.
- **URDF** — loaded with `rapier3d-urdf` from the *same* `quad12.urdf` the other
  backends use, so the policy faces an identical morphology.
- Native state is reported raw (`xyzw` quaternion, world-frame velocities); the
  Python adapter converts to the framework's conventions.

## Status / validation

⚠️ The torque→axis mapping and the joint-angle sign convention are the pieces to
validate numerically against the MuJoCo reference backend. Cross-check a short
rollout (base height, joint angles, survival) against `--sims mujoco` before
trusting the metrics.

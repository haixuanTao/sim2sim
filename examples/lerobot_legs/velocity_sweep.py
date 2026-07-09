"""Velocity-tracking sweep for the zealot policies (MuJoCo / Genesis).

For each (policy, command) point, runs one deterministic episode and records
the mean base-frame velocity over the alive window (after a 0.5 s transient),
i.e. how well the policy tracks its command in each simulator. Commands:
vx sweep at yaw=0, and a yaw sweep at vx=0.2.

Run:  python examples/lerobot_legs/velocity_sweep.py --sims mujoco,genesis
Isaac has the same sweep via:  isaac_zealot.py --sweep (its own venv).
Output: report/velocity_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

VX_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
YAW_SWEEP = [-0.3, 0.3]
TRANSIENT_S = 0.5


def run_point(sim, policy, robot_cfg, policy_cfg, eval_cfg, command):
    """One fixed-command episode; returns alive-window velocity means."""
    from sim2sim.control.actuation import PDController
    from sim2sim.obs.observation import ObservationBuilder

    obs_builder = ObservationBuilder(policy_cfg, robot_cfg)
    pd = PDController(robot_cfg)
    policy.reset()
    obs_builder.reset()
    state = sim.reset()

    decimation = max(1, round(eval_cfg.control_dt / sim.dt))
    skip = int(TRANSIENT_S / eval_cfg.control_dt)
    vx, vy, wz, alive = [], [], [], 0
    cmd = np.asarray(command, dtype=np.float32)

    for step in range(eval_cfg.max_steps):
        obs = obs_builder.build(state, cmd)
        action = policy.act(obs)
        obs_builder.set_last_action(action)
        for _ in range(decimation):
            tau = pd.compute_torque(action, state.joint_pos, state.joint_vel)
            sim.apply_torques(tau)
            sim.step()
            state = sim.get_state()
        if state.base_pos[2] < eval_cfg.fall_height or state.projected_gravity[2] > -eval_cfg.fall_tilt:
            break
        alive = step + 1
        if step >= skip:
            vx.append(float(state.base_lin_vel[0]))
            vy.append(float(state.base_lin_vel[1]))
            wz.append(float(state.base_ang_vel[2]))

    return {
        "cmd": [float(c) for c in command],
        "survival_s": alive * eval_cfg.control_dt,
        "n_samples": len(vx),
        "vx": float(np.mean(vx)) if vx else None,
        "vy": float(np.mean(vy)) if vy else None,
        "yaw_rate": float(np.mean(wz)) if wz else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default="mujoco,genesis")
    ap.add_argument("--out", default=str(REPO / "report/velocity_sweep.json"))
    args = ap.parse_args()

    from sim2sim.config import EvalCfg
    from sim2sim.policy.onnx_policy import OnnxPolicy
    from sim2sim.obs.observation import ObservationBuilder
    from sim2sim.sim import registry

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for sim_name in args.sims.split(","):
        for ver in ("v7", "v6"):
            eval_cfg = EvalCfg.from_yaml(REPO / f"configs/eval_zealot_{ver}.yaml")
            robot_cfg = eval_cfg.load_robot()
            policy_cfg = eval_cfg.load_policy()
            sim = registry.make(sim_name)
            sim.load(robot_cfg, seed=0)
            dim = ObservationBuilder(policy_cfg, robot_cfg).dim
            policy = OnnxPolicy(str(policy_cfg.onnx_path), dim,
                                clip_actions=policy_cfg.clip_actions)
            points = [[v, 0.0, 0.0] for v in VX_SWEEP]
            points += [[0.2, 0.0, w] for w in YAW_SWEEP]
            rows = [run_point(sim, policy, robot_cfg, policy_cfg, eval_cfg, c) for c in points]
            sim.close()
            results[f"{sim_name}_{ver}"] = rows
            print(f"[{sim_name} {ver}]")
            for r in rows:
                print(f"  cmd {r['cmd']} -> vx {r['vx']}, yaw {r['yaw_rate']}, "
                      f"alive {r['survival_s']:.2f}s ({r['n_samples']} samples)")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()

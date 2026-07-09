"""Record videos of a zealot walking policy across simulators.

Runs one seeded episode per simulator with off-screen capture and writes
``report/videos/zealot_<policy>_<sim>.mp4``. Uses the same eval configs and
episode loop as ``sim2sim eval`` (stop_on_fall=False so the fall is visible).

Run:  MUJOCO_GL=egl python examples/lerobot_legs/record_zealot.py [v7|v6] [sims...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio

from sim2sim.config import EvalCfg
from sim2sim.eval.runner import run_episode
from sim2sim.obs.commands import CommandGenerator
from sim2sim.obs.observation import ObservationBuilder
from sim2sim.policy.onnx_policy import OnnxPolicy
from sim2sim.sim import registry

REPO = Path(__file__).resolve().parents[2]
FPS = 50  # one frame per control step (control_dt = 0.02)


def main() -> None:
    ver = sys.argv[1] if len(sys.argv) > 1 else "v7"
    sims = sys.argv[2:] or ["mujoco", "genesis"]
    eval_cfg = EvalCfg.from_yaml(REPO / f"configs/eval_zealot_{ver}.yaml")
    robot_cfg = eval_cfg.load_robot()
    policy_cfg = eval_cfg.load_policy()

    out_dir = REPO / "report/videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in sims:
        if not registry.is_available(name):
            print(f"[skip] {name} unavailable")
            continue
        sim = registry.make(name)
        sim.load(robot_cfg, capture=True, seed=0)
        obs_builder = ObservationBuilder(policy_cfg, robot_cfg)
        cmd_gen = CommandGenerator(policy_cfg)
        policy = OnnxPolicy(
            policy_cfg.onnx_path, obs_builder.dim, clip_actions=policy_cfg.clip_actions
        )
        frames: list = []
        m = run_episode(sim, policy, robot_cfg, obs_builder, cmd_gen, eval_cfg,
                        seed=0, frame_sink=frames.append, stop_on_fall=False)
        sim.close()
        out = out_dir / f"zealot_{ver}_{name}.mp4"
        imageio.mimsave(out, frames, fps=FPS)
        print(f"[wrote] {out}  ({len(frames)} frames, survived {m.survival_time:.2f}s)")


if __name__ == "__main__":
    main()

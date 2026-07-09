"""Overlay all 50 eval episodes of a (sim, policy) cell into ONE ghost video.

Same fixed camera for every episode, then per-frame lighten blend (pixel max):
all 50 robots run simultaneously in a single full-resolution scene. Episodes
that have ended keep their last frame as a dim shadow, so fallen robots pile
up visibly while survivors keep walking.

Memory stays flat: a running max-canvas (n_frames x H x W x 3) is updated one
episode at a time. Output: report/videos/overlay_<ver>_<sim>.mp4 (640x480).

Run:  MUJOCO_GL=egl python examples/lerobot_legs/record_overlay.py <sim> [ver]
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FPS = 25  # every 2nd control step
RENDER_EVERY = 2
DIM = 0.45  # brightness of finished episodes' frozen shadow


def run_cell(sim_name: str, ver: str) -> Path:
    from sim2sim.config import EvalCfg
    from sim2sim.control.actuation import PDController
    from sim2sim.obs.commands import CommandGenerator
    from sim2sim.obs.observation import ObservationBuilder
    from sim2sim.policy.onnx_policy import OnnxPolicy
    from sim2sim.sim import registry

    eval_cfg = EvalCfg.from_yaml(REPO / f"configs/eval_zealot_{ver}.yaml")
    robot_cfg = eval_cfg.load_robot()
    policy_cfg = eval_cfg.load_policy()

    sim = registry.make(sim_name)
    sim.load(robot_cfg, capture=True, seed=0)
    obs_builder = ObservationBuilder(policy_cfg, robot_cfg)
    cmd_gen = CommandGenerator(policy_cfg)
    policy = OnnxPolicy(str(policy_cfg.onnx_path), obs_builder.dim,
                        clip_actions=policy_cfg.clip_actions)
    pd = PDController(robot_cfg)

    blend = sys.argv[3] if len(sys.argv) > 3 else "max"
    n_frames = eval_cfg.max_steps // RENDER_EVERY
    canvas: np.ndarray | None = None  # (n_frames, H, W, 3) running max
    acc: np.ndarray | None = None  # ghost blend: float sum of frames
    bgmin: np.ndarray | None = None  # ghost blend: per-pixel min = background plate

    for seed in eval_cfg.seeds[: eval_cfg.episodes]:
        rng = np.random.default_rng(seed)
        command = cmd_gen.sample(rng)
        policy.reset()
        obs_builder.reset()
        from sim2sim.sim.base import InitState
        init = (InitState.sample(rng, robot_cfg, eval_cfg.init_noise)
                if eval_cfg.init_noise else None)
        state = sim.reset(init)
        decimation = max(1, round(eval_cfg.control_dt / sim.dt))
        fell = False
        last = None
        fi = 0
        for step in range(eval_cfg.max_steps):
            if not fell:
                obs = obs_builder.build(state, command)
                action = policy.act(obs)
                obs_builder.set_last_action(action)
                for _ in range(decimation):
                    tau = pd.compute_torque(action, state.joint_pos, state.joint_vel)
                    sim.apply_torques(tau)
                    sim.step()
                    state = sim.get_state()
                fell = (state.base_pos[2] < eval_cfg.fall_height
                        or state.projected_gravity[2] > -eval_cfg.fall_tilt)
            if step % RENDER_EVERY == 0:
                if not fell or last is None:
                    last = sim.render()
                    frame = last
                else:
                    frame = (last * DIM).astype(np.uint8)  # frozen dim shadow
                if canvas is None:
                    canvas = np.zeros((n_frames,) + frame.shape, dtype=np.uint8)
                    if blend == "ghost":
                        acc = np.zeros((n_frames,) + frame.shape, dtype=np.float32)
                        bgmin = np.full((n_frames,) + frame.shape, 255, dtype=np.uint8)
                if blend == "ghost":
                    acc[fi] += frame
                    np.minimum(bgmin[fi], frame, out=bgmin[fi])
                else:
                    np.maximum(canvas[fi], frame, out=canvas[fi])
                fi += 1
        print(f"  seed {seed}: {'fell' if fell else 'survived'}", flush=True)
    sim.close()

    n_eps = eval_cfg.episodes
    suffix = "_ghost" if blend == "ghost" else ""
    out_path = REPO / f"report/videos/overlay_{ver}_{sim_name}{suffix}.mp4"
    writer = imageio.get_writer(out_path, fps=FPS, macro_block_size=8)
    for i in range(n_frames):
        if blend == "ghost":
            # background plate + boosted mean deviation: one lone robot renders
            # at ~15% opacity, ~7 overlapping robots saturate to solid.
            bg = bgmin[i].astype(np.float32)
            dev = acc[i] - n_eps * bg  # sum of (frame - bg)
            f = np.clip(bg + dev * (7.0 / n_eps), 0, 255).astype(np.uint8)
        else:
            f = canvas[i]
        writer.append_data(f)
    writer.close()
    print(f"[wrote] {out_path}")
    return out_path


if __name__ == "__main__":
    run_cell(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "v7")

"""Render all 50 eval episodes of a (sim, policy) cell as one tiled mosaic video.

Runs the same 50 seeded episodes as the eval harness (same CommandGenerator
seeds), captures each at low resolution, and tiles them into a 10x5 grid.
Finished episodes freeze on their last frame and dim, so surviving robots are
visually obvious. Output: report/videos/grid_<ver>_<sim>.mp4

Run:  MUJOCO_GL=egl python examples/lerobot_legs/record_grid.py <sim> <ver>
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parents[2]
COLS, ROWS = 10, 5
TILE_W, TILE_H = 160, 120  # capture 640x480 downsampled by 4
FPS = 12.5  # render every 4th control step
RENDER_EVERY = 4


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

    episodes: list[list[np.ndarray]] = []
    alive_frames: list[int] = []  # frame index at which the episode fell
    n_eps = min(eval_cfg.episodes, COLS * ROWS)
    for seed in eval_cfg.seeds[:n_eps]:
        rng = np.random.default_rng(seed)
        command = cmd_gen.sample(rng)
        policy.reset()
        obs_builder.reset()
        from sim2sim.sim.base import InitState
        init = (InitState.sample(rng, robot_cfg, eval_cfg.init_noise)
                if eval_cfg.init_noise else None)
        state = sim.reset(init)
        decimation = max(1, round(eval_cfg.control_dt / sim.dt))
        frames: list[np.ndarray] = []
        fell_frame = None
        for step in range(eval_cfg.max_steps):
            obs = obs_builder.build(state, command)
            action = policy.act(obs)
            obs_builder.set_last_action(action)
            for _ in range(decimation):
                tau = pd.compute_torque(action, state.joint_pos, state.joint_vel)
                sim.apply_torques(tau)
                sim.step()
                state = sim.get_state()
            if step % RENDER_EVERY == 0:
                f = sim.render()
                if f is not None:
                    frames.append(np.ascontiguousarray(f[::4, ::4, :]))
            fell = (state.base_pos[2] < eval_cfg.fall_height
                    or state.projected_gravity[2] > -eval_cfg.fall_tilt)
            if fell and fell_frame is None:
                fell_frame = len(frames)
            if fell and step % RENDER_EVERY == 0 and step * eval_cfg.control_dt > 0:
                # keep rendering half a second past the fall, then stop
                if fell_frame is not None and len(frames) - fell_frame > int(0.5 * FPS):
                    break
        episodes.append(frames)
        alive_frames.append(fell_frame if fell_frame is not None else len(frames))
        print(f"  seed {seed}: {len(frames)} frames, "
              f"{'survived' if fell_frame is None else f'fell @ frame {fell_frame}'}",
              flush=True)
    sim.close()

    # Tile into the mosaic; freeze+dim finished episodes.
    n_frames = max(len(f) for f in episodes)
    out_path = REPO / f"report/videos/grid_{ver}_{sim_name}.mp4"
    writer = imageio.get_writer(out_path, fps=FPS, macro_block_size=8)
    for i in range(n_frames):
        canvas = np.zeros((ROWS * TILE_H, COLS * TILE_W, 3), dtype=np.uint8)
        for e, frames in enumerate(episodes):
            r, c = divmod(e, COLS)
            if i < len(frames):
                tile = frames[i]
            else:
                tile = (frames[-1] * 0.35).astype(np.uint8)  # done: freeze + dim
            canvas[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W] = tile[:TILE_H, :TILE_W]
        writer.append_data(canvas)
    writer.close()
    print(f"[wrote] {out_path} ({n_frames} frames)")
    return out_path


if __name__ == "__main__":
    run_cell(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "v7")

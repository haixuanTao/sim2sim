"""Run a zealot walking policy in Isaac Sim (free base) and record a video.

Standalone Isaac Sim script (Isaac lives in its own venv; Isaac Lab is not
installed, so this drives isaacsim.core directly). Reuses the sim2sim harness
pieces — ObservationBuilder, PDController, OnnxPolicy, configs — so the policy
sees exactly the same 45-dim contract as the MuJoCo/Genesis evals.

URDF import quirks are inherited from isaac_render.py (unique-mesh symlinks,
re-applied joint limits/friction). Unlike that demo the base is FREE: the
policy actively balances via external torque-PD (drive gains zeroed, efforts
applied at the physics rate).

Kit swallows stdout, so results are written to <outdir>/result.txt.

Run:
  ~/rt_build/isaac-venv/bin/python examples/lerobot_legs/isaac_zealot.py \
      --outdir /tmp/isaac_zealot [--ver v7] [--vx 0.3] [--duration 12]
Then encode:
  ffmpeg -y -framerate 25 -i /tmp/isaac_zealot/f_%03d.png -pix_fmt yuv420p \
      report/videos/zealot_v7_isaac.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ASSET_DIR = (
    Path.home() / "Documents/work/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms"
)
URDF = ASSET_DIR / "urdf/robot.urdf"

DT = 0.005
CONTROL_DT = 0.02
FPS = 25  # one frame per two control steps
W, H, SPP = 640, 480, 32
R2D = 180.0 / math.pi

# joint limits (rad) and frictionloss (N*m) from the MJCF robot.xml
LIM = {"hipz": (-0.349, 0.349), "hipx": (-0.349, 0.349), "hipy": (-1.047, 1.047), "knee": (-0.524, 0.524)}
LIM_SIDE = {
    ("ankley", "right"): (-0.175, 0.349),
    ("ankley", "left"): (-0.349, 0.175),
    ("anklex", "right"): (-0.175, 0.175),
    ("anklex", "left"): (-0.175, 0.175),
}
FRIC = {"hipz": 1.35, "hipx": 1.16, "hipy": 1.31, "knee": 1.0, "ankley": 0.17, "anklex": 0.26}


def make_unique_mesh_urdf(outdir: Path) -> Path:
    """Symlink every mesh reference to a unique name (importer chokes on dupes)."""
    uniq = outdir / "uniq_assets"
    uniq.mkdir(parents=True, exist_ok=True)
    counter = [0]

    def sub(m: re.Match) -> str:
        link = uniq / f"m{counter[0]:03d}.stl"
        counter[0] += 1
        link.unlink(missing_ok=True)
        link.symlink_to(ASSET_DIR / "urdf/assets" / m.group(1))
        return f'<mesh filename="{link}"/>'

    patched = outdir / "robot_uniq.urdf"
    patched.write_text(
        re.sub(r'<mesh filename="package://assets/([^"]*)"\s*/?>', sub, URDF.read_text())
    )
    return patched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ver", default="v7")
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--episodes", type=int, default=0,
                    help="batch-eval mode: run N seeded episodes (no video), "
                         "commands sampled like the sim2sim harness")
    ap.add_argument("--sweep", action="store_true",
                    help="velocity-tracking sweep (same points as velocity_sweep.py)")
    ap.add_argument("--grid", action="store_true",
                    help="render all 50 eval episodes as a 10x5 mosaic video")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    patched = make_unique_mesh_urdf(outdir)

    from sim2sim.config import EvalCfg
    from sim2sim.control.actuation import PDController
    from sim2sim.obs.observation import ObservationBuilder
    from sim2sim.policy.onnx_policy import OnnxPolicy
    from sim2sim.sim.state import RobotState, quat_to_projected_gravity, world_velocities_to_base

    eval_cfg = EvalCfg.from_yaml(REPO / f"configs/eval_zealot_{args.ver}.yaml")
    robot_cfg = eval_cfg.load_robot()
    policy_cfg = eval_cfg.load_policy()

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": W, "height": H})

    import carb
    import omni.kit.commands
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    from PIL import Image
    from pxr import PhysxSchema, UsdLux, UsdPhysics

    enable_extension("isaacsim.asset.importer.urdf")

    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", SPP)
    s.set("/rtx/pathtracing/totalSpp", SPP)

    world = World(physics_dt=DT, rendering_dt=1.0 / FPS, stage_units_in_meters=1.0)
    GroundPlane(prim_path="/World/ground", z_position=0.0, color=np.array([0.25, 0.25, 0.28]))

    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.fix_base = False  # policy balances; external torque-PD
    cfg.import_inertia_tensor = True
    ok, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(patched), import_config=cfg,
        get_articulation_root=True,
    )
    if not ok or not prim_path:
        raise RuntimeError(f"URDF import failed (ok={ok}, prim_path={prim_path})")

    stage = omni.usd.get_context().get_stage()
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint):
            base, side = p.GetName().rsplit("_", 1)
            if base not in FRIC:
                continue
            lo, hi = LIM_SIDE.get((base, side), LIM.get(base))
            j = UsdPhysics.RevoluteJoint(p)
            j.CreateLowerLimitAttr(0.0).Set(lo * R2D)
            j.CreateUpperLimitAttr(0.0).Set(hi * R2D)
            PhysxSchema.PhysxJointAPI.Apply(p).CreateJointFrictionAttr(0.0).Set(FRIC[base])

    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(300.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(1000.0)
    sun.CreateAngleAttr(0.53)

    robot = SingleArticulation(prim_path)
    world.scene.add(robot)
    world.reset()

    # External torque-PD: zero the PhysX drives, apply efforts each step.
    n = len(robot.dof_names)
    robot.get_articulation_controller().set_gains(kps=np.zeros(n), kds=np.zeros(n))
    dof_map = [list(robot.dof_names).index(j) for j in robot_cfg.joint_names]

    robot.set_world_pose(position=np.array([0.0, 0.0, robot_cfg.base_height_init + 0.06]))
    robot.set_linear_velocity(np.zeros(3))
    robot.set_angular_velocity(np.zeros(3))
    robot.set_joint_positions(np.zeros(n))
    robot.set_joint_velocities(np.zeros(n))

    def get_state() -> RobotState:
        pos, quat = robot.get_world_pose()  # quat (w,x,y,z)
        lin = np.asarray(robot.get_linear_velocity(), dtype=np.float32)
        ang = np.asarray(robot.get_angular_velocity(), dtype=np.float32)
        quat = np.asarray(quat, dtype=np.float32)
        blin, bang = world_velocities_to_base(quat, lin, ang)
        jp = np.asarray(robot.get_joint_positions(), dtype=np.float32)[dof_map]
        jv = np.asarray(robot.get_joint_velocities(), dtype=np.float32)[dof_map]
        return RobotState(
            base_pos=np.asarray(pos, dtype=np.float32), base_quat=quat,
            base_lin_vel=blin, base_ang_vel=bang, joint_pos=jp, joint_vel=jv,
            projected_gravity=quat_to_projected_gravity(quat), sim_time=0.0,
        )

    obs_builder = ObservationBuilder(policy_cfg, robot_cfg)
    policy = OnnxPolicy(str(policy_cfg.onnx_path), obs_builder.dim,
                        clip_actions=policy_cfg.clip_actions)
    pd = PDController(robot_cfg)
    command = np.array([args.vx, 0.0, 0.0], dtype=np.float32)

    if args.sweep or args.episodes or args.grid:
        # Batch-eval: wrap the PhysX loop in the Simulator interface and reuse
        # run_episode/aggregate so the metrics match MuJoCo/Genesis exactly.
        from sim2sim.eval.metrics import aggregate
        from sim2sim.eval.runner import run_episode
        from sim2sim.obs.commands import CommandGenerator
        from sim2sim.sim.base import Simulator

        outer_n, outer_map = n, dof_map

        class IsaacInline(Simulator):
            name = "isaaclab"
            dt = DT

            def load(self, *a, **k):
                pass

            def reset(self, init=None):
                height = (init.base_height if init else robot_cfg.base_height_init) + 0.06
                quat = np.asarray(init.base_quat) if init else np.array([1.0, 0.0, 0.0, 0.0])
                jp = np.zeros(outer_n)
                if init is not None:
                    jp[outer_map] = init.joint_pos
                robot.set_world_pose(
                    position=np.array([0.0, 0.0, height]),
                    orientation=quat,  # (w,x,y,z)
                )
                robot.set_linear_velocity(
                    np.asarray(init.base_lin_vel) if init else np.zeros(3))
                robot.set_angular_velocity(
                    np.asarray(init.base_ang_vel) if init else np.zeros(3))
                robot.set_joint_positions(jp)
                robot.set_joint_velocities(np.zeros(outer_n))
                world.step(render=False)
                return get_state()

            def apply_torques(self, tau):
                t = np.zeros(outer_n, dtype=np.float32)
                t[outer_map] = np.asarray(tau, dtype=np.float32)
                robot.apply_action(ArticulationAction(joint_efforts=t))

            def step(self):
                world.step(render=False)

            def get_state(self):
                return get_state()

            def total_mass(self):
                return 12.7

        isim = IsaacInline()

        if args.grid:
            # Mirror record_grid.py: 50 seeded eval episodes tiled 10x5.
            import imageio.v2 as imageio
            from isaacsim.sensors.camera import Camera
            from isaacsim.core.utils.viewports import set_camera_view
            from sim2sim.obs.commands import CommandGenerator

            s.set("/rtx/pathtracing/spp", 16)
            s.set("/rtx/pathtracing/totalSpp", 16)
            cam = Camera(prim_path="/World/cam", resolution=(W, H))
            cam.initialize()
            set_camera_view(eye=[2.2, -2.2, 1.0], target=[0.0, 0.0, 0.45],
                            camera_prim_path="/World/cam")
            for _ in range(8):
                world.render()

            COLS, ROWS, TW, TH = 10, 5, 160, 120
            FPS_G, REVERY = 12.5, 4
            cmd_gen = CommandGenerator(policy_cfg)
            episodes = []
            for seed in eval_cfg.seeds[:COLS * ROWS]:
                rng = np.random.default_rng(seed)
                command = cmd_gen.sample(rng)
                policy.reset()
                obs_builder.reset()
                state = isim.reset()
                frames, fell_frame = [], None
                for step_i in range(eval_cfg.max_steps):
                    obs = obs_builder.build(state, command)
                    action = policy.act(obs)
                    obs_builder.set_last_action(action)
                    for _ in range(max(1, round(eval_cfg.control_dt / DT))):
                        tau = pd.compute_torque(action, state.joint_pos, state.joint_vel)
                        isim.apply_torques(tau)
                        isim.step()
                        state = isim.get_state()
                    if step_i % REVERY == 0:
                        world.render()
                        rgba = np.asarray(cam.get_rgba())
                        frames.append(np.ascontiguousarray(
                            rgba[::4, ::4, :3].astype(np.uint8)))
                    fell = (state.base_pos[2] < eval_cfg.fall_height
                            or state.projected_gravity[2] > -eval_cfg.fall_tilt)
                    if fell and fell_frame is None:
                        fell_frame = len(frames)
                    if fell_frame is not None and len(frames) - fell_frame > int(0.5 * FPS_G):
                        break
                episodes.append(frames)
                print(f"seed {seed}: {len(frames)} frames "
                      f"{'(survived)' if fell_frame is None else ''}", flush=True)

            n_frames = max(len(f) for f in episodes)
            out_path = outdir / f"grid_{args.ver}_isaac.mp4"
            writer = imageio.get_writer(str(out_path), fps=FPS_G, macro_block_size=8)
            for i in range(n_frames):
                canvas = np.zeros((ROWS * TH, COLS * TW, 3), dtype=np.uint8)
                for e, frames in enumerate(episodes):
                    r, c = divmod(e, COLS)
                    tile = frames[i] if i < len(frames) else (frames[-1] * 0.35).astype(np.uint8)
                    canvas[r*TH:(r+1)*TH, c*TW:(c+1)*TW] = tile[:TH, :TW]
                writer.append_data(canvas)
            writer.close()
            print(f"[wrote] {out_path} ({n_frames} frames)")
            app.close()
            return

        if args.sweep:
            import importlib.util as _ilu

            spec = _ilu.spec_from_file_location(
                "velocity_sweep", Path(__file__).parent / "velocity_sweep.py"
            )
            vs = _ilu.module_from_spec(spec)
            spec.loader.exec_module(vs)
            points = [[v, 0.0, 0.0] for v in vs.VX_SWEEP]
            points += [[0.2, 0.0, w] for w in vs.YAW_SWEEP]
            rows = [
                vs.run_point(isim, policy, robot_cfg, policy_cfg, eval_cfg, c)
                for c in points
            ]
            import json

            (outdir / f"isaac_sweep_{args.ver}.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(rows, indent=2))
            app.close()
            return

        cmd_gen = CommandGenerator(policy_cfg)
        if os.environ.get("ISAAC_DEBUG"):
            st = isim.reset()
            print("reset: z", float(st.base_pos[2]), "pg", st.projected_gravity,
                  "|jp|", float(np.abs(st.joint_pos).max()), flush=True)
            rng = np.random.default_rng(0)
            cmd = cmd_gen.sample(rng)
            obs_builder.reset()
            policy.reset()
            print("dbg cmd:", cmd, flush=True)
            for i in range(8):
                obs = obs_builder.build(st, cmd)
                action = policy.act(obs)
                obs_builder.set_last_action(action)
                print(f"dbg ctrl {i}: |obs|max={float(np.abs(obs).max()):.2f} "
                      f"act[:4]={np.round(action[:4],2)} "
                      f"z={float(st.base_pos[2]):.3f}", flush=True)
                for _ in range(4):
                    tau = pd.compute_torque(action, st.joint_pos, st.joint_vel)
                    isim.apply_torques(tau)
                    isim.step()
                    st = isim.get_state()
        eps = [
            run_episode(isim, policy, robot_cfg, obs_builder, cmd_gen, eval_cfg, seed=s)
            for s in range(args.episodes)
        ]
        agg = aggregate(eps)
        import json

        (outdir / "isaac_eval.json").write_text(
            json.dumps({"policy": args.ver, "episodes": args.episodes, "metrics": agg}, indent=2)
        )
        print(json.dumps(agg, indent=2))
        app.close()
        return

    cam = Camera(prim_path="/World/cam", resolution=(W, H))
    cam.initialize()
    set_camera_view(eye=[2.2, -2.2, 1.0], target=[0.0, 0.0, 0.45],
                    camera_prim_path="/World/cam")
    for _ in range(8):
        world.render()

    obs_builder.reset()
    policy.reset()
    state = get_state()
    decimation = max(1, round(CONTROL_DT / DT))
    steps = int(args.duration / CONTROL_DT)
    render_every = max(1, round(1.0 / (FPS * CONTROL_DT)))

    fell_at = None
    frame_i = 0
    x0 = float(state.base_pos[0])
    tau_isaac = np.zeros(n, dtype=np.float32)
    for step in range(steps):
        obs = obs_builder.build(state, command)
        action = policy.act(obs)
        obs_builder.set_last_action(action)
        for _ in range(decimation):
            tau = pd.compute_torque(action, state.joint_pos, state.joint_vel)
            tau_isaac[dof_map] = tau
            robot.apply_action(ArticulationAction(joint_efforts=tau_isaac.copy()))
            world.step(render=False)
            state = get_state()
        if fell_at is None and (
            state.base_pos[2] < eval_cfg.fall_height
            or state.projected_gravity[2] > -eval_cfg.fall_tilt
        ):
            fell_at = step * CONTROL_DT
        if not args.no_video and step % render_every == 0:
            world.render()
            rgba = cam.get_rgba()
            Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(
                outdir / f"f_{frame_i:03d}.png"
            )
            frame_i += 1

    dist = float(state.base_pos[0]) - x0
    msg = (
        f"policy {args.ver}  cmd vx={args.vx}\n"
        f"survival: {fell_at if fell_at is not None else args.duration:.2f} s"
        f" (cap {args.duration} s)\n"
        f"final base z {float(state.base_pos[2]):.3f}, forward distance {dist:.3f} m\n"
        f"frames written: {frame_i}\n"
    )
    (outdir / "result.txt").write_text(msg)
    print(msg)
    app.close()


if __name__ == "__main__":
    main()

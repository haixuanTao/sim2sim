"""Time-to-first-step per backend — the dev-loop "boot" cost.

Measures wall time from process start until one physics step of the LeRobot
scene has completed (single env, warm JIT caches — what you pay on every
config change / script rerun, not the first-ever install). Three segments:

  imports     heavy module imports (isaac/genesis/warp pay most here)
  build       engine init + scene construction (+ JIT/kernel compile it forces)
  first_step  first stepped frame (deferred compiles land here; for the
              CUDA-graph variant this includes warmup + graph capture)

Run:  python examples/lerobot_legs/boot_bench.py --sim genesis
"""

from __future__ import annotations

import time

T0 = time.perf_counter()  # before the heavy imports below

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

MJCF_DIR = (
    Path.home()
    / "Documents/work/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms/mjcf"
)
SCENE_XML = MJCF_DIR / "sim_scene_safe.xml"
ROBOT_XML = MJCF_DIR / "robot.xml"


def report(tag: str, imports: float, init: float, build: float, first_step: float) -> None:
    total = time.perf_counter() - T0
    # flush: Isaac's app.close() hard-exits and discards buffered stdout.
    print(f"[boot] {tag}: imports={imports:.2f}s init={init:.2f}s build={build:.2f}s "
          f"first_step={first_step:.2f}s total={total:.2f}s", flush=True)


def boot_mujoco() -> None:
    import mujoco
    t_imp = time.perf_counter() - T0
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    t_build = time.perf_counter() - T0 - t_imp
    mujoco.mj_step(model, data)
    report("mujoco", t_imp, 0.0, t_build, time.perf_counter() - T0 - t_imp - t_build)


def boot_mjlab() -> None:
    import mujoco
    import torch
    from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
    t_imp = time.perf_counter() - T0
    import warp as wp
    wp.init()  # CUDA context + warp runtime — the framework "boot"
    t_init = time.perf_counter() - T0 - t_imp
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    cfg = SimulationCfg(mujoco=MujocoCfg(timestep=model.opt.timestep,
                                         gravity=(0.0, 0.0, -9.81)))
    sim = Simulation(num_envs=1, cfg=cfg, model=model, device="cuda:0")
    sim.reset()
    sim.forward()
    t_build = time.perf_counter() - T0 - t_imp - t_init
    sim.step()
    torch.cuda.synchronize()
    report("mjlab", t_imp, t_init, t_build,
           time.perf_counter() - T0 - t_imp - t_init - t_build)


def boot_genesis() -> None:
    import genesis as gs
    t_imp = time.perf_counter() - T0
    gs.init(backend=gs.gpu)
    t_init = time.perf_counter() - T0 - t_imp
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.005), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(ROBOT_XML), pos=(0.0, 0.0, 0.72)))
    scene.build()
    t_build = time.perf_counter() - T0 - t_imp - t_init
    scene.step()
    _ = robot.get_pos()  # device sync
    report("genesis", t_imp, t_init, t_build,
           time.perf_counter() - T0 - t_imp - t_init - t_build)


def boot_nexus(cuda_graph: bool) -> None:
    import nexus3d as nx
    t_imp = time.perf_counter() - T0
    viewer = nx.NexusViewer(64, 64, headless=True)
    viewer.set_draw_ui(False)
    if cuda_graph:
        viewer.with_cuda()
    viewer.init_backend()
    pipeline = nx.NexusPipeline()
    pipeline.preload_pipelines(viewer)
    t_init = time.perf_counter() - T0 - t_imp
    viewer.set_up_axis(nx.Vec3.Z)
    state = nx.NexusState()
    info = state.insert_mjcf(viewer, str(ROBOT_XML), render_colliders=False)
    assert info.loaded
    state.finalize(viewer)
    state.set_rbd_gravity(viewer, nx.Vec3(0.0, 0.0, -9.81))
    state.set_rbd_steps_per_frame(1)
    t_build = time.perf_counter() - T0 - t_imp - t_init
    if cuda_graph:
        for _ in range(5):  # warmup required before capture
            pipeline.simulate(viewer, state, None)
            viewer.sync(state, None)
        assert pipeline.capture_cuda_graph(viewer, state), "graph capture failed"
        pipeline.replay_cuda_graph()
    else:
        pipeline.simulate(viewer, state, None)
    viewer.sync(state, None)
    report("nexus-cuda-graph" if cuda_graph else "nexus",
           t_imp, t_init, t_build,
           time.perf_counter() - T0 - t_imp - t_init - t_build)


def boot_isaac() -> None:
    """Isaac Lab via AppLauncher (headless kit experience — the bare
    SimulationApp default experience crashes on driver 595). Run with the
    Isaac Lab venv python (~/isaaclab/.venv). Scene mirrors bench_isaac in
    batch_bench.py (WBC-AGILE LeRobot no-arms URDF, implicit PD)."""
    from isaaclab.app import AppLauncher
    t_imp = time.perf_counter() - T0
    app = AppLauncher(headless=True).app
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    t_init = time.perf_counter() - T0 - t_imp

    urdf = str(Path.home()
               / "WBC-AGILE/agile/rl_env/assets/robots/lerobot_humanoid_no_arms_new.urdf")
    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf, fix_base=False, merge_fixed_joints=False,
            root_link_name="torso_subassembly",
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0)),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.72), rot=(0.9962, 0.0, -0.0872, 0.0)),
        actuators={"legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={".*hipz.*": 30, ".*hipx.*": 40, ".*hipy.*": 60,
                       ".*knee.*": 60, ".*ankley.*": 20, ".*anklex.*": 20},
            damping={".*hipz.*": 3, ".*hipx.*": 3, ".*hipy.*": 4,
                     ".*knee.*": 4, ".*ankley.*": 1.5, ".*anklex.*": 1.5})},
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground",
                              spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = robot_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    t_build = time.perf_counter() - T0 - t_imp - t_init
    sim.step(render=False)
    torch.cuda.synchronize()
    report("isaac", t_imp, t_init, t_build,
           time.perf_counter() - T0 - t_imp - t_init - t_build)
    app.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True,
                    choices=["mujoco", "mjlab", "genesis", "isaac",
                             "nexus", "nexus_cuda_graph"])
    args = ap.parse_args()
    if args.sim == "mujoco":
        boot_mujoco()
    elif args.sim == "mjlab":
        boot_mjlab()
    elif args.sim == "genesis":
        boot_genesis()
    elif args.sim == "isaac":
        boot_isaac()
    else:
        boot_nexus(cuda_graph=args.sim == "nexus_cuda_graph")


if __name__ == "__main__":
    main()

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
    print(f"[boot] {tag}: imports={imports:.2f}s init={init:.2f}s build={build:.2f}s "
          f"first_step={first_step:.2f}s total={total:.2f}s")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True,
                    choices=["mujoco", "mjlab", "genesis", "nexus", "nexus_cuda_graph"])
    args = ap.parse_args()
    if args.sim == "mujoco":
        boot_mujoco()
    elif args.sim == "mjlab":
        boot_mjlab()
    elif args.sim == "genesis":
        boot_genesis()
    else:
        boot_nexus(cuda_graph=args.sim == "nexus_cuda_graph")


if __name__ == "__main__":
    main()

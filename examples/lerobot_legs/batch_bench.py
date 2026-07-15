"""Batch physics throughput — LeRobot legs × N parallel envs (no rendering).

The single-env scenes on the benchmark page measure per-frame render/readback
overhead; this measures what the GPU engines are actually built for: stepping
thousands of envs at once. Same real LeRobot bipedal-platform asset as the
render demos. No camera — pure physics env-steps/s, timed over a steady-state
window with one device synchronize at the end (standard batch methodology).

Per backend:
  mujoco      threaded ``mujoco.rollout`` (one MjData per thread), CPU
  mjlab       MuJoCo-Warp ``Simulation(num_envs=N)``
  genesis     ``scene.build(n_envs=N)`` + per-DOF PD hold
  nexus / nexus_cuda_graph
              one physics world with N grid-offset robots (visual geoms
              stripped; same passive setup as nexus_render.py), WebGPU or
              native CUDA with whole-step CUDA-graph replay

Run:  python examples/lerobot_legs/batch_bench.py --sim genesis --envs 2048
"""

from __future__ import annotations

import argparse
import math
import re  # noqa: F401 (kept for grid-name tweaks)
import tempfile
import time
from pathlib import Path

import numpy as np

MJCF_DIR = (
    Path.home()
    / "Documents/work/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms/mjcf"
)
SCENE_XML = MJCF_DIR / "sim_scene_safe.xml"  # dt 5 ms, position servos
ROBOT_XML = MJCF_DIR / "robot.xml"

# (joint suffix -> kp, kv) copied from sim_scene_safe.xml <actuator> — same as
# genesis_render.py.
GAINS = {
    "hipz": (30, 3), "hipx": (40, 3), "hipy": (60, 4),
    "knee": (60, 4), "ankley": (20, 1.5), "anklex": (20, 1.5),
}


def report(tag: str, envs: int, steps: int, secs: float, dt: float) -> None:
    rate = envs * steps / secs
    # flush: Isaac's app.close() hard-exits the process and would discard
    # buffered stdout, silently eating the result line.
    print(f"[batch] {tag}: envs={envs} steps={steps} in {secs:.2f}s "
          f"= {rate:,.0f} env-steps/s (dt={1e3 * dt:g}ms)", flush=True)


def bench_mujoco(envs: int, steps: int) -> None:
    import mujoco
    from mujoco import rollout

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d0 = mujoco.MjData(model)
    d0.qpos[2] = 0.72
    mujoco.mj_forward(model, d0)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state0 = np.empty(nstate)
    mujoco.mj_getState(model, d0, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial = np.tile(state0, (envs, 1))

    import os
    nthread = min(os.cpu_count() or 1, envs)
    datas = [mujoco.MjData(model) for _ in range(nthread)]
    rollout.rollout(model, datas, initial[:nthread], nstep=5)  # thread warmup

    t0 = time.perf_counter()
    rollout.rollout(model, datas, initial, nstep=steps)
    report(f"mujoco({nthread}t)", envs, steps, time.perf_counter() - t0,
           model.opt.timestep)


def bench_mjlab(envs: int, steps: int) -> None:
    import mujoco
    from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    dt = model.opt.timestep
    cfg = SimulationCfg(mujoco=MujocoCfg(timestep=dt, gravity=(0.0, 0.0, -9.81)))
    sim = Simulation(num_envs=envs, cfg=cfg, model=model, device="cuda:0")
    sim.reset()
    sim.forward()
    for _ in range(10):  # warp module compile + graph warmup
        sim.step()
    import torch
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    torch.cuda.synchronize()
    report("mjlab", envs, steps, time.perf_counter() - t0, dt)


def bench_genesis(envs: int, steps: int) -> None:
    import genesis as gs

    gs.init(backend=gs.gpu)
    dt = 0.005
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(ROBOT_XML), pos=(0.0, 0.0, 0.72)))
    scene.build(n_envs=envs)

    dofs, kps, kvs, targets = [], [], [], []
    for side in ("left", "right"):
        for jname, (kp, kv) in GAINS.items():
            joint = robot.get_joint(f"{jname}_{side}")
            dofs.append(joint.dof_idx_local)
            kps.append(kp)
            kvs.append(kv)
            bias = 0.25 if side == "right" else -0.25
            targets.append(bias if jname == "hipy" else 0.0)
    robot.set_dofs_kp(np.array(kps, dtype=np.float32), dofs)
    robot.set_dofs_kv(np.array(kvs, dtype=np.float32), dofs)
    target = np.tile(np.array(targets, dtype=np.float32), (envs, 1))

    for _ in range(5):  # taichi JIT warmup
        robot.control_dofs_position(target, dofs)
        scene.step()
    _ = robot.get_pos()  # device sync

    t0 = time.perf_counter()
    for _ in range(steps):
        scene.step()
    _ = robot.get_pos()
    report("genesis", envs, steps, time.perf_counter() - t0, dt)


def bench_isaac(envs: int, steps: int) -> None:
    """Isaac Lab / PhysX 5 via the WBC-AGILE manager-based LeRobot env.

    Drives Velocity-LeRobot-NoArms-v0 (the same task zealot's README benchmarks
    against) with zero actions — env.step = physics decimation + observation/
    reward managers, so this row includes manager overhead the other engines'
    physics-only rows don't have. Needs ~/WBC-AGILE on PYTHONPATH and the
    Isaac Lab venv python (~/isaaclab/.venv). A bare SimulationContext loop
    hangs after the first step on this driver; the manager env path is the
    configuration that verifiably works (it's how the training benchmarks ran).
    """
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app

    import gymnasium as gym
    import torch
    import agile.rl_env.tasks  # noqa: F401  (registers the LeRobot tasks)
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("Velocity-LeRobot-NoArms-v0", device="cuda:0",
                            num_envs=envs)
    # Kill every debug-vis marker: the velocity-arrow point instancers go
    # through the RTX visualization path that crashes on driver 595 (the same
    # class of crash as the bare SimulationApp).
    for grp in ("commands", "observations", "events", "rewards"):
        cfg_grp = getattr(env_cfg, grp, None)
        for name in dir(cfg_grp):
            term = getattr(cfg_grp, name, None)
            if hasattr(term, "debug_vis"):
                term.debug_vis = False
    t_setup = time.perf_counter()
    env = gym.make("Velocity-LeRobot-NoArms-v0", cfg=env_cfg)
    env.reset()
    print(f"[setup] {envs} envs built ({time.perf_counter() - t_setup:.1f}s)", flush=True)

    act = torch.zeros(envs, env.unwrapped.action_manager.total_action_dim,
                      device=env.unwrapped.device)
    # env.step advances `decimation` physics steps per call; count real steps.
    decimation = env.unwrapped.cfg.decimation
    dt = env.unwrapped.cfg.sim.dt
    for _ in range(10):
        env.step(act)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(act)
    torch.cuda.synchronize()
    report("isaac", envs, steps * decimation, time.perf_counter() - t0, dt)
    env.close()
    app.close()


def _grid_xml(envs: int) -> str:
    """One MJCF with `envs` copies of the robot body tree on a 2 m grid.

    All names are suffixed per instance (disjoint trees -> one rapier multibody
    per robot); visual geoms (contype=0 STL meshes) are stripped — physics is
    unchanged and the parse drops the 200+ visual hulls per robot. Written next
    to robot.xml so `meshdir="assets"` keeps resolving (a tempdir would
    silently drop every mesh collider). One insert_mjcf call also means ONE
    auto-floor instead of `envs` coincident slabs under every foot.
    """
    import copy
    import xml.etree.ElementTree as ET

    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    world = root.find("worldbody")
    robot = world.find("body")
    world.remove(robot)
    for actu in root.findall("actuator"):
        root.remove(actu)

    for geom in robot.iter("body"):
        for g in list(geom):
            if g.tag == "geom" and g.get("class") == "visual":
                geom.remove(g)
    for g in list(robot):
        if g.tag == "geom" and g.get("class") == "visual":
            robot.remove(g)

    side = math.ceil(math.sqrt(envs))
    half = side  # 2 m spacing
    for i in range(envs):
        inst = copy.deepcopy(robot)
        inst.set("pos", f"{(i % side) * 2.0 - half} {(i // side) * 2.0 - half} 0.72")
        for el in inst.iter():
            if el.get("name"):
                el.set("name", f"{el.get('name')}_e{i}")
        world.append(inst)
    return ET.tostring(root, encoding="unicode")


def bench_nexus(envs: int, steps: int, cuda_graph: bool) -> None:
    import nexus3d as nx

    viewer = nx.NexusViewer(64, 64, headless=True)
    viewer.set_draw_ui(False)
    if cuda_graph:
        viewer.with_cuda()
    viewer.init_backend()
    pipeline = nx.NexusPipeline()
    pipeline.preload_pipelines(viewer)
    viewer.set_up_axis(nx.Vec3.Z)

    state = nx.NexusState()
    # No extra bodies: batched finalize requires identical collider counts in
    # every env. Queue drains go through viewer.sync (blocking state readback).

    t_setup = time.perf_counter()
    # Preferred layout: one robot per environment (the batched design the GPU
    # solver is built around — constraint buffers scale linearly with envs).
    # Needs the `env=` kwarg on insert_mjcf (per-env bindings); falls back to
    # the all-robots-in-env-0 mega-XML, which is quadratic in robot count and
    # caps out near 96 robots (wgpu 128 MiB binding limit).
    # Physics-only copy of robot.xml (visual STL geoms stripped): every insert
    # re-parses the referenced meshes, and the ~200 visual hulls are discarded
    # anyway (batch envs render nothing). Written next to robot.xml so
    # meshdir="assets" resolves. Cuts per-env insert cost ~10x.
    novis = ROBOT_XML.parent / "_robot_novis.xml"
    novis.write_text(_grid_xml(1))
    per_env = True
    try:
        info = state.insert_mjcf(viewer, str(novis), render_colliders=False, env=0)
        assert info.loaded, "env 0 insert failed"
    except TypeError:
        per_env = False
    if per_env:
        try:
            for i in range(1, envs):
                state.add_environment()
                info = state.insert_mjcf(viewer, str(novis), render_colliders=False, env=i)
                assert info.loaded, f"env {i} insert failed"
                if i % 256 == 0:
                    print(f"[setup] env {i}/{envs} ({time.perf_counter() - t_setup:.1f}s)",
                          flush=True)
        finally:
            novis.unlink(missing_ok=True)
        print(f"[setup] {envs} robots, one per env ({time.perf_counter() - t_setup:.1f}s)",
              flush=True)
        # Default is 4096 pairs/env — sized for one busy scene, not thousands
        # of small envs (pair workspaces bind capacity x envs x ~1.1 KB ≈ 9 GiB
        # at 2048). This scene peaks at ~7 pairs/env; 64 is ample headroom and
        # the Grow resize policy recovers if a scene ever exceeds it.
        if hasattr(state, "set_rbd_collisions_capacity"):
            state.set_rbd_collisions_capacity(64)
    else:
        novis.unlink(missing_ok=True)
        inst = ROBOT_XML.parent / f"_batch_grid_{envs}.xml"
        try:
            inst.write_text(_grid_xml(envs))
            print(f"[setup] grid XML written ({time.perf_counter() - t_setup:.1f}s)", flush=True)
            info = state.insert_mjcf(viewer, str(inst), render_colliders=False)
            assert info.loaded, "grid insert failed"
        finally:
            inst.unlink(missing_ok=True)
        print(f"[setup] {envs} robots in env 0 ({time.perf_counter() - t_setup:.1f}s)", flush=True)
    state.finalize(viewer)
    state.set_rbd_gravity(viewer, nx.Vec3(0.0, 0.0, -9.81))
    state.set_rbd_steps_per_frame(1)

    graphed = False
    for _ in range(5):
        pipeline.simulate(viewer, state, None)
        viewer.sync(state, None)
    if cuda_graph:
        graphed = pipeline.capture_cuda_graph(viewer, state)
        assert graphed, "CUDA graph capture failed"
        for _ in range(3):
            pipeline.replay_cuda_graph()
    viewer.sync(state, None)  # drain queue before the timed window

    dt = 1.0 / 60.0  # one solver step per simulate()
    t0 = time.perf_counter()
    for _ in range(steps):
        if graphed:
            pipeline.replay_cuda_graph()
        else:
            pipeline.simulate(viewer, state, None)
    viewer.sync(state, None)  # device sync (blocking link-state readback)
    report(("nexus-cuda-graph" if graphed else "nexus") + ("-perenv" if per_env else ""), envs, steps,
           time.perf_counter() - t0, dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True,
                    choices=["mujoco", "mjlab", "genesis", "isaac",
                             "nexus", "nexus_cuda_graph"])
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    if args.sim == "mujoco":
        bench_mujoco(args.envs, args.steps)
    elif args.sim == "mjlab":
        bench_mjlab(args.envs, args.steps)
    elif args.sim == "genesis":
        bench_genesis(args.envs, args.steps)
    elif args.sim == "isaac":
        bench_isaac(args.envs, args.steps)
    else:
        bench_nexus(args.envs, args.steps, cuda_graph=args.sim == "nexus_cuda_graph")


if __name__ == "__main__":
    main()

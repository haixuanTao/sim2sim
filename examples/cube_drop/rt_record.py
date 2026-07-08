"""Record a cube-drop *trajectory* (pose per frame) from one physics backend.

Ray tracing isn't a native feature of these simulators, so we follow the
sim2sim philosophy and separate physics from rendering: each backend produces
the cube's world pose over time (+ its physics timing), and a shared path tracer
(rt_render.py) renders every trajectory identically. This keeps the render
apples-to-apples across sims and isolates the per-backend *physics* cost.

Run:  python examples/cube_drop/rt_record.py --sim mujoco --out traj_mujoco.npz
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

N_FRAMES = 150
FRAME_DT = 1.0 / 30.0  # trajectory sample rate (each sim substeps to match) -> 5 s @ 30 fps
HALF = 0.15
START_Z = 1.5
EULER_DEG = (12.0, 22.0, 5.0)  # initial tilt so it tumbles + lands on an edge


def euler_to_quat_wxyz(deg) -> np.ndarray:
    r, p, y = (math.radians(d) for d in deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


Q0 = euler_to_quat_wxyz(EULER_DEG)

_MJCF = f"""
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="cube" pos="0 0 {START_Z}" quat="{Q0[0]} {Q0[1]} {Q0[2]} {Q0[3]}">
      <freejoint/>
      <geom name="box" type="box" size="{HALF} {HALF} {HALF}"/>
    </body>
  </worldbody>
</mujoco>
"""


def _record_mujoco():
    import mujoco

    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    spf = max(1, round(FRAME_DT / model.opt.timestep))
    pos, quat = [], []
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        for _ in range(spf):
            mujoco.mj_step(model, data)
        pos.append(data.qpos[0:3].copy())
        quat.append(data.qpos[3:7].copy())
    return np.array(pos), np.array(quat), time.perf_counter() - t0, spf * N_FRAMES


def _record_pybullet():
    import pybullet as p
    import pybullet_data

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    dt = 1.0 / 240.0
    p.setTimeStep(dt)
    p.loadURDF("plane.urdf")
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[HALF] * 3)
    orn = [Q0[1], Q0[2], Q0[3], Q0[0]]  # wxyz -> xyzw
    cube = p.createMultiBody(1.0, col, basePosition=[0, 0, START_Z], baseOrientation=orn)
    spf = max(1, round(FRAME_DT / dt))
    pos, quat = [], []
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        for _ in range(spf):
            p.stepSimulation()
        xyz, o = p.getBasePositionAndOrientation(cube)
        pos.append(xyz)
        quat.append([o[3], o[0], o[1], o[2]])  # xyzw -> wxyz
    dt_wall = time.perf_counter() - t0
    p.disconnect()
    return np.array(pos), np.array(quat), dt_wall, spf * N_FRAMES


def _record_genesis():
    import genesis as gs

    try:
        gs.init(backend=gs.gpu)
    except Exception:
        gs.init(backend=gs.cpu)
    dt = 1.0 / 120.0
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(size=(2 * HALF, 2 * HALF, 2 * HALF), pos=(0, 0, START_Z), euler=EULER_DEG)
    )
    scene.build(n_envs=1)
    spf = max(1, round(FRAME_DT / dt))

    def _np(x):
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return np.atleast_2d(a)[0]

    pos, quat = [], []
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        for _ in range(spf):
            scene.step()
        pos.append(_np(cube.get_pos()))
        quat.append(_np(cube.get_quat()))  # wxyz
    return np.array(pos), np.array(quat), time.perf_counter() - t0, spf * N_FRAMES


def _record_mjlab():
    import mujoco
    from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

    model = mujoco.MjModel.from_xml_string(_MJCF)
    cfg = SimulationCfg(mujoco=MujocoCfg(timestep=0.005, gravity=(0.0, 0.0, -9.81)))
    sim = Simulation(num_envs=1, cfg=cfg, model=model, device="cuda:0")
    sim.reset()
    sim.forward()
    spf = max(1, round(FRAME_DT / 0.005))

    def _row0(a):
        r = a[0]
        return r.detach().cpu().numpy() if hasattr(r, "detach") else np.asarray(r)

    pos, quat = [], []
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        for _ in range(spf):
            sim.step()
        q = _row0(sim.data.qpos)
        pos.append(q[0:3])
        quat.append(q[3:7])  # wxyz
    return np.array(pos), np.array(quat), time.perf_counter() - t0, spf * N_FRAMES


def _record_isaac():
    import os

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api.objects.ground_plane import GroundPlane

    dt = 1.0 / 240.0
    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
    GroundPlane(prim_path="/World/ground", z_position=0.0)
    cube = DynamicCuboid(
        prim_path="/World/cube",
        position=np.array([0.0, 0.0, START_Z]),
        orientation=Q0,  # wxyz
        size=2 * HALF,
        mass=1.0,
    )
    world.scene.add(cube)
    world.reset()
    spf = max(1, round(FRAME_DT / dt))
    pos, quat = [], []
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        for _ in range(spf):
            world.step(render=False)
        p, q = cube.get_world_pose()
        pos.append(np.asarray(p, dtype=np.float64))
        quat.append(np.asarray(q, dtype=np.float64))  # wxyz
    dt_wall = time.perf_counter() - t0
    global _isaac_app  # close() exits the process, so defer it until after savez
    _isaac_app = app
    return np.array(pos), np.array(quat), dt_wall, spf * N_FRAMES


_RECORDERS = {
    "mujoco": _record_mujoco,
    "pybullet": _record_pybullet,
    "genesis": _record_genesis,
    "mjlab": _record_mjlab,
    "isaac": _record_isaac,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, choices=list(_RECORDERS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pos, quat, phys_s, n_steps = _RECORDERS[args.sim]()
    np.savez(
        args.out,
        pos=pos,
        quat=quat,
        phys_s=phys_s,
        n_steps=n_steps,
        n_frames=len(pos),
        half=HALF,
    )
    step_fps = n_steps / phys_s
    print(
        f"[phys] {args.sim}: {n_steps} physics steps in {phys_s:.3f}s = "
        f"{step_fps:,.0f} steps/s  ({len(pos)} frames)"
    )
    if "_isaac_app" in globals():
        _isaac_app.close()


if __name__ == "__main__":
    main()

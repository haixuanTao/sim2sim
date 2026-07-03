//! Rapier (Rust) physics backend for the sim2sim locomotion harness, exposed to
//! Python via PyO3 as the module `sim2sim_rapier`.
//!
//! The Rust side is deliberately dumb: it loads the URDF, advances the physics,
//! accepts raw joint torques, and reports *native* state (world-frame velocities,
//! `xyzw` quaternion). All the cross-simulator normalisation — base-frame
//! velocity rotation, projected gravity, canonical joint ordering — is done on
//! the Python side by `RapierSimulator`, using the exact same shared helpers as
//! every other backend. That keeps this file free of framework conventions.
//!
//! Robot model: a reduced-coordinate `Multibody` (rapier's articulated
//! representation), which mirrors how robotics stacks think about joints and
//! gives clean per-joint position/velocity readouts. Joint torque control is
//! applied as a world-space torque about each revolute axis on the child body;
//! rapier's articulated solver projects that into the correct generalised force
//! (multibody.rs gathers `rb.forces.torque` into the reduced coordinates).
//!
//! rapier 0.33 uses glam math types: `Vector` = `glam::Vec3`, `Rotation` =
//! `glam::Quat` (fields `.x/.y/.z/.w`), `Pose` = position + rotation.
//!
//! NOTE (torque-control fidelity): the torque->axis mapping and the joint-angle
//! sign convention are the pieces to validate numerically against the MuJoCo
//! reference backend before trusting the numbers. See the README.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rapier3d::na::Matrix3;
use rapier3d::prelude::*;
use rapier3d_urdf::{UrdfLoaderOptions, UrdfMultibodyOptions, UrdfRobot};

/// Per-joint bookkeeping in canonical order (the order Python asks for).
struct JointRef {
    /// Child rigid body — receives +tau about the joint axis.
    child: RigidBodyHandle,
    /// Parent rigid body — receives -tau about the joint axis. Applying the
    /// equal-and-opposite pair makes the actuation a *pure internal* joint
    /// torque (zero net wrench), so it drives only the joint coordinate and
    /// does not push the floating base — applying it to the child alone would
    /// project onto the base DOFs and launch the robot.
    parent: RigidBodyHandle,
    /// Link id of this joint inside the multibody (child link).
    link_id: usize,
    /// Rotation of the joint's frame2 relative to the child body. The revolute
    /// free axis is X in that frame (rapier's URDF loader convention), so the
    /// world axis is `child_rotation * frame2_rot * X`.
    frame2_rot: Rotation,
    /// Default (nominal standing) angle for this joint, set on reset().
    default_angle: Real,
}

#[pyclass(unsendable)]
pub struct RapierSim {
    // --- physics world ---
    gravity: Vector,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,

    // --- robot ---
    /// Handle to any joint of the (single, connected) robot multibody, used to
    /// reach the `Multibody` via the joint set.
    mb_handle: Option<MultibodyJointHandle>,
    root: Option<RigidBodyHandle>,
    joints: Vec<JointRef>,
    /// Total robot mass (kg), summed from the URDF <inertial> masses at load —
    /// matching PyBullet's URDF_USE_INERTIA_FROM_FILE. Read straight from the
    /// URDF because rapier stores imported masses as *additional* mass props,
    /// which RigidBody::mass() (collider-derived only) does not report.
    mass_total: Real,
    base_height: Real,
    /// Last torques requested by Python, re-applied every physics step so the
    /// actuation persists across sub-steps (matching PyBullet TORQUE_CONTROL).
    tau: Vec<Real>,
    dt_val: Real,
}

#[pymethods]
impl RapierSim {
    #[new]
    fn new() -> Self {
        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = 0.005;
        RapierSim {
            gravity: Vector::new(0.0, 0.0, -9.81),
            integration_parameters,
            physics_pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            mb_handle: None,
            root: None,
            joints: Vec::new(),
            mass_total: 0.0,
            base_height: 0.4,
            tau: Vec::new(),
            dt_val: 0.005,
        }
    }

    #[getter]
    fn dt(&self) -> f64 {
        self.dt_val as f64
    }

    /// Load the URDF and wire up the canonical joint order.
    ///
    /// `joint_names` is the canonical order from `RobotCfg`; every readout and
    /// every torque is indexed in this order so the policy sees a consistent
    /// layout across backends.
    /// `armature` (reflected motor inertia, kg·m²) and `joint_damping` (N·m·s/rad)
    /// bring the reduced-coordinate model in line with the MuJoCo reference,
    /// whose MJCF declares `armature`/`damping` on every joint. `rapier3d-urdf`
    /// ignores URDF `<dynamics>`, so without these the rapier joints have ~10x
    /// less effective inertia and no damping — the same PD gains then go stiff
    /// and unstable. Pass 0.0 for both to get the raw URDF dynamics.
    #[pyo3(signature = (urdf_path, joint_names, base_height, dt, armature=0.0, joint_damping=0.0))]
    fn load(
        &mut self,
        urdf_path: &str,
        joint_names: Vec<String>,
        base_height: f64,
        dt: f64,
        armature: f64,
        joint_damping: f64,
    ) -> PyResult<()> {
        self.base_height = base_height as Real;
        self.dt_val = dt as Real;
        self.integration_parameters.dt = dt as Real;

        // Static ground: a thin, wide box whose top surface sits at z = 0.
        let ground = ColliderBuilder::cuboid(50.0, 50.0, 0.1)
            .translation(Vector::new(0.0, 0.0, -0.1))
            .build();
        self.colliders.insert(ground);

        // Colliders come from <collision> shapes; visuals are ignored. Lift the
        // whole robot to the nominal base height so it settles onto the ground.
        let options = UrdfLoaderOptions {
            create_colliders_from_collision_shapes: true,
            create_colliders_from_visual_shapes: false,
            make_roots_fixed: false,
            shift: Pose::from_parts(
                Vector::new(0.0, 0.0, self.base_height),
                Rotation::IDENTITY,
            ),
            ..Default::default()
        };

        let (robot, urdf) = UrdfRobot::from_file(urdf_path, options, None)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to load URDF: {e}")))?;

        // Map URDF joint name -> its index in the loader's joint list.
        let mut name_to_urdf_idx = std::collections::HashMap::new();
        for (i, j) in urdf.joints.iter().enumerate() {
            name_to_urdf_idx.insert(j.name.clone(), i);
        }
        // Capture frame2 rotations before `robot` is consumed by insertion.
        let frame2_rots: Vec<Rotation> = robot
            .joints
            .iter()
            .map(|j| j.joint.local_frame2.rotation)
            .collect();

        let handles = robot.insert_using_multibody_joints(
            &mut self.bodies,
            &mut self.colliders,
            &mut self.multibody_joints,
            UrdfMultibodyOptions::default(),
        );

        self.mass_total = urdf
            .links
            .iter()
            .map(|l| l.inertial.mass.value as Real)
            .sum();

        // Armature: add reflected motor inertia to every *actuated* link (the
        // child of a joint; the free base is untouched). The URDF inertia lives
        // in the body's additional mass-properties, so we rebuild them as
        // (URDF inertia + armature·I) rather than overwrite. Isotropic is a
        // deliberate simplification — off-axis inertia barely matters because
        // the joint constrains all but its one revolute axis.
        if armature > 0.0 {
            let a = armature as Real;
            for j in &urdf.joints {
                let ci = name_to_urdf_idx[&j.child.link];
                let inert = &urdf.links[ci].inertial;
                let com = Vector::new(
                    inert.origin.xyz[0] as Real,
                    inert.origin.xyz[1] as Real,
                    inert.origin.xyz[2] as Real,
                );
                let (ixx, iyy, izz) = (
                    inert.inertia.ixx as Real,
                    inert.inertia.iyy as Real,
                    inert.inertia.izz as Real,
                );
                let (ixy, ixz, iyz) = (
                    inert.inertia.ixy as Real,
                    inert.inertia.ixz as Real,
                    inert.inertia.iyz as Real,
                );
                let m = Matrix3::new(
                    ixx + a, ixy, ixz, ixy, iyy + a, iyz, ixz, iyz, izz + a,
                );
                let props =
                    MassProperties::with_inertia_matrix(com, inert.mass.value as Real, m.into());
                if let Some(rb) = self.bodies.get_mut(handles.links[ci].body) {
                    rb.set_additional_mass_properties(props, false);
                }
            }
        }

        // Build the canonical joint references.
        self.joints.clear();
        for name in &joint_names {
            let idx = *name_to_urdf_idx.get(name).ok_or_else(|| {
                PyRuntimeError::new_err(format!("joint '{name}' not found in URDF"))
            })?;
            let jh = handles.joints[idx].joint.ok_or_else(|| {
                PyRuntimeError::new_err(format!("joint '{name}' has no multibody handle"))
            })?;
            let (_, link_id) = self
                .multibody_joints
                .get(jh)
                .ok_or_else(|| PyRuntimeError::new_err("multibody handle resolved to nothing"))?;
            self.mb_handle.get_or_insert(jh);
            self.joints.push(JointRef {
                child: handles.joints[idx].link2,
                parent: handles.joints[idx].link1,
                link_id,
                frame2_rot: frame2_rots[idx],
                default_angle: 0.0,
            });
        }

        // Base rigid body = root of the multibody, and per-joint viscous damping.
        if let Some(jh) = self.mb_handle {
            if let Some((mb, _)) = self.multibody_joints.get_mut(jh) {
                self.root = Some(mb.root().rigid_body_handle());
                if joint_damping > 0.0 {
                    // Assemble the generalized DOFs, then damp every joint DOF
                    // while leaving the free base (first `root_ndofs`) undamped
                    // so we don't add artificial drag to the floating base.
                    mb.forward_kinematics(&self.bodies, true);
                    let root_ndofs = mb.root().joint().ndofs();
                    let damping = mb.damping_mut();
                    for i in root_ndofs..damping.len() {
                        damping[i] = joint_damping as Real;
                    }
                }
            }
        }

        self.tau = vec![0.0; self.joints.len()];
        Ok(())
    }

    /// Reset the robot to its nominal standing pose and zero all velocities.
    fn reset(&mut self, default_joint_pos: Vec<f64>) -> PyResult<()> {
        for (jr, q) in self.joints.iter_mut().zip(default_joint_pos.iter()) {
            jr.default_angle = *q as Real;
        }
        self.tau = vec![0.0; self.joints.len()];

        let root = self
            .root
            .ok_or_else(|| PyRuntimeError::new_err("reset() before load()"))?;
        // Place the base upright at the nominal height with zero velocity.
        let base_pose = Pose::from_parts(
            Vector::new(0.0, 0.0, self.base_height),
            Rotation::IDENTITY,
        );
        if let Some(rb) = self.bodies.get_mut(root) {
            rb.set_position(base_pose, true);
            rb.set_linvel(Vector::ZERO, true);
            rb.set_angvel(Vector::ZERO, true);
        }

        let jh = self
            .mb_handle
            .ok_or_else(|| PyRuntimeError::new_err("reset() before load()"))?;

        // Snapshot (link_id, target) first to avoid overlapping borrows.
        let targets: Vec<(usize, Real)> = self
            .joints
            .iter()
            .map(|jr| (jr.link_id, jr.default_angle))
            .collect();

        let (mb, _) = self
            .multibody_joints
            .get_mut(jh)
            .ok_or_else(|| PyRuntimeError::new_err("multibody missing"))?;
        // Normalise the root DOFs (reads the base pose we just set).
        mb.forward_kinematics(&self.bodies, true);
        for (link_id, target) in targets {
            let current = link_angle(mb, link_id);
            if let Some(link) = mb.link_mut(link_id) {
                link.joint.apply_displacement(&[target - current]);
            }
        }
        // Zero every generalised velocity (root + all joints).
        mb.generalized_velocity_mut().fill(0.0);
        mb.forward_kinematics(&self.bodies, true);
        mb.update_rigid_bodies(&mut self.bodies, true);
        Ok(())
    }

    /// Store the joint torques to apply on subsequent steps (canonical order).
    fn apply_torques(&mut self, tau: Vec<f64>) {
        self.tau = tau.iter().map(|&t| t as Real).collect();
    }

    /// Advance the physics by exactly `dt` seconds.
    fn step(&mut self) -> PyResult<()> {
        // Set the actuation fresh each step. rapier *persists* user forces across
        // steps until reset (see rapier's `user_force_persists_across_steps`
        // test), so we must clear last step's torques first — otherwise every
        // step's add_torque compounds onto the previous and the robot launches.
        for jr in &self.joints {
            if let Some(rb) = self.bodies.get_mut(jr.child) {
                rb.reset_torques(false);
            }
            if let Some(rb) = self.bodies.get_mut(jr.parent) {
                rb.reset_torques(false);
            }
        }
        // The joint axis is fixed in the child frame; the equal-and-opposite
        // pair (+tau on child, -tau on parent) is a pure internal joint torque.
        for (jr, &t) in self.joints.iter().zip(self.tau.iter()) {
            let torque = match self.bodies.get(jr.child) {
                Some(rb) => ((*rb.rotation()) * jr.frame2_rot * Vector::X) * t,
                None => continue,
            };
            if let Some(rb) = self.bodies.get_mut(jr.child) {
                rb.add_torque(torque, true);
            }
            if let Some(rb) = self.bodies.get_mut(jr.parent) {
                rb.add_torque(-torque, true);
            }
        }

        let hooks = ();
        let events = ();
        self.physics_pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &hooks,
            &events,
        );
        Ok(())
    }

    /// Native robot state: base position (3), base quaternion as `xyzw` (4),
    /// base linear velocity in WORLD frame (3), base angular velocity in WORLD
    /// frame (3), joint positions (n) and joint velocities (n) in canonical
    /// order. The Python adapter converts to the framework's conventions.
    fn get_state(
        &self,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let root = self
            .root
            .and_then(|h| self.bodies.get(h))
            .ok_or_else(|| PyRuntimeError::new_err("get_state() before load()"))?;
        let pos = root.translation();
        let quat = root.rotation(); // glam::Quat (x, y, z, w)
        let lin = root.linvel();
        let ang = root.angvel();

        let base_pos = vec![pos.x as f64, pos.y as f64, pos.z as f64];
        let base_quat_xyzw = vec![quat.x as f64, quat.y as f64, quat.z as f64, quat.w as f64];
        let lin_world = vec![lin.x as f64, lin.y as f64, lin.z as f64];
        let ang_world = vec![ang.x as f64, ang.y as f64, ang.z as f64];

        let jh = self
            .mb_handle
            .ok_or_else(|| PyRuntimeError::new_err("get_state() before load()"))?;
        let (mb, _) = self
            .multibody_joints
            .get(jh)
            .ok_or_else(|| PyRuntimeError::new_err("multibody missing"))?;

        let mut jpos = Vec::with_capacity(self.joints.len());
        let mut jvel = Vec::with_capacity(self.joints.len());
        for jr in &self.joints {
            jpos.push(link_angle(mb, jr.link_id) as f64);
            let v = mb
                .link(jr.link_id)
                .map(|link| {
                    let jv = mb.joint_velocity(link);
                    if jv.len() == 0 { 0.0 } else { jv[0] }
                })
                .unwrap_or(0.0);
            jvel.push(v as f64);
        }

        Ok((base_pos, base_quat_xyzw, lin_world, ang_world, jpos, jvel))
    }

    /// Total robot mass (kg), from the URDF <inertial> masses (see `mass_total`).
    fn total_mass(&self) -> f64 {
        self.mass_total as f64
    }
}

/// Signed revolute joint angle (radians) about the joint's free X axis.
///
/// `joint_rot` is the joint's relative rotation in its internal frame, where the
/// free axis is X, so the angle is `2*atan2(q.x, q.w)`.
fn link_angle(mb: &Multibody, link_id: usize) -> Real {
    match mb.link(link_id) {
        Some(link) => {
            let q = link.joint().joint_rot();
            2.0 * q.x.atan2(q.w)
        }
        None => 0.0,
    }
}

#[pymodule]
fn sim2sim_rapier(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RapierSim>()?;
    Ok(())
}

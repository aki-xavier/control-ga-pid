// The Rust plant the closed-loop tests drive: semi-implicit Euler over the PGA dynamics.

#![allow(dead_code)] // each integration test compiles this module on its own

use control_base::plant::{motor_of_rotor, Plant, PlantStructure, TaskMap};
use control_math::mat::Mat;
use control_math::vec3::Vec3;
use control_model::body_tree::{load_body_tree, BodyTree};
use control_model::mjcf_model::MjcfModel;
use control_model::pga_dynamics::PgaDynamicsModel;
use control_model::pga_layer::{mat_from_rotor, quat_from_rotor, rotor_from_mat};
use control_model::tree_dynamics::TreeDynamicsModel;
use control_model::urdf::{load_urdf_chain, UrdfChain};
use pga::Multivector;
use std::sync::Arc;

/// A fixed-base serial chain: model terms from the PGA model, state integrated here.
pub struct ChainPlant {
    pub n: usize,
    pub pdyn: PgaDynamicsModel,
    pub chain: UrdfChain,
    pub q: Vec<f64>,
    pub v: Vec<f64>,
    pub dt: f64,
    /// how many times the loop asked for the CONFIGURATION STAMP: the loop's metric refresh has to be
    /// driven by it and not by a position, which is what lets a plant whose coordinates are not a
    /// configuration (this machine's stance reduction) be driven at all
    pub stamp_calls: usize,
}

impl ChainPlant {
    pub fn new(urdf: &str, ee: &str, dt: f64) -> ChainPlant {
        let chain = load_urdf_chain(urdf, "link00", ee).expect("the chain parses");
        let n = chain.n;
        ChainPlant {
            n,
            pdyn: PgaDynamicsModel::new(chain.clone()),
            chain,
            q: vec![0.0; n],
            v: vec![0.0; n],
            dt,
            stamp_calls: 0,
        }
    }

    pub fn set_state(&mut self, q: &[f64], v: &[f64]) {
        assert!(q.len() == self.n && v.len() == self.n, "set_state arity");
        self.q.copy_from_slice(q);
        self.v.copy_from_slice(v);
    }

    /// Attaches a point mass at the terminal link; com_offset is in that link's frame.
    pub fn add_payload(&mut self, mass: f64, com_offset: Vec3) {
        self.pdyn.payload_mass = mass;
        self.pdyn.payload_com = com_offset;
    }

    /// Integrates the ODE for n_sub substeps; joint limits are acceleration constraints.
    pub fn step(&mut self, tau: &[f64], n_sub: usize) {
        let n = self.n;
        for _ in 0..n_sub {
            let m = self.mass_matrix();
            let zero = vec![0.0; n];
            let rg = self
                .pdyn
                .inverse_dynamics(&self.q.clone(), &self.v.clone(), &zero);
            let mut rhs = vec![0.0; n];
            for i in 0..n {
                rhs[i] = tau[i] - rg[i] - self.chain.dampings[i] * self.v[i];
            }
            let mut meff = m.clone();
            for i in 0..n {
                meff.set(i, i, meff.at(i, i) + self.dt * self.chain.dampings[i]);
            }
            let mut qdd = meff.solve(&rhs);
            let mut active: Vec<usize> = Vec::new();
            for i in 0..n {
                if (self.q[i] >= self.chain.limit_hi[i] - 1e-12 && qdd[i] > 0.0)
                    || (self.q[i] <= self.chain.limit_lo[i] + 1e-12 && qdd[i] < 0.0)
                {
                    active.push(i);
                }
            }
            if !active.is_empty() {
                let mut k = Mat::zeros(n + active.len(), n + active.len());
                for i in 0..n {
                    for j in 0..n {
                        k.set(i, j, meff.at(i, j));
                    }
                }
                for (a, ai) in active.iter().enumerate() {
                    k.set(*ai, n + a, -1.0);
                    k.set(n + a, *ai, 1.0);
                }
                let mut rhs_ext = vec![0.0; n + active.len()];
                rhs_ext[..n].copy_from_slice(&rhs[..n]);
                qdd = k.solve(&rhs_ext)[..n].to_vec();
            }
            for i in 0..n {
                self.v[i] += self.dt * qdd[i];
            }
            for i in 0..n {
                self.q[i] += self.dt * self.v[i];
                if self.q[i] > self.chain.limit_hi[i] {
                    self.q[i] = self.chain.limit_hi[i];
                    if self.v[i] > 0.0 {
                        self.v[i] = 0.0;
                    }
                } else if self.q[i] < self.chain.limit_lo[i] {
                    self.q[i] = self.chain.limit_lo[i];
                    if self.v[i] < 0.0 {
                        self.v[i] = 0.0;
                    }
                }
            }
        }
    }
}

/// The name this plant declares for its task frame: the tip.
const TASK_FRAME: &str = "tip";

impl ChainPlant {
    /// Resolves a body name to its chain index; an unknown name is a caller's error.
    fn body_index(&self, name: &str) -> usize {
        self.chain
            .child_names
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("this chain has no body {name}"))
    }
}

impl Plant for ChainPlant {
    fn structure(&self) -> PlantStructure {
        PlantStructure::fixed_base(
            self.n,
            self.chain.child_names.clone(),
            TaskMap::Frame(TASK_FRAME.to_string()),
        )
    }

    fn joint_positions(&mut self) -> Vec<f64> {
        self.q.clone()
    }

    fn joint_velocities(&mut self) -> Vec<f64> {
        self.v.clone()
    }

    /// The chain's coordinates ARE a configuration, so its stamp is its position — which is also what
    /// the contract's default does. It is written out here so the loop's USE of it is observable.
    fn configuration_stamp(&mut self) -> Vec<f64> {
        self.stamp_calls += 1;
        self.q.clone()
    }

    fn mass_matrix(&mut self) -> Mat {
        let q = self.q.clone();
        self.pdyn.mass_matrix(&q)
    }

    fn bias_torques(&mut self) -> Vec<f64> {
        let (q, v) = (self.q.clone(), self.v.clone());
        self.pdyn.bias_torques(&q, &v)
    }

    fn gravity_torques(&mut self) -> Vec<f64> {
        let q = self.q.clone();
        self.pdyn.gravity_torques(&q)
    }

    fn frame_motor(&mut self, name: &str) -> Multivector {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        if name == TASK_FRAME {
            return self.chain.tip_motor(&self.pdyn.fr_o, &self.pdyn.fr_r);
        }
        let i = self.body_index(name);
        motor_of_rotor(self.pdyn.fr_o[i], rotor_from_mat(&self.pdyn.fr_r[i]))
    }

    fn frame_position(&mut self, name: &str) -> Vec3 {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        if name == TASK_FRAME {
            return self.chain.tip_position(&self.pdyn.fr_o, &self.pdyn.fr_r);
        }
        self.pdyn.fr_o[self.body_index(name)]
    }

    fn frame_jacobian(&mut self, name: &str) -> Mat {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        let p = if name == TASK_FRAME {
            self.chain.tip_position(&self.pdyn.fr_o, &self.pdyn.fr_r)
        } else {
            self.pdyn.fr_o[self.body_index(name)]
        };
        self.chain
            .point_jacobian(&self.pdyn.fr_o, &self.pdyn.fr_r, p)
    }

    fn frame_full_jacobian(&mut self, name: &str) -> Mat {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        let p = if name == TASK_FRAME {
            self.chain.tip_position(&self.pdyn.fr_o, &self.pdyn.fr_r)
        } else {
            self.pdyn.fr_o[self.body_index(name)]
        };
        self.chain
            .full_jacobian(&self.pdyn.fr_o, &self.pdyn.fr_r, p)
    }

    /// point_position / point_jacobian: this chain presents NO non-frame points, which is what
    /// `PlantStructure::points` declares — a chain's every point of interest is on a link, and
    /// `frame_*` answers for those.
    fn point_position(&mut self, name: &str) -> Vec3 {
        unreachable!("{name}: this plant declares no points (PlantStructure::points is empty)")
    }

    fn point_jacobian(&mut self, name: &str) -> Mat {
        unreachable!("{name}: this plant declares no points (PlantStructure::points is empty)")
    }

    fn body_frame(&mut self, name: &str) -> (Vec3, Mat) {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        let i = self.body_index(name);
        (self.pdyn.fr_o[i], self.pdyn.fr_r[i].clone())
    }

    fn body_jacobian(&mut self, name: &str) -> Mat {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        let i = self.body_index(name);
        self.chain
            .link_jacobian(&self.pdyn.fr_o, &self.pdyn.fr_r, i)
    }
}

// ---------------------------------------------------------------------------------------------------
// THE FLOATING MACHINE: the same z1 chain with its ROOT IN THE STATE, which is the shape this
// project's "fixed base" is now a special case of. The base actuator's bound is the LOOP's to state —
// it is the leading `u_lim` rows — so this plant applies whatever it is handed, and the whole
// difference between fixed and floating is the number in that table: a very large one leaves the base
// where it is (the joint rows are then the welded chain's above), and one short of the dynamics' own
// demand lets the base give way under the joint reaction.
//
// The joints keep no limit-constraint active set the way `ChainPlant` does — the tests here stay far
// inside the range, and a limit's impulse belongs to the contact rather than to this file.
// ---------------------------------------------------------------------------------------------------

pub struct FloatingChainPlant {
    pub chain: UrdfChain,
    pub tree: Arc<BodyTree>,
    pub pdyn: TreeDynamicsModel,
    /// the machine's state: the base POSE (a position and a rotation) and the joint coordinates
    pub base_p: Vec3,
    pub base_rotor: Multivector,
    pub q: Vec<f64>,
    /// the whole state's velocity, in the tree dynamics' own order: `[omega; v; qd]`, world frame
    pub nu: Vec<f64>,
    /// per-coordinate viscous damping: zero on the base, the joints' own on the joints
    pub damp: Vec<f64>,
    pub dt: f64,
    /// the terminal link the task frame is measured on, and the chain's own tool reach off it
    pub tip_body: String,
    pub tool_reach: f64,
    pub stamp_calls: usize,
}

impl FloatingChainPlant {
    pub fn new(urdf: &str, ee: &str, dt: f64) -> FloatingChainPlant {
        let chain = load_urdf_chain(urdf, "link00", ee).expect("the chain parses");
        let tree = Arc::new(
            load_body_tree(urdf, &MjcfModel::default()).expect("the floating tree parses"),
        );
        // the two models must be the SAME machine, so their joint order is checked rather than assumed
        assert_eq!(
            tree.n_q(),
            chain.n,
            "the tree and the chain disagree on the joint count"
        );
        for (i, j) in chain.joint_names.iter().enumerate() {
            assert_eq!(
                &tree.q_names[i], j,
                "q slot {i} is {} in the tree and {j} in the chain",
                tree.q_names[i]
            );
        }
        let mut damp = vec![0.0; tree.nv()];
        for nd in tree.nodes.iter() {
            if nd.jo >= 0 {
                damp[6 + nd.jo as usize] = nd.damping;
            }
        }
        let nv = tree.nv();
        FloatingChainPlant {
            tool_reach: chain.tool_reach,
            chain,
            pdyn: TreeDynamicsModel::new(Arc::clone(&tree)),
            tree,
            base_p: Vec3::ZERO,
            base_rotor: pga::rotor_identity(),
            q: vec![0.0; nv - 6],
            nu: vec![0.0; nv],
            damp,
            dt,
            tip_body: ee.to_string(),
            stamp_calls: 0,
        }
    }

    pub fn nv(&self) -> usize {
        self.nu.len()
    }

    /// Puts the machine at a state: the base pose, the joints, and the whole velocity.
    pub fn set_state(&mut self, base_p: Vec3, base_rotor: Multivector, q: &[f64], nu: &[f64]) {
        assert!(q.len() == self.q.len(), "set_state joint arity");
        assert!(nu.len() == self.nv(), "set_state velocity arity");
        self.base_p = base_p;
        self.base_rotor = base_rotor;
        self.q.copy_from_slice(q);
        self.nu.copy_from_slice(nu);
    }

    /// Integrates the floating ODE for n_sub substeps: `M qdd = tau - C nu - g - D nu`, semi-implicit.
    pub fn step(&mut self, tau: &[f64], n_sub: usize) {
        let n = self.nv();
        for _ in 0..n_sub {
            let t = tau.to_vec();
            let m = self.pdyn.mass_matrix(self.base_p, self.base_rotor, &self.q);
            let zero = vec![0.0; n];
            let q = self.q.clone();
            let nu = self.nu.clone();
            let rg = self
                .pdyn
                .inverse_dynamics(self.base_p, self.base_rotor, &q, &nu, &zero);
            let mut rhs = vec![0.0; n];
            for i in 0..n {
                rhs[i] = t[i] - rg[i] - self.damp[i] * self.nu[i];
            }
            let mut meff = m.clone();
            for i in 0..n {
                meff.set(i, i, meff.at(i, i) + self.dt * self.damp[i]);
            }
            let qdd = meff.solve(&rhs);
            for i in 0..n {
                self.nu[i] += self.dt * qdd[i];
            }
            for j in 0..self.q.len() {
                self.q[j] += self.dt * self.nu[6 + j];
            }
            // the base: the twist's linear part translates the origin, its angular part turns the frame
            self.base_p = self
                .base_p
                .add(Vec3::new(self.nu[3], self.nu[4], self.nu[5]).scale(self.dt));
            let w = Vec3::new(self.nu[0], self.nu[1], self.nu[2]);
            let wn = w.norm();
            if wn > 1e-12 {
                let inc = pga::rotor([w.x / wn, w.y / wn, w.z / wn], wn * self.dt);
                self.base_rotor = inc.gp(self.base_rotor);
            }
        }
    }

    /// The cached world frames of the current state.
    fn frames(&mut self) -> (Vec<Vec3>, Vec<Mat>) {
        self.pdyn
            .refresh_frames(self.base_p, self.base_rotor, &self.q);
        let (o, r) = self.pdyn.frames_now();
        (o.to_vec(), r.to_vec())
    }

    fn node_of(&self, name: &str) -> usize {
        self.tree
            .node_index(name)
            .unwrap_or_else(|| panic!("this tree has no body {name}"))
    }

    /// The task frame: the terminal link's own frame, offset by the chain's tool reach — the same point
    /// `ChainPlant` answers for, so a welded run can be compared against it.
    fn tip(&mut self) -> (Vec3, Mat) {
        let i = self.node_of(&self.tip_body);
        let (o, r) = self.frames();
        let off = Vec3::new(self.tool_reach, 0.0, 0.0);
        (o[i].add(r[i].mul_vec3(off)), r[i].clone())
    }

    /// The six-row Jacobian at a point held by node `i`, stacked [linear; angular] — the task loop's
    /// own order, in this machine's `[omega; v; qd]` coordinates.
    fn full_jacobian_at(&self, o: &[Vec3], r: &[Mat], i: usize, off: Vec3) -> Mat {
        let nv = self.tree.nv();
        let lin = self.tree.offset_point_jacobian(o, r, i, off);
        let mut j = Mat::zeros(6, nv);
        for a in 0..3 {
            for b in 0..nv {
                j.set(a, b, lin.at(a, b));
            }
        }
        // the base's angular columns are the identity: a root rotation turns every world axis with it
        for b in 0..3 {
            j.set(3 + b, b, 1.0);
        }
        for qj in self.tree.path_joints(i) {
            let ni = self.tree.node_of_q(qj).expect("a q slot has a node");
            let z = self.tree.world_z(r, ni);
            for b in 0..3 {
                j.set(3 + b, 6 + qj, z.to_array()[b]);
            }
        }
        j
    }
}

impl Plant for FloatingChainPlant {
    fn structure(&self) -> PlantStructure {
        let nj = self.q.len();
        let mut actuated = vec![false; 6];
        actuated.extend(std::iter::repeat_n(true, nj));
        PlantStructure {
            dof: 6 + nj,
            actuated,
            // the leading six coordinates are the base pose; `contacts` stays empty because what holds
            // them is the loop's own unbounded actuator and not a distal contact
            base_dof: 6,
            contacts: Vec::new(),
            bodies: self.chain.child_names.clone(),
            points: Vec::new(),
            task_map: TaskMap::Frame(TASK_FRAME.to_string()),
        }
    }

    fn joint_positions(&mut self) -> Vec<f64> {
        // the coordinate reading of the same 12 coordinates the velocities live in: the base's position
        // and the base rotation's rotvec, then the joints
        let rv = mat_from_rotor(&self.base_rotor).to_rotvec();
        let mut out = vec![
            self.base_p.x,
            self.base_p.y,
            self.base_p.z,
            rv.x,
            rv.y,
            rv.z,
        ];
        out.extend_from_slice(&self.q);
        out
    }

    fn joint_velocities(&mut self) -> Vec<f64> {
        self.nu.clone()
    }

    /// The stamp IS the configuration here — a position and a rotation have no common vector, so the
    /// quaternion rides along and the metric's refresh notices a base that turned.
    fn configuration_stamp(&mut self) -> Vec<f64> {
        self.stamp_calls += 1;
        let q = quat_from_rotor(self.base_rotor);
        let mut out = vec![
            self.base_p.x,
            self.base_p.y,
            self.base_p.z,
            q.w,
            q.x,
            q.y,
            q.z,
        ];
        out.extend_from_slice(&self.q);
        out
    }

    fn mass_matrix(&mut self) -> Mat {
        let q = self.q.clone();
        self.pdyn.mass_matrix(self.base_p, self.base_rotor, &q)
    }

    fn bias_torques(&mut self) -> Vec<f64> {
        let (q, nu) = (self.q.clone(), self.nu.clone());
        self.pdyn
            .bias_torques(self.base_p, self.base_rotor, &q, &nu)
    }

    fn gravity_torques(&mut self) -> Vec<f64> {
        let q = self.q.clone();
        self.pdyn.gravity_torques(self.base_p, self.base_rotor, &q)
    }

    fn frame_motor(&mut self, name: &str) -> Multivector {
        if name == TASK_FRAME {
            let (p, r) = self.tip();
            return motor_of_rotor(p, rotor_from_mat(&r));
        }
        let i = self.node_of(name);
        let (o, r) = self.frames();
        motor_of_rotor(o[i], rotor_from_mat(&r[i]))
    }

    fn frame_position(&mut self, name: &str) -> Vec3 {
        if name == TASK_FRAME {
            return self.tip().0;
        }
        let i = self.node_of(name);
        self.frames().0[i]
    }

    fn frame_jacobian(&mut self, name: &str) -> Mat {
        let (i, off) = if name == TASK_FRAME {
            (
                self.node_of(&self.tip_body),
                Vec3::new(self.tool_reach, 0.0, 0.0),
            )
        } else {
            (self.node_of(name), Vec3::ZERO)
        };
        let (o, r) = self.frames();
        self.tree.offset_point_jacobian(&o, &r, i, off)
    }

    fn frame_full_jacobian(&mut self, name: &str) -> Mat {
        let (i, off) = if name == TASK_FRAME {
            (
                self.node_of(&self.tip_body),
                Vec3::new(self.tool_reach, 0.0, 0.0),
            )
        } else {
            (self.node_of(name), Vec3::ZERO)
        };
        let (o, r) = self.frames();
        self.full_jacobian_at(&o, &r, i, off)
    }

    /// point_position / point_jacobian: this machine presents NO non-frame points, which is what
    /// `PlantStructure::points` declares.
    fn point_position(&mut self, name: &str) -> Vec3 {
        unreachable!("{name}: this plant declares no points (PlantStructure::points is empty)")
    }

    fn point_jacobian(&mut self, name: &str) -> Mat {
        unreachable!("{name}: this plant declares no points (PlantStructure::points is empty)")
    }

    fn body_frame(&mut self, name: &str) -> (Vec3, Mat) {
        let i = self.node_of(name);
        let (o, r) = self.frames();
        (o[i], r[i].clone())
    }

    fn body_jacobian(&mut self, name: &str) -> Mat {
        let i = self.node_of(name);
        let (o, r) = self.frames();
        self.tree.offset_point_jacobian(&o, &r, i, Vec3::ZERO)
    }
}

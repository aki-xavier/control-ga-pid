// common/mod.rs — the Rust plant the closed-loop tests drive, and the whole of what this crate needs
// from a dynamics backend.
//
// It is the SAME integrator simu's engine binding runs, not a stand-in: semi-implicit Euler over the
// PGA dynamics, the joint damping made implicit as (M + dt D), joint limits as bilateral constraints
// on the acceleration, and the position clamp after the velocity update. What is left out is the
// engine's world — the actor, the scene, the contact model — which no law reads (the engine only
// mirrors the state, and its `Plant` methods are the same pure-Rust calls this file makes). So a
// number measured here is the number simu's engine-backed bench measures, up to what contact adds.
//
// It lives in `tests/` and not in `src/` on purpose: this crate states the law, and an implementor of
// `Plant` is a caller's object — an engine binding, a model-based view, or this.

#![allow(dead_code)] // each integration test compiles this module on its own

use control_base::plant::Plant;
use control_math::mat::Mat;
use control_math::quat::Quat;
use control_math::vec3::Vec3;
use control_model::pga_dynamics::PgaDynamicsModel;
use control_model::urdf::{load_urdf_chain, UrdfChain};

/// ChainPlant is a fixed-base serial chain: model terms from the PGA dynamics model, state integrated
/// here.
pub struct ChainPlant {
    pub n: usize,
    pub pdyn: PgaDynamicsModel,
    pub chain: UrdfChain,
    pub q: Vec<f64>,
    pub v: Vec<f64>,
    pub dt: f64,
}

impl ChainPlant {
    /// new builds the chain at the zero configuration; set_state places it.
    pub fn new(urdf: &str, ee: &str, dt: f64) -> ChainPlant {
        let chain = load_urdf_chain(urdf, "link00", ee).expect("the Z1 chain parses");
        let n = chain.n;
        ChainPlant {
            n,
            pdyn: PgaDynamicsModel::new(chain.clone()),
            chain,
            q: vec![0.0; n],
            v: vec![0.0; n],
            dt,
        }
    }

    pub fn set_state(&mut self, q: &[f64], v: &[f64]) {
        assert!(q.len() == self.n && v.len() == self.n, "set_state arity");
        self.q.copy_from_slice(q);
        self.v.copy_from_slice(v);
    }

    /// add_payload attaches a point mass at the terminal link (com_offset in the terminal link frame).
    pub fn add_payload(&mut self, mass: f64, com_offset: Vec3) {
        self.pdyn.payload_mass = mass;
        self.pdyn.payload_com = com_offset;
    }

    /// step integrates the rigid-body ODE for n_sub substeps, all dynamics terms from the PGA unified
    /// layer. rg = M qdd + C qd + g at qdd = 0, so rhs = tau - rg - damp*v assembles the bias and the
    /// gravity in one inverse-dynamics pass.
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
            // joint limits as bilateral constraints on the acceleration
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

impl Plant for ChainPlant {
    fn joint_positions(&mut self) -> Vec<f64> {
        self.q.clone()
    }

    fn joint_velocities(&mut self) -> Vec<f64> {
        self.v.clone()
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

    fn compute_jacobian(&mut self) -> Mat {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        let tip = self.chain.tip_pose(&self.pdyn.fr_o, &self.pdyn.fr_r).0;
        self.chain
            .point_jacobian(&self.pdyn.fr_o, &self.pdyn.fr_r, tip)
    }

    fn compute_full_jacobian(&mut self) -> Mat {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        let tip = self.chain.tip_pose(&self.pdyn.fr_o, &self.pdyn.fr_r).0;
        self.chain
            .full_jacobian(&self.pdyn.fr_o, &self.pdyn.fr_r, tip)
    }

    fn body_pose(&mut self) -> (Vec3, Quat) {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        self.chain.tip_pose(&self.pdyn.fr_o, &self.pdyn.fr_r)
    }

    fn link_frame(&mut self, i: usize) -> (Vec3, Mat) {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        (self.pdyn.fr_o[i], self.pdyn.fr_r[i].clone())
    }

    fn link_jacobian(&mut self, i: usize) -> Mat {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        self.chain
            .link_jacobian(&self.pdyn.fr_o, &self.pdyn.fr_r, i)
    }
}

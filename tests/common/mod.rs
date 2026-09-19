// The Rust plant the closed-loop tests drive: semi-implicit Euler over the PGA dynamics.

#![allow(dead_code)] // each integration test compiles this module on its own

use control_base::plant::{Plant, PlantStructure};
use control_math::mat::Mat;
use control_math::quat::Quat;
use control_math::vec3::Vec3;
use control_model::pga_dynamics::PgaDynamicsModel;
use control_model::urdf::{load_urdf_chain, UrdfChain};

/// A fixed-base serial chain: model terms from the PGA model, state integrated here.
pub struct ChainPlant {
    pub n: usize,
    pub pdyn: PgaDynamicsModel,
    pub chain: UrdfChain,
    pub q: Vec<f64>,
    pub v: Vec<f64>,
    pub dt: f64,
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
            TASK_FRAME.to_string(),
        )
    }

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

    fn frame_pose(&mut self, name: &str) -> (Vec3, Quat) {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        if name == TASK_FRAME {
            return self.chain.tip_pose(&self.pdyn.fr_o, &self.pdyn.fr_r);
        }
        let i = self.body_index(name);
        (
            self.pdyn.fr_o[i],
            Quat::from_mat3(&self.pdyn.fr_r[i].clone()),
        )
    }

    fn frame_jacobian(&mut self, name: &str) -> Mat {
        let q = self.q.clone();
        self.pdyn.frames(&q);
        let p = if name == TASK_FRAME {
            self.chain.tip_pose(&self.pdyn.fr_o, &self.pdyn.fr_r).0
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
            self.chain.tip_pose(&self.pdyn.fr_o, &self.pdyn.fr_r).0
        } else {
            self.pdyn.fr_o[self.body_index(name)]
        };
        self.chain
            .full_jacobian(&self.pdyn.fr_o, &self.pdyn.fr_r, p)
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

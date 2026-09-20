// base.rs — the base end of a floating machine, as a realization needs it: the JOINT BLOCK a held base
// leaves behind (M_jj, J_j, the bias and the velocity rows), and the CONSTRAINT WRENCH that holding it
// demands.
//
// WHY A WELD IS A CONSTRAINT AND NOT A DIFFERENT MACHINE: a fixed-base machine is not another model, it
// is this one with an actuator at the base bound by a number no command reaches. Whatever the dynamics
// demand it supplies, so the base cannot accelerate, and every joint row then obeys exactly the equation
// the welded machine's own model would have written — which is why the split below exists at all. The
// joints are computed as if the base were not there; the base rows are computed as what makes that true.
//
// The bound that separates the two is the same number the rest of the command is clamped by (`u_lim`'s
// leading rows), so this module states the split and the wrench and leaves the magnitude to the caller:
// a bound short of the demand is a machine whose base gives way, which is the floating reading again.

use control_math::mat::Mat;

/// HeldBase: the leading `nb` of an `n`-coordinate machine, read as a base the loop holds. Why a type
/// and not two indices: every quantity a realization reads has to be sliced the same way, and the wrench
/// has to be assembled with the same split it was read with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldBase {
    /// how many LEADING coordinates are the base pose
    pub nb: usize,
    /// the machine's coordinate count, the base included
    pub n: usize,
}

impl HeldBase {
    pub fn new(nb: usize, n: usize) -> HeldBase {
        assert!(nb <= n, "a held base is a prefix of the coordinates");
        HeldBase { nb, n }
    }

    /// the coordinates the law is written in: everything but the base
    pub fn nj(&self) -> usize {
        self.n - self.nb
    }

    /// M_jj, the joint block. Why this block and not an inverse: a held base's acceleration is zero,
    /// so its rows leave the joint equation, and the inertia that multiplies qdd_j is exactly this one.
    pub fn joint_block(&self, m: &Mat) -> Mat {
        let nj = self.nj();
        let mut out = Mat::zeros(nj, nj);
        for i in 0..nj {
            for j in 0..nj {
                out.set(i, j, m.at(self.nb + i, self.nb + j));
            }
        }
        out
    }

    /// the joint COLUMNS of a task Jacobian: a base that does not move contributes no task motion.
    pub fn joint_cols(&self, a: &Mat) -> Mat {
        let nj = self.nj();
        let mut out = Mat::zeros(a.rows, nj);
        for i in 0..a.rows {
            for j in 0..nj {
                out.set(i, j, a.at(i, self.nb + j));
            }
        }
        out
    }

    /// M_bj, the joint columns of the base's rows: the coupling the wrench has to cancel.
    pub fn base_coupling(&self, m: &Mat) -> Mat {
        let nj = self.nj();
        let mut out = Mat::zeros(self.nb, nj);
        for i in 0..self.nb {
            for j in 0..nj {
                out.set(i, j, m.at(i, self.nb + j));
            }
        }
        out
    }

    /// rows `nb..n` of a full-coordinate vector: the joint part of a bias, a velocity, a limit.
    pub fn tail(&self, v: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.nj());
        for i in self.nb..self.n {
            out.push(if i < v.len() { v[i] } else { 0.0 });
        }
        out
    }

    /// rows `..nb`: the base part of the same vector.
    pub fn head(&self, v: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.nb);
        for i in 0..self.nb {
            out.push(if i < v.len() { v[i] } else { 0.0 });
        }
        out
    }

    /// tau_base = M_bj qdd_j + h_base: the base rows of `M qdd + h` at qdd_base = 0, which is what an
    /// actuator with no torque bound must supply for the base to stay where it is.
    pub fn wrench(&self, m: &Mat, h: &[f64], qdd_j: &[f64]) -> Vec<f64> {
        let c = self.base_coupling(m);
        let mut out = Vec::with_capacity(self.nb);
        for i in 0..self.nb {
            let mut s = if i < h.len() { h[i] } else { 0.0 };
            for j in 0..self.nj() {
                s += c.at(i, j) * qdd_j[j];
            }
            out.push(s);
        }
        out
    }

    /// the command in the machine's own coordinates: the base's rows, then the joints'.
    pub fn assemble(&self, base: &[f64], joints: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n];
        let nb = self.nb.min(base.len());
        out[..nb].copy_from_slice(&base[..nb]);
        let nj = self.nj().min(joints.len());
        out[self.nb..self.nb + nj].copy_from_slice(&joints[..nj]);
        out
    }
}

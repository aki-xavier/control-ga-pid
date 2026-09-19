// inertia.rs — Lambda = (J M^-1 J^T)^-1, the per-plane quantities derived from it, and the passivity floor.

use control_math::mat::Mat;

/// lambda_x = (J M^-1 J^T)^-1 of the task frame.
pub fn effective_task_inertia(m: &Mat, j: &Mat, n: usize) -> Mat {
    task_space_inertia(m, j, n)
}

/// The diagonal of lambda_x as per-plane masses (result length = Jacobian rows).
pub fn effective_task_mass(m: &Mat, j: &Mat, n: usize) -> Vec<f64> {
    effective_task_inertia(m, j, n).diag()
}

/// The damping a passive port needs: D >= 2 sqrt(K M), returned as max(D, floor).
pub fn passivity_floor(k: f64, d: f64, mass: f64) -> f64 {
    if mass <= 1e-12 || k <= 0.0 {
        return d;
    }
    let crit = 2.0 * (k * mass).sqrt();
    if d < crit {
        crit
    } else {
        d
    }
}

/// Lambda = (J M^-1 J^T)^-1 for an m-row task Jacobian.
/// The full matrix is needed: the off-diagonal terms are as large as the diagonal.
pub fn task_space_inertia(m: &Mat, j: &Mat, n: usize) -> Mat {
    let mrows = j.rows;
    let mut minv = Mat::zeros(n, n);
    for c in 0..n {
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        let x = m.solve(&e);
        for i in 0..n {
            minv.set(i, c, x[i]);
        }
    }
    let mut l = Mat::zeros(mrows, mrows);
    for a in 0..mrows {
        for b in 0..mrows {
            let mut s = 0.0;
            for k in 0..n {
                for q in 0..n {
                    s += j.at(a, k) * minv.at(k, q) * j.at(b, q);
                }
            }
            l.set(a, b, s);
        }
    }
    let mut lam = Mat::zeros(mrows, mrows);
    for c in 0..mrows {
        let mut e = vec![0.0; mrows];
        e[c] = 1.0;
        let x = l.solve(&e);
        for i in 0..mrows {
            lam.set(i, c, x[i]);
        }
    }
    lam
}

pub(crate) fn fill(x: &[f64], m: usize, def: f64) -> Vec<f64> {
    let mut out = vec![def; m];
    for i in 0..m {
        if i < x.len() {
            out[i] = x[i];
        }
    }
    out
}

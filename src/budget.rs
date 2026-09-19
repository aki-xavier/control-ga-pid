// budget.rs — MotionBudget and the demand scan; the actuator budget |g_i + wn^2 (J' Lambda u)_i| <= u_lim_i bounds wn.

use crate::design::{wn_for_settling, zeta_from_overshoot};
use control_math::mat::Mat;
use control_math::lstsq::DampedLstsq;
use std::f64::consts::PI;

/// The window the budgets leave open: wn_lo from settling, wn_hi from the actuator scan; feasible = wn_hi >= wn_lo.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MotionBudget {
    pub zeta_lo: f64,
    pub wn_lo: f64,
    pub wn_hi: f64,
    pub binding: String,
    pub feasible: bool,
    /// names[i] is joint i's name in the scan's chain order; empty until `with_names` fills it in.
    pub names: Vec<String>,
}

impl MotionBudget {
    /// Resolves the binding channel through a model's own joint names; an out-of-range index or "none" is kept.
    pub fn with_names(mut self, names: &[String]) -> MotionBudget {
        if let Some(i) = self.binding_index() {
            if let Some(n) = names.get(i) {
                self.binding = n.clone();
            }
        }
        self.names = names.to_vec();
        self
    }

    /// Index the binding channel names, or None when nothing binds or the name is already resolved.
    pub fn binding_index(&self) -> Option<usize> {
        self.binding.strip_prefix("joint ")?.parse().ok()
    }
}

/// Per-joint peak torque at the step instant as a fraction of u_lim, plus the joint owning the worst ratio.
#[allow(clippy::too_many_arguments)]
pub fn step_demand(
    m: &Mat,
    j: &Mat,
    g: &[f64],
    k_eff: &[f64],
    u_lim: &[f64],
    step: &[f64],
    wn: f64,
    ridge: f64,
    max_step: f64,
) -> (f64, String) {
    let nu = j.rows;
    let nj = j.cols;
    // damped least squares dq = J' (J J' + ridge I)^-1 e
    let dq_full = DampedLstsq::new(nu, ridge).solve_right(j, step);
    let mut dq = vec![0.0; nj];
    for k in 0..nj {
        let mut s = dq_full[k];
        if max_step > 0.0 {
            if s > max_step {
                s = max_step;
            }
            if s < -max_step {
                s = -max_step;
            }
        }
        dq[k] = s;
    }
    let mut a = vec![0.0; nj];
    for k in 0..nj {
        let mut kp = wn * wn;
        if k < k_eff.len() {
            kp -= k_eff[k];
        }
        a[k] = kp * dq[k];
    }
    let av = m.mul_vec(&a);
    let mut worst = 0.0;
    let mut owner = "none".to_string();
    for i in 0..nj {
        if i >= u_lim.len() || u_lim[i] <= 0.0 {
            continue;
        }
        let tau = g[i] + av[i];
        let ratio = tau.abs() / u_lim[i];
        if ratio > worst {
            worst = ratio;
            owner = format!("joint {i}");
        }
    }
    (worst, owner)
}

/// Solves the window for a step, scanning wn with the law's own demand estimate.
#[allow(clippy::too_many_arguments)]
pub fn motion_budget(
    m: &Mat,
    j: &Mat,
    g: &[f64],
    k_eff: &[f64],
    u_lim: &[f64],
    step: &[f64],
    mp: f64,
    ts: f64,
    ridge: f64,
    max_step: f64,
) -> MotionBudget {
    let zeta = zeta_from_overshoot(mp);
    let mut hi = 0.0;
    let mut bind = "none".to_string();
    let mut w = 1.0;
    while w <= 400.0 {
        let (d, owner) = step_demand(m, j, g, k_eff, u_lim, step, w, ridge, max_step);
        if d > 1.0 {
            bind = owner;
            break;
        }
        hi = w;
        w += 0.5;
    }
    let lo = wn_for_settling(zeta, ts);
    MotionBudget {
        zeta_lo: zeta,
        wn_lo: lo,
        wn_hi: hi,
        binding: bind,
        feasible: hi > 0.0 && hi >= lo,
        names: Vec::new(),
    }
}

/// Feasibility verdict for a commanded tip step from the model a run will drive; RECORDED, not enforced.
#[allow(clippy::too_many_arguments)]
pub fn budget_verdict(
    m: &Mat,
    j: &Mat,
    g: &[f64],
    k_eff: &[f64],
    u_lim: &[f64],
    step: &[f64],
    names: &[String],
    wn: f64,
    zeta: f64,
) -> MotionBudget {
    let ts = if zeta > 0.0 { 4.0 / (zeta * wn) } else { 0.0 };
    let mp = mp_from_zeta(zeta);
    motion_budget(m, j, g, k_eff, u_lim, step, mp, ts, 1e-6, 0.2).with_names(names)
}

/// Standard second-order overshoot relation; 0.0 at or past critical damping.
fn mp_from_zeta(zeta: f64) -> f64 {
    if zeta >= 1.0 {
        return 0.0;
    }
    (-PI * zeta / (1.0 - zeta * zeta).sqrt()).exp()
}

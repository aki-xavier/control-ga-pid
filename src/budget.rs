// budget.rs — MotionBudget and the demand scan that produces it: where the motion tier's numbers come
// from. A bare sweep cannot choose (wn, zeta), so a set of budgets does: overshoot `zeta >= |ln Mp| / sqrt(pi^2
// + ln^2 Mp)`, settling `wn >= 4/(zeta Ts)`, sampling `wn * dt << 1`, integral, and the per-joint ACTUATOR
// inequality `|g_i + wn^2 e (J' Lambda u)_i| <= u_lim_i`, which turns the task's step into a ceiling on wn.

use crate::design::{wn_for_settling, zeta_from_overshoot};
use control_math::mat::Mat;
use control_math::lstsq::DampedLstsq;
use std::f64::consts::PI;

/// MotionBudget is the window the budgets leave open; `feasible` is false when wn_hi < wn_lo (the actuators
/// cannot deliver the settling time the task asks, and no gain choice repairs it). Unlike MotionWindow
/// (window.rs), its ceiling is the actuators' and not the sampling bound 0.2/dt.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MotionBudget {
    pub zeta_lo: f64,
    pub wn_lo: f64,
    pub wn_hi: f64,
    pub binding: String,
    pub feasible: bool,
    /// names[i] is joint i's own name in the chain order the scan indexes; EMPTY IS THE SHIPPED READING because
    /// the scan works by index — `with_names` is where a caller that has the model puts its names in.
    pub names: Vec<String>,
}

impl MotionBudget {
    /// with_names resolves the binding channel through a model's own joint names in the scan's chain order; an
    /// index past the list, or a binding of "none", keeps the form it had. The names are kept on the value too.
    pub fn with_names(mut self, names: &[String]) -> MotionBudget {
        if let Some(i) = self.binding_index() {
            if let Some(n) = names.get(i) {
                self.binding = n.clone();
            }
        }
        self.names = names.to_vec();
        self
    }

    /// binding_index is the joint the binding channel names (the scan's own "joint {i}"), or None when nothing
    /// binds or the name has already been resolved through `with_names`.
    pub fn binding_index(&self) -> Option<usize> {
        self.binding.strip_prefix("joint ")?.parse().ok()
    }
}

/// step_demand returns the per-joint peak torque the law commands at the step instant as a fraction of u_lim,
/// plus the joint that owns the worst ratio: dq = J+ e (clipped by max_step), scaled by kappa_p = wn^2 - k_eff,
/// then M (kappa_p .* dq) + g — the law's own demand, not the operational-space shortcut tau = J' Lambda wn^2 e.
// The demand's inputs stay separate arguments; the doc above names each.
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
    // damped least squares: dq = J' (J J' + ridge I)^-1 e, the right-inverse ridge form used throughout the core
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

/// motion_budget solves the window for a step, scanning wn so the ceiling uses the same demand estimate the law
/// would produce rather than an analytic shortcut through wn^2.
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
        // the scan names joints by index and this module holds no model; `with_names` is where names come in
        names: Vec::new(),
    }
}

/// budget_verdict is the arm bench's caller of the motion budget: the feasibility verdict for a commanded
/// tip step, taken from the model the run will drive before its first command. The window's floor is the
/// design's own (wn, zeta) pair; the verdict is RECORDED, not enforced, and no number in the run comes from it.
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

/// mp_from_zeta is the standard second-order overshoot relation that `design::zeta_from_overshoot` inverts; 0.0 at or past critical damping.
fn mp_from_zeta(zeta: f64) -> f64 {
    if zeta >= 1.0 {
        return 0.0;
    }
    (-PI * zeta / (1.0 - zeta * zeta).sqrt()).exp()
}

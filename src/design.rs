// design.rs — PlaneDesign, the GA-PID per-plane design tuple (wn, zeta, alpha) and the
// pole-placement operations derived from it. Value type, DOF-agnostic; the gain triple it
// produces is gains.rs, its task-class inputs are spec.rs / class.rs.

use crate::gains::PlaneGains;

/// PlaneDesign carries the design tuple and derives per-plane gains:
/// kappa_p = wn^2 - k_eff, kappa_d = 2 zeta wn - b_eff, kappa_i = alpha.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaneDesign {
    pub wn: f64,
    pub zeta: f64,
    pub alpha: f64,
}

/// NEGATIVE_STIFFNESS_WARNED makes the `kappa_p <= 0` report below once per process: a design
/// error rather than a runtime condition, so every tick would be noise.
static NEGATIVE_STIFFNESS_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

impl PlaneDesign {
    pub fn new(wn: f64, zeta: f64, alpha: f64) -> PlaneDesign {
        PlaneDesign { wn, zeta, alpha }
    }

    /// gains applies the pole placement for one (k_eff, b_eff) pair, subtracting the plant's
    /// stiffness and damping from it. The loop instead removes both by FEEDFORWARD and calls this
    /// with k_eff = b_eff = 0 (GA_PID_AUDIT.md #4/#5); a non-positive kappa_p is a design error,
    /// reported once per process and never clamped, so it stays visible in the response.
    pub fn gains(&self, k_eff: f64, b_eff: f64) -> PlaneGains {
        let kappa_p = self.wn * self.wn - k_eff;
        if kappa_p <= 0.0
            && !NEGATIVE_STIFFNESS_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "plane_design: kappa_p = wn^2 - k_eff = {:.6} (wn = {}, k_eff = {k_eff}) is not \
                 positive: this plane has no authority over its own response. Reported once per \
                 process; the value is passed through unclamped so the design error is visible \
                 in the response rather than hidden as a small gain.",
                kappa_p, self.wn
            );
        }
        debug_assert!(kappa_p.is_finite(), "kappa_p must be finite");
        PlaneGains {
            kappa_p,
            kappa_d: 2.0 * self.zeta * self.wn - b_eff,
            kappa_i: self.alpha,
        }
    }
}

/// alpha_for_ti sizes the integral gain from an integral time constant: alpha = kp / Ti = wn^2 / Ti
/// (ti <= 0 gives 0). Measured on the Z1 with an unknown constant torque, wn = 15: the time-constant
/// rule (Ti = 1 s, alpha = 225) rejects the disturbance where a flat alpha is either too slow to
/// remove the offset or drifts on its own.
pub fn alpha_for_ti(wn: f64, ti: f64) -> f64 {
    if ti <= 0.0 {
        return 0.0;
    }
    wn * wn / ti
}

impl PlaneDesign {
    /// with_ti builds a design whose integral comes from the time constant, keeping wn and zeta
    /// explicit. Default policy, measured on the Z1 (wn = 15, kp = 225) by two orthogonal tests,
    /// disturbance rejection and a saturated task: Ti = 1 s -> alpha = 225. Anti-windup stays on as
    /// a transient constraint, not as the recovery mechanism.
    pub fn with_ti(wn: f64, zeta: f64, ti: f64) -> PlaneDesign {
        PlaneDesign::new(wn, zeta, alpha_for_ti(wn, ti))
    }
}

/// zeta_from_overshoot inverts the second-order relation Mp = exp(-pi zeta / sqrt(1 - zeta^2)),
/// so a class that must not overshoot names its zeta here instead of through taste.
pub fn zeta_from_overshoot(mp: f64) -> f64 {
    if mp <= 0.0 {
        return 1.0;
    }
    let l = mp.ln();
    l.abs() / (std::f64::consts::PI * std::f64::consts::PI + l * l).sqrt()
}

/// wn_for_settling is the 2 percent settling-time lower bound.
pub fn wn_for_settling(zeta: f64, ts: f64) -> f64 {
    if ts <= 0.0 || zeta <= 0.0 {
        return 0.0;
    }
    4.0 / (zeta * ts)
}

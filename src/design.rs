// design.rs — PlaneDesign (wn, zeta, alpha) and the pole placement that produces PlaneGains.

use crate::gains::PlaneGains;

/// Per-plane design tuple; gains are kappa_p = wn^2 - k_eff, kappa_d = 2 zeta wn - b_eff, kappa_i = alpha.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaneDesign {
    pub wn: f64,
    pub zeta: f64,
    pub alpha: f64,
}

static NEGATIVE_STIFFNESS_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

impl PlaneDesign {
    pub fn new(wn: f64, zeta: f64, alpha: f64) -> PlaneDesign {
        PlaneDesign { wn, zeta, alpha }
    }

    /// Pole placement for one (k_eff, b_eff); a non-positive kappa_p is reported once and never clamped.
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

/// Integral gain from an integral time constant: alpha = wn^2 / Ti; ti <= 0 gives 0.
pub fn alpha_for_ti(wn: f64, ti: f64) -> f64 {
    if ti <= 0.0 {
        return 0.0;
    }
    wn * wn / ti
}

impl PlaneDesign {
    /// Design whose integral comes from Ti, with wn and zeta kept explicit.
    pub fn with_ti(wn: f64, zeta: f64, ti: f64) -> PlaneDesign {
        PlaneDesign::new(wn, zeta, alpha_for_ti(wn, ti))
    }
}

/// Inverts Mp = exp(-pi zeta / sqrt(1 - zeta^2)); mp <= 0 gives 1.0.
pub fn zeta_from_overshoot(mp: f64) -> f64 {
    if mp <= 0.0 {
        return 1.0;
    }
    let l = mp.ln();
    l.abs() / (std::f64::consts::PI * std::f64::consts::PI + l * l).sqrt()
}

/// 2 percent settling-time bound wn = 4/(zeta ts); ts <= 0 or zeta <= 0 gives 0.
pub fn wn_for_settling(zeta: f64, ts: f64) -> f64 {
    if ts <= 0.0 || zeta <= 0.0 {
        return 0.0;
    }
    4.0 / (zeta * ts)
}

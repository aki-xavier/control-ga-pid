// law.rs — THE LAW, one plane: num_i = k_i e_i - d_i (v_i - v_ref_i) + alpha_i i_acc_i.

use crate::design::PlaneDesign;
use crate::gains::PlaneGains;

/// Pole placement for a loop stating its own design: k = wn^2, d = 2 zeta wn, alpha passed through.
pub fn gains(wn: f64, zeta: f64, alpha: f64) -> PlaneGains {
    PlaneDesign::new(wn, zeta, alpha).gains(0.0, 0.0)
}

/// The tracking law alone: the law's two poles, with no integral tier.
pub fn num_pd(g: PlaneGains, e: f64, v: f64, v_ref: f64) -> f64 {
    g.kappa_p * e - g.kappa_d * (v - v_ref)
}

/// THE law: the numerator for one plane, from e, v, v_ref, i_acc in that order.
/// `g` carries the EFFECTIVE triple, so any scaling is stated in the gains handed over.
pub fn num(g: PlaneGains, e: f64, v: f64, v_ref: f64, i_acc: f64) -> f64 {
    num_pd(g, e, v, v_ref) + g.kappa_i * i_acc
}

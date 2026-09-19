// law.rs — THE LAW, as one expression and no realization of it:
//   num_i = k_i e_i - d_i (v_i - v_ref_i) + alpha_i i_acc_i,
// with its two design inputs named where they are built — the pole placement (k = wn^2 - k_eff,
// d = 2 zeta wn - b_eff) in design.rs, the metric Lambda = (J M^-1 J^T)^-1 in inertia.rs.
//
// WHY THIS FILE EXISTS: the expression above was written out by hand at every call site — the arm's
// three readings (task_loop.rs: step_joint, step_points, step_ff) and the biped's CoM loop
// (simu's src/standing_loop.rs) — and hand-copied laws are as many laws as there are copies. ONE LAW,
// TWO REALIZATIONS: the arm fixes its base and maps the plane numerator through J', the biped floats
// and must produce it through ground contact, so what a loop adds around the law (a contact schedule's
// soft factor, a passivity floor, an acceleration feedforward, the biped's pressure-point offset)
// stays in the loop that adds it and never here.

use crate::design::PlaneDesign;
use crate::gains::PlaneGains;

/// gains is the pole placement for a loop that states its own design (wn, zeta, alpha) rather than
/// its plant's stiffness and damping: k = wn^2, d = 2 zeta wn, alpha passed through. It is
/// `PlaneDesign::gains` with k_eff = b_eff = 0, which is what a loop that removes both by FEEDFORWARD
/// asks for (GA_PID_AUDIT.md #4/#5) and what both realizations do.
pub fn gains(wn: f64, zeta: f64, alpha: f64) -> PlaneGains {
    PlaneDesign::new(wn, zeta, alpha).gains(0.0, 0.0)
}

/// num_pd is the law's two poles without its integral tier: the same expression with alpha_i i_acc_i
/// left off, for a caller that evaluates the tracking law alone (the biped's pressure-point reference
/// carries the error and the velocity, and adds no zero term to say so).
pub fn num_pd(g: PlaneGains, e: f64, v: f64, v_ref: f64) -> f64 {
    g.kappa_p * e - g.kappa_d * (v - v_ref)
}

/// num is THE law: one plane's demanded numerator from its error, its velocity, the reference
/// velocity it tracks and its integral accumulator, in that order of terms. `g` carries the plane's
/// EFFECTIVE triple, so a caller that scales or floors the design (the contact schedule's soft
/// factor, `passivity_floor`) states the scaling in the gains it hands over rather than inside the
/// law: the product is the same and the law stays one expression.
pub fn num(g: PlaneGains, e: f64, v: f64, v_ref: f64, i_acc: f64) -> f64 {
    num_pd(g, e, v, v_ref) + g.kappa_i * i_acc
}

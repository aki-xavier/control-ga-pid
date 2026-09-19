// window.rs — MotionWindow: the theory's window. `feasible` means the window is non-empty, NOT machine feasibility.

/// Damping from the overshoot budget, bandwidth floor from settling, sampling ceiling 0.2/dt.
/// The lower bound uses the same rule as MotionBudget; only the ceilings differ (sampling vs actuators).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MotionWindow {
    pub zeta: f64,
    pub wn_lo: f64,
    pub wn_hi: f64,
    pub wn: f64,
    pub feasible: bool,
    pub reason: String,
}

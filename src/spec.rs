// spec.rs — TaskSpec, the task-class input to the design.
//
// The three tiers map to the three knobs independently (motion -> wn/zeta, disturbance -> ti and
// e_deadband, feedforward -> lead), so one flat gain set cannot serve both ends of the range; the
// TaskClass that produces specs lives in class.rs.

use crate::design::{alpha_for_ti, PlaneDesign};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TaskSpec {
    pub wn: f64,
    pub zeta: f64,
    pub ti: f64,
    pub e_deadband: f64,
    pub lead: bool,
}

impl TaskSpec {
    /// design turns the spec into the per-plane tuple: kp/kd from the motion tier, kappa_i from the disturbance tier.
    pub fn design(&self) -> PlaneDesign {
        PlaneDesign::new(self.wn, self.zeta, alpha_for_ti(self.wn, self.ti))
    }
}

// class.rs — TaskClass: the spec a task class fixes, from which window/spec derive the gains.

use crate::design::{wn_for_settling, zeta_from_overshoot};
use crate::spec::TaskSpec;
use crate::window::MotionWindow;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TaskClass {
    /// overshoot budget as a fraction of the step
    pub mp: f64,
    /// 2 percent settling time [s]
    pub ts: f64,
    /// integral time constant; <= 0 leaves the integral out
    pub ti: f64,
    /// integrate only inside this error band; 0 disables the gate
    pub e_deadband: f64,
    /// carry the reference velocity and acceleration through J+
    pub lead: bool,
}

impl TaskClass {
    /// Bounds the theory supports at control period dt; recommends the settling floor, not the ceiling.
    pub fn window(&self, dt: f64) -> MotionWindow {
        let zeta = zeta_from_overshoot(self.mp);
        let lo = wn_for_settling(zeta, self.ts);
        // wn * dt = 0.2 keeps the discrete poles inside the unit circle.
        let hi = if dt > 0.0 { 0.2 / dt } else { 0.0 };
        let ok = lo > 0.0 && lo <= hi;
        let mut why = String::new();
        if !ok {
            why = format!("settling bound wn={lo:.1} exceeds the sampling ceiling wn={hi:.1}");
        }
        MotionWindow {
            zeta,
            wn_lo: lo,
            wn_hi: hi,
            wn: if ok { lo } else { 0.0 },
            feasible: ok,
            reason: why,
        }
    }

    /// Resolves the class at its window point; an infeasible window comes back zeroed (check feasible first).
    pub fn spec(&self, dt: f64) -> TaskSpec {
        let w = self.window(dt);
        TaskSpec {
            wn: w.wn,
            zeta: w.zeta,
            ti: self.ti,
            e_deadband: self.e_deadband,
            lead: self.lead,
        }
    }
}

/// Fine manipulation: tight overshoot, integral on.
pub fn class_precision_hand() -> TaskClass {
    TaskClass {
        mp: 0.02,
        ts: 0.5,
        ti: 1.0,
        e_deadband: 0.0,
        lead: false,
    }
}

/// Fast motion: overshoot tolerated, short settling, no integral, feedforward on.
pub fn class_fast_swing() -> TaskClass {
    TaskClass {
        mp: 0.10,
        ts: 0.2,
        ti: 0.0,
        e_deadband: 0.0,
        lead: true,
    }
}

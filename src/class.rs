// class.rs — TaskClass, what a task class actually fixes: the spec from which the gains follow
// (storing the gains themselves per class is not supported by measurement); window/spec resolve it.

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
    /// window solves the bounds the theory supports at a control period and recommends the floor rather
    /// than the ceiling: near the boundary a stiffer loop can lose a target it would otherwise reach.
    pub fn window(&self, dt: f64) -> MotionWindow {
        let zeta = zeta_from_overshoot(self.mp);
        let lo = wn_for_settling(zeta, self.ts);
        // wn * dt = 0.2 keeps the discrete poles well inside the unit circle and the velocity estimate usable.
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

    /// spec resolves the class at its window point; an empty window comes back zeroed, so check feasible first.
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

/// class_precision_hand is the fine-manipulation class: tight overshoot (contact force scales with it), integral removes drift.
pub fn class_precision_hand() -> TaskClass {
    TaskClass {
        mp: 0.02,
        ts: 0.5,
        ti: 1.0,
        e_deadband: 0.0,
        lead: false,
    }
}

/// class_fast_swing is the fast-motion class: overshoot tolerable, short settling, no integral (windup), feedforward.
pub fn class_fast_swing() -> TaskClass {
    TaskClass {
        mp: 0.10,
        ts: 0.2,
        ti: 0.0,
        e_deadband: 0.0,
        lead: true,
    }
}

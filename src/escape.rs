// escape.rs — TaskAvoidance reference shaping: only the safe goal reaches the law; lead evaluates each keep-out at c + vc * tau.

use crate::keepout::Keepout;
use crate::sphere_ko::zero_sphere;
use control_math::vec3::Vec3;

#[derive(Clone, Copy, Debug, Default)]
pub struct TaskAvoidance;

impl TaskAvoidance {
    /// Projects goal out of keepouts; returns (safe goal, min segment distance, active).
    pub fn safe_target(
        &self,
        cur: Vec3,
        goal: Vec3,
        keepouts: &[Keepout],
        margin: f64,
    ) -> (Vec3, f64, bool) {
        self.safe_target_lead(cur, goal, keepouts, margin, 0.0, &[])
    }

    pub fn safe_target_lead(
        &self,
        cur: Vec3,
        goal: Vec3,
        keepouts: &[Keepout],
        margin: f64,
        tau: f64,
        vc: &[Vec3],
    ) -> (Vec3, f64, bool) {
        let mut klist = vec![Keepout::Sphere(zero_sphere()); keepouts.len()];
        for i in 0..keepouts.len() {
            klist[i] = keepouts[i];
            if tau > 0.0 && vc.len() == keepouts.len() {
                klist[i] = klist[i].shift(vc[i].scale(tau));
            }
        }
        let mut best_dist = 1e30;
        for k in &klist {
            let d = k.seg_min_signed(cur, goal);
            if d < best_dist {
                best_dist = d;
            }
        }
        let mut candidate = goal;
        for _ in 0..8 {
            let mut best = candidate;
            let mut best_viol = 0.0;
            for k in &klist {
                let s = k.safe_point(cur, candidate, margin);
                let viol = s.sub(candidate).norm();
                if viol > best_viol {
                    best_viol = viol;
                    best = s;
                }
            }
            if best_viol < 1e-9 {
                break;
            }
            candidate = best;
        }
        let active = candidate.sub(goal).norm() > 1e-9;
        (candidate, best_dist, active)
    }
}

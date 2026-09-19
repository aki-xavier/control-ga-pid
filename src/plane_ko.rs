// plane_ko.rs — PlaneKo, the half-space keep-out primitive (free side n.(x - p) > 0), part of
// keepout.rs's sum type. Readouts normalize n; a degenerate normal gives no constraint, not NaN.

use control_math::vec3::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneKo {
    pub n: Vec3,
    pub p: Vec3,
}

pub(crate) fn plane_unit_normal(n: Vec3) -> Vec3 {
    let l2 = n.dot(n);
    if l2 < 1e-24 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    n.scale(1.0 / l2.sqrt())
}

impl PlaneKo {
    pub(crate) fn plane_point(&self, goal: Vec3, margin: f64) -> Vec3 {
        let nn = plane_unit_normal(self.n);
        let sg = nn.dot(goal.sub(self.p));
        if sg >= margin {
            return goal;
        }
        goal.add(nn.scale(margin - sg))
    }
}

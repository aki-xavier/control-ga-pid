// sphere_ko.rs — SphereKo, the spherical keep-out primitive; part of the Keepout sum type.

use control_math::vec3::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereKo {
    pub c: Vec3,
    pub r: f64,
}

impl SphereKo {
    pub(crate) fn sphere_point(&self, cur: Vec3, goal: Vec3, margin: f64) -> Vec3 {
        let rb = self.r + margin;
        if seg_dist(cur, goal, self.c) >= rb {
            return goal;
        }
        let l = cur.sub(self.c).norm();
        let u = cur.sub(self.c).scale(1.0 / l.max(1e-12));
        let w = goal.sub(self.c).sub(u.scale(u.dot(goal.sub(self.c))));
        let mut bt = w.normalized();
        if bt.norm() < 1e-9 {
            bt = u.perp();
        }
        if l <= rb {
            // inside the inflated sphere: creep toward the free side
            return self
                .c
                .add(u.scale(rb * 0.08f64.cos()))
                .add(bt.scale(rb * 0.08f64.sin()));
        }
        let phi = (rb / l).min(1.0).acos();
        let t1 = self
            .c
            .add(u.scale(rb * phi.cos()))
            .add(bt.scale(rb * phi.sin()));
        let t2 = self
            .c
            .add(u.scale(rb * phi.cos()))
            .sub(bt.scale(rb * phi.sin()));
        let gt = goal.sub(self.c);
        if t1.sub(self.c).dot(gt) >= t2.sub(self.c).dot(gt) {
            return t1;
        }
        t2
    }
}

pub(crate) fn seg_dist(a: Vec3, b: Vec3, p: Vec3) -> f64 {
    let ab = b.sub(a);
    let l2 = ab.dot(ab);
    if l2 < 1e-12 {
        return p.sub(a).norm();
    }
    let t = ((p.sub(a).dot(ab)) / l2).clamp(0.0, 1.0);
    p.sub(a.add(ab.scale(t))).norm()
}

pub(crate) fn zero_sphere() -> SphereKo {
    SphereKo {
        c: Vec3::new(0.0, 0.0, 0.0),
        r: 0.0,
    }
}

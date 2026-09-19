// box_ko.rs — BoxKo, the axis-aligned box keep-out [lo, hi]; part of the Keepout sum type.

use control_math::vec3::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxKo {
    pub lo: Vec3,
    pub hi: Vec3,
}

impl BoxKo {
    pub(crate) fn box_point(&self, cur: Vec3, goal: Vec3, margin: f64) -> Vec3 {
        let (dmin, _) = box_seg_dist(cur, goal, self);
        if dmin >= margin {
            return goal;
        }
        let ctr = self.lo.add(self.hi).scale(0.5);
        let mut corner = self.hi;
        if goal.x <= ctr.x {
            corner.x = self.lo.x;
        }
        if goal.y <= ctr.y {
            corner.y = self.lo.y;
        }
        if goal.z <= ctr.z {
            corner.z = self.lo.z;
        }
        let dir = corner.sub(ctr).normalized();
        let ext = corner.sub(ctr).norm();
        ctr.add(dir.scale(ext + margin))
    }

    /// Outward normal and face point of the least-penetrating box wall at p.
    pub fn binding_face(&self, p: Vec3) -> (Vec3, Vec3) {
        let norms = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let vals = [
            self.hi.x, -self.lo.x, self.hi.y, -self.lo.y, self.hi.z, -self.lo.z,
        ];
        let mut best = -1e30;
        let mut nn = norms[0];
        let mut pp = norms[0].scale(vals[0]);
        for f in 0..6 {
            let face_pt = norms[f].scale(vals[f]);
            let s = norms[f].dot(p.sub(face_pt));
            if s > best {
                best = s;
                nn = norms[f];
                pp = face_pt;
            }
        }
        (nn, pp)
    }
}

/// Segment sampled at 41 points: (minimum signed distance, point at argmin).
pub(crate) fn box_seg_dist(a: Vec3, b: Vec3, k: &BoxKo) -> (f64, Vec3) {
    let mut best_sd = 1e30;
    let mut best_q = a;
    for i in 0..41 {
        let t = i as f64 / 40.0;
        let x = a.add(b.sub(a).scale(t));
        let sd = box_min_binding_sd(x, k);
        if sd < best_sd {
            best_sd = sd;
            best_q = x;
        }
    }
    (best_sd, best_q)
}

pub(crate) fn box_min_binding_sd(x: Vec3, k: &BoxKo) -> f64 {
    let mut m = x.x - k.hi.x;
    let c2 = k.lo.x - x.x;
    if c2 > m {
        m = c2;
    }
    let c3 = x.y - k.hi.y;
    if c3 > m {
        m = c3;
    }
    let c4 = k.lo.y - x.y;
    if c4 > m {
        m = c4;
    }
    let c5 = x.z - k.hi.z;
    if c5 > m {
        m = c5;
    }
    let c6 = k.lo.z - x.z;
    if c6 > m {
        m = c6;
    }
    m
}

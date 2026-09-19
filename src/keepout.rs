// keepout.rs — Keepout, the union of the supported convex keep-out primitives (sphere / half-space plane /
// AAB box) with signed distance, avoidance sub-target shaping and the whole-arm escape push-out. Pure
// Euclidean geometry, no control dependency; signed_dist(x, k) > 0 means x is outside (free), < 0 penetrating.

use crate::box_ko::{box_min_binding_sd, box_seg_dist, BoxKo};
use crate::plane_ko::{plane_unit_normal, PlaneKo};
use crate::sphere_ko::{seg_dist, SphereKo};
use control_math::vec3::Vec3;

pub const BODY_LINK_RADIUS: f64 = 0.045;

/// BODY_LINK_RADII: measured z1 mesh half-thickness per chain link, index i covering [link0i, link0i+1], the last the tool flange beyond link06.
pub const BODY_LINK_RADII: [f64; 6] = [0.066, 0.0613, 0.0583, 0.0472, 0.0451, 0.0325];

/// TOOL_REACH: terminal tool-flange reach beyond link06 (collision cylinder: 0.051 long on the link local +x, radius 0.0325).
pub const TOOL_REACH: f64 = 0.051;

/// TOOL_ENVELOPE: max distance from the flange end-plane center to any tool point (back rim), for orientation-free shaping.
pub const TOOL_ENVELOPE: f64 = 0.060475;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Keepout {
    Sphere(SphereKo),
    Plane(PlaneKo),
    Box(BoxKo),
}

impl Keepout {
    pub fn signed_dist(&self, x: Vec3) -> f64 {
        match self {
            Keepout::Sphere(k) => x.sub(k.c).norm() - k.r,
            Keepout::Plane(k) => plane_unit_normal(k.n).dot(x.sub(k.p)),
            Keepout::Box(k) => box_min_binding_sd(x, k),
        }
    }

    /// shift translates the keep-out by dv (velocity look-ahead: obstacle at c + v * tau is represented by shifting the geometry).
    pub fn shift(&self, dv: Vec3) -> Keepout {
        match self {
            Keepout::Sphere(k) => Keepout::Sphere(SphereKo {
                c: k.c.add(dv),
                r: k.r,
            }),
            Keepout::Plane(k) => Keepout::Plane(PlaneKo {
                n: k.n,
                p: k.p.add(dv),
            }),
            Keepout::Box(k) => Keepout::Box(BoxKo {
                lo: k.lo.add(dv),
                hi: k.hi.add(dv),
            }),
        }
    }

    /// safe_point returns a sub-target on the free side of k with margin, given the current position cur and the goal.
    pub fn safe_point(&self, cur: Vec3, goal: Vec3, margin: f64) -> Vec3 {
        match self {
            Keepout::Sphere(k) => k.sphere_point(cur, goal, margin),
            Keepout::Plane(k) => k.plane_point(goal, margin),
            Keepout::Box(k) => k.box_point(cur, goal, margin),
        }
    }

    /// seg_min_signed: minimum signed distance of segment a-b to k (>0: fully free).
    pub fn seg_min_signed(&self, a: Vec3, b: Vec3) -> f64 {
        match self {
            Keepout::Sphere(k) => seg_dist(a, b, k.c) - k.r,
            Keepout::Plane(k) => {
                let nn = plane_unit_normal(k.n);
                let sa = nn.dot(a.sub(k.p));
                let sb = nn.dot(b.sub(k.p));
                if sa < sb {
                    return sa;
                }
                sb
            }
            Keepout::Box(k) => {
                let (best, _) = box_seg_dist(a, b, k);
                best
            }
        }
    }

    /// escape returns the push-out vector for p against k inflated by r_link (link body radius, capsule approximation), zero when clear by margin.
    pub fn escape(&self, p: Vec3, r_link: f64, margin: f64) -> Vec3 {
        match self {
            Keepout::Sphere(k) => {
                let d = p.sub(k.c).norm();
                let need = k.r + r_link + margin;
                if d >= need {
                    return Vec3::new(0.0, 0.0, 0.0);
                }
                let dir = if d > 1e-9 {
                    p.sub(k.c).scale(1.0 / d)
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                dir.scale(need - d)
            }
            Keepout::Plane(k) => {
                let l = k.n.dot(k.n).sqrt();
                let nn = k.n.scale(1.0 / l);
                let s = nn.dot(p.sub(k.p));
                let need = r_link + margin;
                if s >= need {
                    return Vec3::new(0.0, 0.0, 0.0);
                }
                nn.scale(need - s)
            }
            Keepout::Box(k) => {
                let infl = BoxKo {
                    lo: k.lo.sub(Vec3::new(r_link, r_link, r_link)),
                    hi: k.hi.add(Vec3::new(r_link, r_link, r_link)),
                };
                let (nn, pp) = infl.binding_face(p);
                let s = nn.dot(p.sub(pp));
                let need = margin;
                if s >= need {
                    return Vec3::new(0.0, 0.0, 0.0);
                }
                nn.scale(need - s)
            }
        }
    }
}

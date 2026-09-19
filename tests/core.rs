// core.rs — unit tests for the plane_design / bridge / keepout building blocks of the GA-PID
// core. The math half lives in control-math's own tests/math.rs, which is that layer's home.

use control_ga_pid::box_ko::BoxKo;
use control_ga_pid::budget::{budget_verdict, MotionBudget};
use control_ga_pid::class::{class_fast_swing, class_precision_hand, TaskClass};
use control_ga_pid::design::PlaneDesign;
use control_ga_pid::design::{alpha_for_ti, zeta_from_overshoot};
use control_ga_pid::escape::TaskAvoidance;
use control_ga_pid::keepout::Keepout;
use control_ga_pid::plane_ko::PlaneKo;
use control_ga_pid::sphere_ko::SphereKo;
use control_math::mat::Mat;
use control_math::quat::Quat;
use control_math::task_space_bridge::TaskSpaceBridge;
use control_math::vec3::Vec3;
use control_model::pga_dynamics::PgaDynamicsModel;
use control_model::urdf::{home_q, load_urdf_chain, urdf_path};
use std::f64::consts::PI;

#[test]
fn mul_and_vec() {
    let a = Mat::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let x = [1.0, 1.0, 1.0];
    let y = a.mul_vec(&x);
    assert_eq!(y.len(), 2);
    assert!((y[0] - 6.0).abs() < 1e-12);
    assert!((y[1] - 15.0).abs() < 1e-12);
    let at = a.transposed();
    assert!(at.rows == 3);
    assert!(at.cols == 2);
}

#[test]
fn quat_rotvec_between_pi() {
    // rotation of pi about z maps to the rotation vector pi*z
    let q = Quat {
        w: 0.0,
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    let w = Quat::rotvec_between(q, Quat::IDENTITY);
    assert!((w.z - PI).abs() < 1e-9);
    assert!(w.x.abs() < 1e-12);
    assert!(w.y.abs() < 1e-12);
}

#[test]
fn plane_design_gains() {
    let d = PlaneDesign::new(15.0, 0.9, 30.0);
    let pg = d.gains(0.0, 0.0);
    assert!((pg.kappa_p - 225.0).abs() < 1e-12);
    assert!((pg.kappa_d - 27.0).abs() < 1e-12);
    assert!((pg.kappa_i - 30.0).abs() < 1e-12);
    let pg2 = d.gains(5.0, 2.0);
    assert!((pg2.kappa_p - 220.0).abs() < 1e-12);
    assert!((pg2.kappa_d - 25.0).abs() < 1e-12);
}

#[test]
fn plane_design_integral_from_time_constant() {
    // alpha = kp / Ti = wn^2 / Ti; a non-positive Ti switches the integral off
    assert!((alpha_for_ti(15.0, 1.0) - 225.0).abs() < 1e-12);

    assert!((alpha_for_ti(15.0, 2.0) - 112.5).abs() < 1e-12);
    assert_eq!(alpha_for_ti(15.0, 0.0), 0.0);
    let d = PlaneDesign::with_ti(15.0, 0.9, 1.0);
    let pg = d.gains(0.0, 0.0);
    assert!((pg.kappa_i - 225.0).abs() < 1e-12);
    assert!((pg.kappa_p - 225.0).abs() < 1e-12);
}

#[test]
fn zeta_from_overshoot_hits_the_textbook_points() {
    // inverted relation: a zero budget is deadbeat (zeta = 1), 16.3 percent is the classic 0.5
    assert!((zeta_from_overshoot(0.0) - 1.0).abs() < 1e-15);
    let z = zeta_from_overshoot((-(PI * 0.5) / (1.0f64 - 0.25).sqrt()).exp());
    assert!(
        (z - 0.5).abs() < 1e-12,
        "Mp for zeta = 0.5 inverted to zeta = {z}"
    );
    assert!(zeta_from_overshoot(0.02) > zeta_from_overshoot(0.10));
}

#[test]
fn task_class_derives_the_motion_tier() {
    // the class fixes the spec and the spec derives the gains: a tighter overshoot budget buys
    // damping, a shorter settling requirement buys bandwidth, the integral tier the disturbance class
    let ph = class_precision_hand();
    let fs = class_fast_swing();
    let wp = ph.window(1e-3);
    let wf = fs.window(1e-3);
    assert!(wp.feasible);
    assert!(wf.feasible);
    assert!(wp.zeta > wf.zeta);
    assert!(wf.wn > wp.wn);
    assert!((wp.zeta - 0.78).abs() < 0.01, "zeta = {}", wp.zeta);
    assert!((wp.wn - 10.3).abs() < 0.1, "wn = {}", wp.wn);
    assert!((wf.wn - 33.8).abs() < 0.2, "wn = {}", wf.wn);
    assert!(ph.spec(1e-3).design().alpha > 0.0);
    assert_eq!(fs.spec(1e-3).design().alpha, 0.0);
    assert!(fs.spec(1e-3).lead);
    assert!(!ph.spec(1e-3).lead);
    let hard = TaskClass {
        mp: 0.0,
        ts: 1e-5,
        ti: 0.0,
        e_deadband: 0.0,
        lead: false,
    };
    assert!(!hard.window(1e-3).feasible);
    assert_eq!(hard.window(1e-3).wn, 0.0);
    assert!(!hard.window(1e-3).reason.is_empty());
}

#[test]
fn bridge_damped_least_squares() {
    // 2x3 J with lam = 0: dq = J^T e, the right-inverse minimal-norm solution
    let j = Mat::from_rows(&[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
    let b = TaskSpaceBridge::new(3, 0.0);
    let dq = b.step(&j, &[2.0, 3.0]);
    assert_eq!(dq.len(), 3);
    assert!((dq[0] - 2.0).abs() < 1e-12);
    assert!((dq[1] - 3.0).abs() < 1e-12);
    assert!(dq[2].abs() < 1e-12);
    // Lam damped: full-rank 3x3 with ridge regularization; dq = (J'J+li)-1 J'e.
    let j2 = Mat::eye(3);
    let b2 = TaskSpaceBridge::new(3, 0.1);
    let dq2 = b2.step(&j2, &[1.0, 0.0, 0.0]);
    assert!((dq2[0] - 1.0 / 1.1).abs() < 1e-12);
}

#[test]
fn keepout_sphere_signed_dist() {
    let ko = Keepout::Sphere(SphereKo {
        c: Vec3::new(0.0, 0.0, 0.0),
        r: 1.0,
    });
    let x = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 2.5,
    };
    assert!((ko.signed_dist(x) - 1.5).abs() < 1e-12);
    let xi = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.25,
    };
    assert!((ko.signed_dist(xi) + 0.75).abs() < 1e-12);
}

#[test]
fn keepout_plane_and_box() {
    let pl = Keepout::Plane(PlaneKo {
        n: Vec3::new(0.0, 1.0, 0.0),
        p: Vec3::new(0.0, 1.0, 0.0),
    });
    assert!((pl.signed_dist(Vec3::new(0.0, 2.0, 0.0)) - 1.0).abs() < 1e-12);
    assert!((pl.signed_dist(Vec3::new(0.0, 0.5, 0.0)) + 0.5).abs() < 1e-12);
    // `box` is a Rust keyword, so the local is renamed; the value is the same
    let bx = Keepout::Box(BoxKo {
        lo: Vec3::default(),
        hi: Vec3::new(1.0, 1.0, 1.0),
    });
    assert!((bx.signed_dist(Vec3::new(2.0, 0.5, 0.5)) - 1.0).abs() < 1e-12);
    assert!((bx.signed_dist(Vec3::new(0.5, 0.5, 0.5)) + 0.5).abs() < 1e-12);
}

#[test]
fn keepout_shift_and_safe_point() {
    let ko = Keepout::Sphere(SphereKo {
        c: Vec3::new(1.0, 0.0, 0.0),
        r: 0.5,
    });
    let ko2 = ko.shift(Vec3::new(2.0, 0.0, 0.0));
    match ko2 {
        Keepout::Sphere(s) => assert!((s.c.x - 3.0).abs() < 1e-12),
        _ => panic!("shift changed the variant"),
    }
    // a goal on the far side of a big sphere is pulled to the free side
    let big = Keepout::Sphere(SphereKo {
        c: Vec3::new(0.0, 0.0, 0.0),
        r: 1.0,
    });
    let cur = Vec3 {
        x: -2.0,
        y: 0.0,
        z: 0.0,
    };
    let goal = Vec3 {
        x: 2.0,
        y: 0.0,
        z: 0.0,
    };
    let s = big.safe_point(cur, goal, 0.05);
    assert!(s.x < 0.0); // stays on the near (free) side
    match big {
        Keepout::Sphere(sp) => assert!(s.sub(sp.c).norm() > 1.0 + 0.05 - 1e-9),
        _ => panic!("not a sphere"),
    }
}

#[test]
fn task_avoidance_projection() {
    let t = TaskAvoidance;
    let ko = Keepout::Sphere(SphereKo {
        c: Vec3::new(0.0, 0.0, 0.0),
        r: 0.5,
    });
    let cur = Vec3 {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    };
    let goal = Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    let (safe, dist, active) = t.safe_target(cur, goal, &[ko], 0.05);
    assert!(active);
    assert!(dist < 0.0); // segment crosses the keep-out
    assert!(safe.x < 0.01); // projected back to the free side
}

/// The keep-out escape an inflated shape asks for — the one keepout.rs entry point this file did not
/// otherwise cover. The link Jacobian against finite differences of FK lives in
/// `../control-model/tests/urdf.rs`.
#[test]
fn keepout_escape_inflated() {
    let c = Vec3::new(0.0, 0.0, 0.0);
    let k = Keepout::Sphere(SphereKo { c, r: 0.05 });
    let e = k.escape(Vec3::new(0.06, 0.0, 0.0), 0.045, 0.005);
    assert!((e.norm() - (0.05 + 0.045 + 0.005 - 0.06)).abs() < 1e-9);
    assert!(e.x > 0.0);
    let e2 = k.escape(Vec3::new(0.2, 0.0, 0.0), 0.045, 0.005);
    assert!(e2.norm() < 1e-12);
    // plane: n = +x, wall at x = 0.30, p at x = 0.20 -> push 0.15
    let pk = Keepout::Plane(PlaneKo {
        n: Vec3::new(1.0, 0.0, 0.0),
        p: Vec3::new(0.3, 0.0, 0.0),
    });
    let e3 = pk.escape(Vec3::new(0.20, 0.0, 0.0), 0.045, 0.005);
    assert!((e3.x - 0.15).abs() < 1e-9);
    // box: point inside the inflated box -> nonzero push to margin beyond wall
    let bk = Keepout::Box(BoxKo {
        lo: Vec3::new(0.15, -0.03, -0.03),
        hi: Vec3::new(0.20, 0.03, 0.03),
    });
    let e4 = bk.escape(Vec3::new(0.16, 0.0, 0.0), 0.045, 0.005);
    assert!(e4.norm() > 0.005);
}

/// THE BUDGET'S BINDING CHANNEL NAMES A JOINT, NOT A SLOT: `motion_budget` scans joints by index
/// and keeps answering "joint {i}" (../z1-arm/tests/bench_sims.rs pins that reading), while a caller
/// that HAS the model's names resolves them through `with_names`. Pinned: the resolved name, the
/// kept fallback, the non-answer for a budget that binds nothing, and no re-resolution.
#[test]
fn the_budget_reads_the_binding_joint_through_the_models_names() {
    let index_form = MotionBudget {
        zeta_lo: 0.780,
        wn_lo: 10.26,
        wn_hi: 8.00,
        binding: "joint 1".to_string(),
        feasible: false,
        names: Vec::new(),
    };
    assert_eq!(index_form.binding_index(), Some(1));
    assert_eq!(index_form.binding, "joint 1", "the unnamed reading moved");
    // the model's own names, in the scan's own order: slot 1 is the second joint
    let names = vec![
        "joint1".to_string(),
        "joint2".to_string(),
        "joint3".to_string(),
        "joint4".to_string(),
        "joint5".to_string(),
        "joint6".to_string(),
    ];
    let named = index_form.clone().with_names(&names);
    assert_eq!(
        named.binding, "joint2",
        "the named reading is not the joint"
    );
    assert_eq!(named.names, names, "the names did not stay on the value");
    assert_eq!(named.feasible, index_form.feasible, "the verdict moved");
    assert_eq!(named.binding_index(), None, "a resolved name re-resolved");
    // a list too short for the index keeps the scan's own words, and a budget that binds nothing
    // is not an index at all
    let short = index_form.clone().with_names(&["joint1".to_string()]);
    assert_eq!(short.binding, "joint 1");
    let none = MotionBudget {
        binding: "none".to_string(),
        ..index_form.clone()
    };
    assert_eq!(none.binding_index(), None);
    assert_eq!(none.with_names(&names).binding, "none");
}

/// The Z1 at home as the arm's own run path sees it (../z1-arm/src/bench/bench_sims.rs reads these
/// terms off a model-only plant): the PGA model and chain, engine-free, plus the identified gravity
/// stiffness the bench's `joint_k_eff` takes by central differences.
fn z1_home_model() -> (Mat, Mat, Vec<f64>, Vec<f64>, Vec<String>) {
    let chain = load_urdf_chain(&urdf_path(), "link00", "link06").expect("the Z1 chain");
    let mut pdyn = PgaDynamicsModel::new(chain.clone());
    let q = home_q();
    let m = pdyn.mass_matrix(&q);
    // the tip Jacobian through the chain's own FK, which is the frame set the model's cached
    // `frames` builds (pga_dynamics.rs) and therefore the Jacobian the bench's plant reports
    let (o, r) = chain.fk(&q);
    let tip = chain.tip_pose(&o, &r).0;
    let j = chain.point_jacobian(&o, &r, tip);
    let g = pdyn.gravity_torques(&q);
    let md = m.diag();
    let eps = 1e-5;
    let mut k_eff = vec![0.0; 6];
    for i in 0..6 {
        let mut qp = q.clone();
        qp[i] += eps;
        let gp = pdyn.gravity_torques(&qp);
        let mut qm = q.clone();
        qm[i] -= eps;
        let gm = pdyn.gravity_torques(&qm);
        let mi = if md[i] > 1e-9 { md[i] } else { 1e-9 };
        k_eff[i] = (gp[i] - gm[i]) / (2.0 * eps) / mi;
    }
    (m, j, g, k_eff, chain.joint_names.clone())
}

/// THE ARM'S OWN RUN PATH ASKS THE BUDGET, AND THE VERDICT NAMES THE JOINT: the arm has no runtime
/// (the arm's benches in `src/bench/` and its examples are the run path), so the caller is that
/// crate's `bench_sims::sim_setpoint_taskloop`, which asks this module's `budget_verdict` before its
/// first command and RECORDS it, never enforced (the legs' guard ships off, the arm has no guard
/// switch).
///
/// The reading pinned here is the T5 arm's: a 0.25 m step with the study's [30, 6, ...] N.m ceiling
/// saturates the SECOND joint and the verdict names it, while the shipped T1 arm states no ceiling,
/// so nothing binds and the same design sits inside its window.
#[test]
fn the_arms_run_path_records_the_verdict_and_names_the_binding_joint() {
    let (m, j, g, k_eff, names) = z1_home_model();
    let lim = vec![30.0, 6.0, 30.0, 30.0, 30.0, 30.0];
    let fast = budget_verdict(
        &m,
        &j,
        &g,
        &k_eff,
        &lim,
        &[0.25, 0.0, -0.10],
        &names,
        15.0,
        0.9,
    );
    // the design's own settling requirement is the floor: Ts = 4/(zeta wn) inverts back to wn
    assert!(
        (fast.wn_lo - 15.0).abs() < 1e-9,
        "the floor is {} against the design's own 15",
        fast.wn_lo
    );
    assert_eq!(
        fast.binding, "joint2",
        "the binding joint is {}",
        fast.binding
    );
    assert!(
        (fast.wn_hi - 8.0).abs() < 0.5,
        "the ceiling is {}",
        fast.wn_hi
    );
    assert!(
        !fast.feasible,
        "the saturated T5 arm reads feasible: wn_hi {} against wn_lo {}",
        fast.wn_hi, fast.wn_lo
    );
    // the shipped T1 arm: no ceiling stated, so nothing binds and the design's window is open
    let t1 = budget_verdict(
        &m,
        &j,
        &g,
        &k_eff,
        &[],
        &[0.10, 0.06, -0.05],
        &names,
        15.0,
        0.9,
    );
    assert!(t1.feasible);
    assert_eq!(t1.binding, "none");
    println!(
        "arm budget: T5 binding {} at wn in [{:.1}, {:.1}] (feasible {}); T1 binding {} (feasible {})",
        fast.binding, fast.wn_lo, fast.wn_hi, fast.feasible, t1.binding, t1.feasible
    );
}

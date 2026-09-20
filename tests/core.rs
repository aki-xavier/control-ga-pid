// Unit tests for the plane-design, least-squares and keep-out building blocks.

use control_ga_pid::box_ko::BoxKo;
use control_ga_pid::budget::{budget_verdict, MotionBudget};
use control_ga_pid::class::{class_fast_swing, class_precision_hand, TaskClass};
use control_ga_pid::design::PlaneDesign;
use control_ga_pid::design::{alpha_for_ti, zeta_from_overshoot};
use control_ga_pid::escape::TaskAvoidance;
use control_ga_pid::keepout::Keepout;
use control_ga_pid::plane_ko::PlaneKo;
use control_ga_pid::sphere_ko::SphereKo;
use control_math::lstsq::DampedLstsq;
use control_math::mat::Mat;
use control_math::quat::Quat;
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
    // a non-positive Ti switches the integral off
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
    // inverted relation: a zero budget is deadbeat (zeta = 1)
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
    // tighter Mp buys damping, shorter Ts buys bandwidth
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
fn damped_least_squares() {
    // 2x3 J with lam = 0: dq = J^T e, the right-inverse minimal-norm solution
    let j = Mat::from_rows(&[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
    let b = DampedLstsq::new(3, 0.0);
    let dq = b.solve(&j, &[2.0, 3.0]);
    assert_eq!(dq.len(), 3);
    assert!((dq[0] - 2.0).abs() < 1e-12);
    assert!((dq[1] - 3.0).abs() < 1e-12);
    assert!(dq[2].abs() < 1e-12);
    let j2 = Mat::eye(3);
    let b2 = DampedLstsq::new(3, 0.1);
    let dq2 = b2.solve(&j2, &[1.0, 0.0, 0.0]);
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
    // `box` is a Rust keyword, so the local is renamed
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
    assert!(s.x < 0.0);
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
    assert!(dist < 0.0);
    assert!(safe.x < 0.01);
}

/// The keep-out escape an inflated shape asks for.
#[test]
fn keepout_escape_inflated() {
    let c = Vec3::new(0.0, 0.0, 0.0);
    let k = Keepout::Sphere(SphereKo { c, r: 0.05 });
    let e = k.escape(Vec3::new(0.06, 0.0, 0.0), 0.045, 0.005);
    assert!((e.norm() - (0.05 + 0.045 + 0.005 - 0.06)).abs() < 1e-9);
    assert!(e.x > 0.0);
    let e2 = k.escape(Vec3::new(0.2, 0.0, 0.0), 0.045, 0.005);
    assert!(e2.norm() < 1e-12);
    let pk = Keepout::Plane(PlaneKo {
        n: Vec3::new(1.0, 0.0, 0.0),
        p: Vec3::new(0.3, 0.0, 0.0),
    });
    let e3 = pk.escape(Vec3::new(0.20, 0.0, 0.0), 0.045, 0.005);
    assert!((e3.x - 0.15).abs() < 1e-9);
    let bk = Keepout::Box(BoxKo {
        lo: Vec3::new(0.15, -0.03, -0.03),
        hi: Vec3::new(0.20, 0.03, 0.03),
    });
    let e4 = bk.escape(Vec3::new(0.16, 0.0, 0.0), 0.045, 0.005);
    assert!(e4.norm() > 0.005);
}

/// The binding channel resolves through the model's names; fallback and "none" survive.
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
    let short = index_form.clone().with_names(&["joint1".to_string()]);
    assert_eq!(short.binding, "joint 1");
    let none = MotionBudget {
        binding: "none".to_string(),
        ..index_form.clone()
    };
    assert_eq!(none.binding_index(), None);
    assert_eq!(none.with_names(&names).binding, "none");
}

/// The chain at home: M, tip Jacobian, gravity, gravity stiffness, joint names.
fn home_model() -> (Mat, Mat, Vec<f64>, Vec<f64>, Vec<String>) {
    let chain = load_urdf_chain(&urdf_path(), "link00", "link06").expect("the chain parses");
    let mut pdyn = PgaDynamicsModel::new(chain.clone());
    let q = home_q();
    let m = pdyn.mass_matrix(&q);
    let (o, r) = chain.fk(&q);
    let tip = chain.tip_position(&o, &r);
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

/// The verdict names the binding joint and is recorded, never enforced.
#[test]
fn the_verdict_is_recorded_and_names_the_binding_joint() {
    let (m, j, g, k_eff, names) = home_model();
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
        "the saturated design reads feasible: wn_hi {} against wn_lo {}",
        fast.wn_hi, fast.wn_lo
    );
    let unclamped = budget_verdict(
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
    assert!(unclamped.feasible);
    assert_eq!(unclamped.binding, "none");
    println!(
        "budget: saturated binding {} at wn in [{:.1}, {:.1}] (feasible {}); unclamped binding {} (feasible {})",
        fast.binding, fast.wn_lo, fast.wn_hi, fast.feasible, unclamped.binding, unclamped.feasible
    );
}

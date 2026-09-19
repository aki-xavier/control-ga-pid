// task_frame.rs — GA_PID_AUDIT.md #19's pins: the task frame is world axes ABOUT THE TIP,
// [v; w] order, J's linear block the tip's point Jacobian (pinned in simu's tests/urdf.rs). Lambda is
// strongly coupled (cross-block correlation 0.80 at home, 0.996 bent), so "per-mode" placement
// would be a different design; K = wn^2 Lambda is symmetric only under the shipped uniform poles.

use control_ga_pid::inertia::task_space_inertia;
use control_math::mat::Mat;
use control_model::pga_dynamics::PgaDynamicsModel;
use control_model::urdf::{home_q, load_urdf_chain, urdf_path};

fn lam_at(chain: &control_model::urdf::UrdfChain, pdyn: &mut PgaDynamicsModel, q: &[f64]) -> Mat {
    let m = pdyn.mass_matrix(q);
    let (o, r) = chain.fk(q);
    let (tip, _) = chain.tip_pose(&o, &r);
    let j = chain.full_jacobian(&o, &r, tip);
    task_space_inertia(&m, &j, chain.n)
}

fn z1() -> (control_model::urdf::UrdfChain, PgaDynamicsModel) {
    let chain = load_urdf_chain(&urdf_path(), "link00", "link06").expect("z1 chain parses");
    let pdyn = PgaDynamicsModel::new(chain.clone());
    (chain, pdyn)
}

fn frob_block(m: &Mat, r0: usize, r1: usize, c0: usize, c1: usize) -> f64 {
    let mut s = 0.0;
    for i in r0..r1 {
        for j in c0..c1 {
            s += m.at(i, j) * m.at(i, j);
        }
    }
    s.sqrt()
}

fn max_abs(m: &Mat) -> f64 {
    let mut mx: f64 = 0.0;
    for i in 0..m.rows {
        for j in 0..m.cols {
            mx = mx.max(m.at(i, j).abs());
        }
    }
    mx
}

fn max_asym(m: &Mat) -> f64 {
    let mut mx: f64 = 0.0;
    for i in 0..m.rows {
        for j in i + 1..m.cols {
            mx = mx.max((m.at(i, j) - m.at(j, i)).abs());
        }
    }
    mx
}

#[test]
fn lambda_is_strongly_coupled_so_per_mode_would_be_a_different_design() {
    let (chain, mut pdyn) = z1();
    let home = home_q();
    let lam = lam_at(&chain, &mut pdyn, &home);
    let cross = frob_block(&lam, 0, 3, 3, 6)
        / (frob_block(&lam, 0, 3, 0, 3) * frob_block(&lam, 3, 6, 3, 6)).sqrt();
    assert!(
        (cross - 0.8033).abs() < 0.002,
        "home cross-block correlation moved: {cross}"
    );
    let mut bent = home_q();
    bent[1] -= 0.4;
    bent[2] += 0.5;
    let lam_b = lam_at(&chain, &mut pdyn, &bent);
    let cross_b = frob_block(&lam_b, 0, 3, 3, 6)
        / (frob_block(&lam_b, 0, 3, 0, 3) * frob_block(&lam_b, 3, 6, 3, 6)).sqrt();
    assert!(
        cross_b > 0.99,
        "the bent pose's coupling should be near-total, got {cross_b}"
    );
}

#[test]
fn the_shipped_uniform_poles_present_a_conservative_stiffness() {
    let (chain, mut pdyn) = z1();
    let lam = lam_at(&chain, &mut pdyn, &home_q());
    // every shipped Poles construction broadcasts one (wn, zeta), so K = wn^2 Lambda
    let k = lam.scale(15.0 * 15.0);
    let rel = max_asym(&k) / max_abs(&k);
    assert!(
        rel < 1e-12,
        "K = wn^2 Lambda must be symmetric to machine precision, got {rel:.3e}"
    );
}

#[test]
fn the_moment_the_poles_part_the_stiffness_becomes_an_energy_pump() {
    let (chain, mut pdyn) = z1();
    let lam = lam_at(&chain, &mut pdyn, &home_q());
    let mut wn2 = Mat::zeros(6, 6);
    let split = [30.0, 15.0, 15.0, 15.0, 15.0, 15.0];
    for i in 0..6 {
        wn2.set(i, i, split[i] * split[i]);
    }
    let k = lam.mul(&wn2);
    let rel = max_asym(&k) / max_abs(&k);
    assert!(
        (rel - 0.1022).abs() < 0.002,
        "a 2:1 split must go ~10 percent asymmetric, got {rel}"
    );
    // the antisymmetric part is an energy pump: work per lap = pi r^2 (K_ij - K_ji),
    // strongest between planes 0 and 2
    let pump = (k.at(0, 2) - k.at(2, 0)).abs();
    assert!(
        (pump - 201.9345).abs() < 0.5,
        "the (x, z) pump coefficient moved: {pump}"
    );
}

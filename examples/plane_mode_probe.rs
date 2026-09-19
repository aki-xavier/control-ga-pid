// Probe of K = Lambda diag(wn^2): the coupling, and when per-plane placement stops being safe.
// K is symmetric only while the wn^2 diagonal is uniform; a split makes it an energy pump.
// Run: cargo run --release --example plane_mode_probe

use control_ga_pid::inertia::task_space_inertia;
use control_math::mat::Mat;
use control_model::pga_dynamics::PgaDynamicsModel;
use control_model::urdf::{home_q, load_urdf_chain, urdf_path, UrdfChain};

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

fn jacobi_eig(a: &Mat) -> Vec<f64> {
    let n = a.rows;
    let mut d: Vec<f64> = (0..n * n).map(|k| a.at(k / n, k % n)).collect();
    for _ in 0..64 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += d[p * n + q] * d[p * n + q];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = d[p * n + q];
                if apq.abs() < 1e-18 {
                    continue;
                }
                let app = d[p * n + p];
                let aqq = d[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let dkp = d[k * n + p];
                    let dkq = d[k * n + q];
                    d[k * n + p] = c * dkp - s * dkq;
                    d[k * n + q] = s * dkp + c * dkq;
                }
                for k in 0..n {
                    let dpk = d[p * n + k];
                    let dqk = d[q * n + k];
                    d[p * n + k] = c * dpk - s * dqk;
                    d[q * n + k] = s * dpk + c * dqk;
                }
            }
        }
    }
    let mut ev: Vec<f64> = (0..n).map(|i| d[i * n + i]).collect();
    ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ev
}

fn block(m: &Mat, r0: usize, c0: usize) -> Mat {
    let mut b = Mat::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            b.set(i, j, m.at(r0 + i, c0 + j));
        }
    }
    b
}

fn lam_at(pdyn: &mut PgaDynamicsModel, chain: &UrdfChain, q: &[f64]) -> Mat {
    let m = pdyn.mass_matrix(q);
    let (o, r) = chain.fk(q);
    let (tip, _) = chain.tip_pose(&o, &r);
    let j = chain.full_jacobian(&o, &r, tip);
    task_space_inertia(&m, &j, chain.n)
}

fn report_pose(name: &str, lam: &Mat) {
    let lxx = frob_block(lam, 0, 3, 0, 3);
    let lxw = frob_block(lam, 0, 3, 3, 6);
    let lww = frob_block(lam, 3, 6, 3, 6);
    let cross = lxw / (lxx * lww).sqrt();
    let ev = jacobi_eig(lam);
    let ev_x = jacobi_eig(&block(lam, 0, 0));
    let ev_w = jacobi_eig(&block(lam, 3, 3));
    println!(
        "  {name}: ||L_xx||_F {lxx:.4} kg, ||L_xw||_F {lxw:.4} kg m, ||L_ww||_F {lww:.6} kg m^2"
    );
    println!(
        "  {:width$} normalized cross-block correlation {cross:.4}",
        "",
        width = name.len()
    );
    println!(
        "  {:width$} spectra: xx [{:.4}, {:.4}] kg, ww [{:.6}, {:.6}] kg m^2, full (unit-mixed) [{:.6}, {:.4}]",
        "",
        ev_x[0],
        ev_x[2],
        ev_w[0],
        ev_w[2],
        ev[0],
        ev[5],
        width = name.len()
    );
}

fn main() {
    let chain = load_urdf_chain(&urdf_path(), "link00", "link06").expect("the chain parses");
    let mut pdyn = PgaDynamicsModel::new(chain.clone());

    let home = home_q();
    let mut bent = home_q();
    bent[1] -= 0.4;
    bent[2] += 0.5;
    let twist: Vec<f64> = home_q()
        .iter()
        .enumerate()
        .map(|(i, v)| v + if i % 2 == 0 { 0.3 } else { -0.3 })
        .collect();

    println!("1. Lambda's coupling, three poses (J and Lambda about the tip, world axes, [v; w])");
    let lam_home = lam_at(&mut pdyn, &chain, &home);
    report_pose("home ", &lam_home);
    report_pose("bent ", &lam_at(&mut pdyn, &chain, &bent));
    report_pose("twist", &lam_at(&mut pdyn, &chain, &twist));

    println!();
    println!("2. the shipped identity: K = wn^2 Lambda at home, wn = 15 uniform on six planes");
    let k_uniform = lam_home.scale(15.0 * 15.0);
    let rel = max_asym(&k_uniform) / max_abs(&k_uniform);
    println!(
        "   max|K - K^T| = {:.3e}, max|K| = {:.4}, relative asymmetry {:.3e}",
        max_asym(&k_uniform),
        max_abs(&k_uniform),
        rel
    );

    println!();
    println!("3. the defect: same Lambda, wn = [30, 15, 15, 15, 15, 15] (2x on the x plane)");
    let mut wn2 = Mat::zeros(6, 6);
    let split = [30.0, 15.0, 15.0, 15.0, 15.0, 15.0];
    for i in 0..6 {
        wn2.set(i, i, split[i] * split[i]);
    }
    let k_split = lam_home.mul(&wn2);
    let kmax = max_abs(&k_split);
    let asym = max_asym(&k_split);
    println!(
        "   max|K - K^T| = {:.4}, max|K| = {:.4}, relative asymmetry {:.4}",
        asym,
        kmax,
        asym / kmax
    );
    let mut worst = (0.0f64, 0usize, 0usize);
    for i in 0..6 {
        for j in i + 1..6 {
            let d = (k_split.at(i, j) - k_split.at(j, i)).abs();
            if d > worst.0 {
                worst = (d, i, j);
            }
        }
    }
    println!(
        "   strongest pump: planes ({}, {}), pi r^2 * {:.4} per lap (a 10 mm-radius circle is charged {:.4} mJ)",
        worst.1,
        worst.2,
        worst.0,
        std::f64::consts::PI * 1e-4 * worst.0
    );
    println!(
        "   x-theta_x pair: pi r^2 * {:.4} per lap",
        k_split.at(0, 3) - k_split.at(3, 0)
    );
}

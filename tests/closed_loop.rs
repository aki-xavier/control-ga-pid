// closed_loop.rs — the loop against a plant, with no engine anywhere. This is the coverage the
// module could not have while it lived inside `simu`: every closed-loop test there needed a
// `CEnginePlant`, and the crate linked the shim unconditionally. The plant is `common/mod.rs`, which
// is the same ODE the products' plants integrate, with the engine's mirror left out, so these are
// measurements and not smoke tests — the first one is exact arithmetic.

mod common;

use common::ChainPlant;
use control_base::plant::Plant;
use control_ga_pid::inertia::task_space_inertia;
use control_ga_pid::opts::{GainMode, GainOpts, PlaneTaskLoopOpts};
use control_ga_pid::task_loop::PlaneTaskLoop;
use control_math::vec3::Vec3;
use control_model::urdf::{home_q, urdf_path};

const DT: f64 = 1e-3;

/// arm is the Z1 at home with the loop's `damp` set to the chain's own joint damping — the pair the
/// efference copy's contract names (the same dt, one plant step per call, and the plant's viscous
/// damping equal to the loop's `damp`).
fn arm(wn: f64, zeta: f64) -> (ChainPlant, PlaneTaskLoop) {
    let mut plant = ChainPlant::new(&urdf_path(), "link06", DT);
    plant.set_state(&home_q(), &[0.0; 6]);
    let damp = plant.chain.dampings.clone();
    let lp = PlaneTaskLoop::new(
        6,
        PlaneTaskLoopOpts {
            gains: GainOpts {
                m: 3,
                mode: GainMode::Poles,
                wn: vec![wn; 3],
                zeta: vec![zeta; 3],
                damp,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    (plant, lp)
}

/// THE POLES READING IS THE METRIC APPLIED TO THE LAW, and the command is otherwise only the plant's
/// own model terms: tau = J' Lambda num + C q_dot + g, with num_i = wn^2 e_i on a rest state. The
/// body of this check is the crate's own `task_space_inertia` re-deriving Lambda from the plant's M
/// and J, so a wrong metric, a wrong plane map or a missing feedforward each move the answer.
#[test]
fn the_poles_command_is_the_metric_applied_to_the_law() {
    let (mut plant, mut lp) = arm(15.0, 0.9);
    let (cur, cq) = plant.body_pose();
    // a 10 mm step in z is the whole error: the other two planes are told to stay where they are
    let target = cur.add(Vec3::new(0.0, 0.0, 0.01));

    let tau = lp.step(&mut plant, target, cq, DT, &[]);

    let kp = 15.0 * 15.0;
    let num = [0.0, 0.0, kp * 0.01];
    let j = plant.compute_jacobian();
    let m = plant.mass_matrix();
    let lam = task_space_inertia(&m, &j, 6);
    let f = lam.mul_vec(&num);
    let expected = j.transposed().mul_vec(&f);
    let bias = plant.bias_torques();
    let g = plant.gravity_torques();
    for i in 0..6 {
        let want = expected[i] + bias[i] + g[i];
        assert!(
            (tau[i] - want).abs() < 1e-9 * (1.0 + want.abs()),
            "joint {i}: commanded {} against J' Lambda num + C q_dot + g = {want}",
            tau[i]
        );
    }
    // and the step really is the whole of the demand: a zero step asks for no plane torque at all
    let (cur2, _) = plant.body_pose();
    let tau0 = lp.step(&mut plant, cur2, cq, DT, &[]);
    for i in 0..6 {
        assert!(
            (tau0[i] - (bias[i] + g[i])).abs() < 1e-9 * (1.0 + bias[i].abs() + g[i].abs()),
            "joint {i}: a held target still asks for {} beyond the model terms",
            tau0[i] - (bias[i] + g[i])
        );
    }
}

/// The closed loop settles: a 3-plane setpoint on a 6-joint arm, driven only by the law and the
/// plant's own model terms, reaches its target and does so without leaving the neighbourhood it was
/// given. The threshold is the tip's position, not a gain: 0.5 mm on a 50 mm step.
#[test]
fn a_setpoint_step_settles_on_its_target() {
    let (mut plant, mut lp) = arm(15.0, 0.9);
    let (cur, cq) = plant.body_pose();
    let target = cur.add(Vec3::new(0.05, 0.03, -0.02));

    let mut worst: f64 = 0.0;
    for _ in 0..4000 {
        let tau = lp.step(&mut plant, target, cq, DT, &[]);
        plant.step(&tau, 1);
        let (p, _) = plant.body_pose();
        worst = worst.max(p.sub(target).norm());
    }
    let (p, _) = plant.body_pose();
    let e = p.sub(target).norm();
    assert!(e < 5e-4, "the tip ended {e} m from its target");
    assert!(
        worst < 0.20,
        "the transient reached {worst} m on a 62 mm step: that is not this loop's answer"
    );
}

/// THE EFFERENCE COPY PAIRS THE COMMAND WITH WHAT THE PLANT APPLIED, and on an ideal plant the
/// difference is zero: the loop reconstructs the applied torque from the velocity the integrator
/// produced, and that reconstruction is the command it sent. This is the first test `Efference` has
/// had as a reader — the arm's own note on `PlaneTaskLoop.eff` records that, in `src/`, nothing
/// computes a number in the law from it — and it doubles as the pin on `take_efference`'s inverse of
/// the plant's step.
#[test]
fn the_efference_copy_pairs_the_command_with_what_the_plant_applied() {
    let (mut plant, mut lp) = arm(15.0, 0.9);
    let (cur, cq) = plant.body_pose();
    let target = cur.add(Vec3::new(0.02, 0.0, -0.03));

    let ticks = 50;
    for _ in 0..ticks {
        let tau = lp.step(&mut plant, target, cq, DT, &[]);
        plant.step(&tau, 1);
    }
    // the first tick has no earlier command to pair, so the copy holds one sample per tick after it
    assert_eq!(
        lp.eff.samples,
        ticks - 1,
        "the copy took {} samples over {ticks} ticks",
        lp.eff.samples
    );
    assert_eq!(lp.eff.commanded.len(), 6);
    assert_eq!(lp.eff.unit, "N.m");
    for i in 0..6 {
        assert!(
            lp.eff.residual(i).abs() < 1e-6,
            "channel {i} reads an external share of {} N.m on a plant that adds nothing",
            lp.eff.residual(i)
        );
    }
    assert!(
        lp.eff.external_share(0).abs() < 1e-6,
        "the copy attributes the command to the world: {}",
        lp.eff.report()
    );
}

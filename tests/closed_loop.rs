// The loop against `common/mod.rs`'s plant: the same ODE an engine-backed plant integrates.

mod common;

use common::ChainPlant;
use control_base::plant::Plant;
use control_ga_pid::inertia::task_space_inertia;
use control_ga_pid::opts::{GainMode, GainOpts, PlaneTaskLoopOpts};
use control_ga_pid::task_loop::PlaneTaskLoop;
use control_math::vec3::Vec3;
use control_model::urdf::{home_q, urdf_path};

const DT: f64 = 1e-3;

/// Chain at home; the loop's `damp` is the chain's own joint damping.
fn chain_home(wn: f64, zeta: f64) -> (ChainPlant, PlaneTaskLoop) {
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

/// tau = J' Lambda num + C q_dot + g, with Lambda re-derived by `task_space_inertia`.
#[test]
fn the_poles_command_is_the_metric_applied_to_the_law() {
    let (mut plant, mut lp) = chain_home(15.0, 0.9);
    let cur = plant.task_position();
    let cq = plant.task_rotation();
    // only z moves; the other planes are commanded to hold
    let target = cur.add(Vec3::new(0.0, 0.0, 0.01));

    let tau = lp.step(&mut plant, target, cq, DT, &[]);

    let kp = 15.0 * 15.0;
    let num = [0.0, 0.0, kp * 0.01];
    let j = plant.task_jacobian();
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
    let cur2 = plant.task_position();
    let tau0 = lp.step(&mut plant, cur2, cq, DT, &[]);
    for i in 0..6 {
        assert!(
            (tau0[i] - (bias[i] + g[i])).abs() < 1e-9 * (1.0 + bias[i].abs() + g[i].abs()),
            "joint {i}: a held target still asks for {} beyond the model terms",
            tau0[i] - (bias[i] + g[i])
        );
    }
}

/// A 3-plane setpoint reaches its target, and the transient stays bounded.
#[test]
fn a_setpoint_step_settles_on_its_target() {
    let (mut plant, mut lp) = chain_home(15.0, 0.9);
    let cur = plant.task_position();
    let cq = plant.task_rotation();
    let target = cur.add(Vec3::new(0.05, 0.03, -0.02));

    let mut worst: f64 = 0.0;
    for _ in 0..4000 {
        let tau = lp.step(&mut plant, target, cq, DT, &[]);
        plant.step(&tau, 1);
        let p = plant.task_position();
        worst = worst.max(p.sub(target).norm());
    }
    let p = plant.task_position();
    let e = p.sub(target).norm();
    assert!(e < 5e-4, "the tip ended {e} m from its target");
    assert!(
        worst < 0.20,
        "the transient reached {worst} m on a 62 mm step: that is not this loop's answer"
    );
}

/// The efference copy reconstructs the applied torque: zero residual on an ideal plant.
#[test]
fn the_efference_copy_pairs_the_command_with_what_the_plant_applied() {
    let (mut plant, mut lp) = chain_home(15.0, 0.9);
    let cur = plant.task_position();
    let cq = plant.task_rotation();
    let target = cur.add(Vec3::new(0.02, 0.0, -0.03));

    let ticks = 50;
    for _ in 0..ticks {
        let tau = lp.step(&mut plant, target, cq, DT, &[]);
        plant.step(&tau, 1);
    }
    // the first tick has no earlier command to pair
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

/// The metric refresh is driven by the CONFIGURATION STAMP and not by a position, which is what lets a
/// plant whose coordinates are NOT a configuration be driven at all — this machine's stance reduction
/// has a velocity-level coordinate set and no position vector to hand over, while the loop's actual
/// question ("has the metric's configuration moved?") it can answer.
///
/// `ChainPlant` counts the calls, so a revert inside `step_ff` to `joint_positions` fails here rather
/// than silently narrowing the contract back to plants that have a configuration.
#[test]
fn the_metric_refresh_is_driven_by_the_stamp() {
    let (mut plant, mut lp) = chain_home(15.0, 0.9);
    let cur = plant.task_position();
    let cq = plant.task_rotation();
    assert_eq!(
        plant.stamp_calls, 0,
        "the loop asked for the stamp before it ticked"
    );
    let _ = lp.step(&mut plant, cur, cq, DT, &[]);
    assert_eq!(plant.stamp_calls, 1, "one tick, one stamp");
    let _ = lp.step(&mut plant, cur, cq, DT, &[]);
    assert_eq!(
        plant.stamp_calls, 2,
        "the stamp is asked once per tick, and the loop's refresh reads it"
    );
    // and the stamp is what the contract says it is for a plant whose coordinates are a configuration:
    // its position. A plant is free to answer something else — that is the point of the separation.
    assert_eq!(plant.configuration_stamp(), plant.joint_positions());
}

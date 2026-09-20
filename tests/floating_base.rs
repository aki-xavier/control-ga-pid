// The BASE END: a floating machine read against the welded chain it has to be the same machine as when
// its base rows are bounded by a very large number — and a floating machine when they are not.

mod common;

use common::{ChainPlant, FloatingChainPlant};
use control_base::plant::Plant;
use control_ga_pid::inertia::task_space_inertia;
use control_ga_pid::keepout::Keepout;
use control_ga_pid::opts::{AvoidOpts, GainMode, GainOpts, IntegralOpts, PlaneTaskLoopOpts};
use control_ga_pid::plane_ko::PlaneKo;
use control_ga_pid::task_loop::PlaneTaskLoop;
use control_math::quat::Quat;
use control_math::vec3::Vec3;
use control_model::urdf::{home_q, urdf_path};

const DT: f64 = 1e-3;
/// the chain the two plants are the same machine of, and its terminal link
const EE: &str = "link06";
/// A bound this side of anything the hold ever asks for. Why a NUMBER and not a flag: the base end is
/// one entry in the actuator table (`u_lim`), and "unbounded" is a value no command reaches — this one
/// is a thousand tonnes of force, twelve orders above the 34 N this machine's base weighs.
const WELD: f64 = 1e12;

/// The welded chain: this project's plant until now, kept as the REFERENCE the weld is checked against.
fn fixed_chain(wn: f64, zeta: f64) -> (ChainPlant, PlaneTaskLoop) {
    let mut plant = ChainPlant::new(&urdf_path(), EE, DT);
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

/// The same machine with its root in the state: at home, at the origin, nothing moving.
fn floating_home() -> FloatingChainPlant {
    let mut plant = FloatingChainPlant::new(&urdf_path(), EE, DT);
    let nu = vec![0.0; plant.nv()];
    plant.set_state(Vec3::ZERO, pga::rotor_identity(), &home_q(), &nu);
    plant
}

/// The loop for that machine. `u_base` is the base actuator's bound, read off the leading `u_lim` rows:
/// `None` bounds none of them, which is no actuator at the base at all. The joints carry no bound here
/// (a zero row is one the clamp does not touch), so the table says one thing and nothing else.
fn loop_for(plant: &FloatingChainPlant, wn: f64, zeta: f64, u_base: Option<f64>) -> PlaneTaskLoop {
    let n = plant.nv();
    let u_lim = match u_base {
        None => Vec::new(),
        Some(v) => {
            let mut u = vec![0.0; n];
            for x in u.iter_mut().take(6) {
                *x = v;
            }
            u
        }
    };
    PlaneTaskLoop::new(
        n,
        PlaneTaskLoopOpts {
            gains: GainOpts {
                m: 3,
                mode: GainMode::Poles,
                wn: vec![wn; 3],
                zeta: vec![zeta; 3],
                damp: plant.damp.clone(),
                ..Default::default()
            },
            integral: IntegralOpts {
                u_lim,
                ..Default::default()
            },
            ..Default::default()
        },
    )
}

/// The models behind the two plants must be the same machine before any command is compared: the joint
/// block of the floating one against the welded one's mass matrix, and the joint columns of its task
/// Jacobian against the welded one's.
#[test]
fn the_two_models_are_the_same_machine_at_the_same_state() {
    let (mut fixed, _) = fixed_chain(15.0, 0.9);
    let mut float = floating_home();

    let m_f = fixed.mass_matrix();
    let j_f = fixed.task_jacobian();
    let m_g = float.mass_matrix();
    let j_g = float.task_jacobian();
    assert_eq!(
        m_g.rows, 12,
        "the floating machine's coordinates are a pose plus six joints"
    );
    assert_eq!(j_g.cols, 12);

    let mut worst_m: f64 = 0.0;
    let mut worst_j: f64 = 0.0;
    for i in 0..6 {
        for j in 0..6 {
            let a = m_f.at(i, j);
            let b = m_g.at(6 + i, 6 + j);
            worst_m = worst_m.max((a - b).abs() / (1.0 + a.abs()));
        }
        for c in 0..3 {
            let a = j_f.at(c, i);
            let b = j_g.at(c, 6 + i);
            worst_j = worst_j.max((a - b).abs() / (1.0 + a.abs()));
        }
    }
    assert!(
        worst_m < 1e-12,
        "the floating machine's joint block is not the welded machine's mass matrix: {worst_m:e}"
    );
    assert!(
        worst_j < 1e-12,
        "the joint columns of the task Jacobian are not the welded machine's: {worst_j:e}"
    );
    // and the task frame itself is one point, not two conventions of it
    let (p_f, q_f) = fixed.task_pose();
    let (p_g, q_g) = float.task_pose();
    assert!(
        (p_f.sub(p_g)).norm() < 1e-12,
        "the two tips are {} apart",
        (p_f.sub(p_g)).norm()
    );
    assert!(Quat::rotvec_between(q_f, q_g).norm() < 1e-12);
}

/// THE STATEMENT THE WHOLE CHANGE RESTS ON: bounding the base rows by a very large number is a weld — a
/// floating machine read that way commands the joint rows the welded machine commands, and its own base
/// rows are what the hold costs.
#[test]
fn a_large_bound_at_the_base_commands_the_fixed_machines_joint_rows() {
    let (mut fixed, mut lp_f) = fixed_chain(15.0, 0.9);
    let mut float = floating_home();
    let mut lp_g = loop_for(&float, 15.0, 0.9, Some(WELD));
    let (cur_f, cq_f) = fixed.task_pose();
    let (cur_g, _) = float.task_pose();
    let step = Vec3::new(0.01, -0.02, 0.03);

    let tau_f = lp_f.step(&mut fixed, cur_f.add(step), cq_f, DT, &[]);
    let tau_g = lp_g.step(&mut float, cur_g.add(step), cq_f, DT, &[]);
    assert_eq!(tau_f.len(), 6);
    assert_eq!(
        tau_g.len(),
        12,
        "a floating machine's command covers its base too"
    );
    for i in 0..6 {
        let want = tau_f[i];
        assert!(
            (tau_g[6 + i] - want).abs() < 1e-9 * (1.0 + want.abs()),
            "joint {i}: the welded floating machine commanded {} against the welded chain's {want}",
            tau_g[6 + i]
        );
    }
    // the base rows are not the welded machine's (it has none): they are the wrench the hold costs, and
    // at this state that is the machine's weight in the vertical direction
    assert!(
        tau_g[5].abs() > 1.0,
        "the base's vertical row reads {} N, which is not a wrench",
        tau_g[5]
    );
}

/// And it stays equal over a whole motion, which is what "treated as fixed" has to mean.
#[test]
fn a_large_bound_at_the_base_moves_the_machine_like_the_fixed_one() {
    let (mut fixed, mut lp_f) = fixed_chain(15.0, 0.9);
    let mut float = floating_home();
    let mut lp_g = loop_for(&float, 15.0, 0.9, Some(WELD));
    let (cur_f, cq_f) = fixed.task_pose();
    let (_cur_g, _) = float.task_pose();
    let target = cur_f.add(Vec3::new(0.04, 0.02, -0.03));

    let mut worst: f64 = 0.0;
    for _ in 0..500 {
        let tau_f = lp_f.step(&mut fixed, target, cq_f, DT, &[]);
        let tau_g = lp_g.step(&mut float, target, cq_f, DT, &[]);
        fixed.step(&tau_f, 1);
        float.step(&tau_g, 1);
        let (p_f, _) = fixed.task_pose();
        let (p_g, _) = float.task_pose();
        worst = worst.max(p_f.sub(p_g).norm());
    }
    assert!(
        worst < 1e-9,
        "the welded floating machine's tip drifted {worst} m from the welded chain's"
    );
    // and the base did not move: the hold is a hold, not a slow drift
    assert!(
        float.base_p.norm() < 1e-9,
        "the held base walked {}",
        float.base_p.norm()
    );
}

/// The complement, without which the tests above prove nothing about the number: a bound SHORT of what
/// the hold asks for is a floating machine, and the base gives way under the joint reaction.
#[test]
fn a_bound_short_of_the_demand_is_what_lets_the_base_give_way() {
    let mut float = floating_home();
    // HALF of what the base's own weight asks for, read off the model rather than guessed: it is the
    // same gravity term the loop adds back
    let weight = float.gravity_torques();
    assert!(
        weight[5].abs() > 1.0,
        "the base's own weight row reads {} N, which is not a wrench",
        weight[5]
    );
    let bound = 0.5 * weight[5].abs();
    let mut lp_g = loop_for(&float, 15.0, 0.9, Some(bound));
    let (cur, cq) = float.task_pose();
    for _ in 0..200 {
        let tau = lp_g.step(&mut float, cur, cq, DT, &[]);
        assert!(
            tau[5].abs() <= bound + 1e-9,
            "the loop commanded {} N at the base against its own {} N bound",
            tau[5],
            bound
        );
        float.step(&tau, 1);
    }
    let walked = float.base_p.norm();
    assert!(
        walked > 1e-3,
        "a base actuator short of the weight held it anyway ({walked} m): the bound is not read"
    );
    assert!(
        float.base_p.z < -1e-3,
        "the base should sag under its own weight, and it reads {}",
        float.base_p.z
    );
}

/// The default reading: a base nobody bounds has to have its wrench sourced by contact, so the rows come
/// back zero and the metric is the FREE one — `(J M^-1 J')^-1` over the whole machine, not a joint block.
#[test]
fn a_base_nobody_bounds_is_written_by_nobody_and_the_reading_is_the_free_one() {
    let mut float = floating_home();
    let mut lp_g = loop_for(&float, 15.0, 0.9, None);
    let (cur, cq) = float.task_pose();
    let target = cur.add(Vec3::new(0.0, 0.02, 0.0));
    let tau = lp_g.step(&mut float, target, cq, DT, &[]);
    for i in 0..6 {
        assert_eq!(
            tau[i], 0.0,
            "row {i} of an unbounded base was written: {}",
            tau[i]
        );
    }

    // the joint rows are `J' f + h` with f = Lambda num over the WHOLE machine's metric
    let j = float.task_jacobian();
    let m = float.mass_matrix();
    let lam = task_space_inertia(&m, &j, 12);
    let wn = 15.0;
    let e = [0.0, 0.02, 0.0];
    let num = [wn * wn * e[0], wn * wn * e[1], wn * wn * e[2]];
    let f = lam.mul_vec(&num);
    let expected = j.transposed().mul_vec(&f);
    let bias = float.bias_torques();
    let g = float.gravity_torques();
    for i in 0..6 {
        let want = expected[6 + i] + bias[6 + i] + g[6 + i];
        assert!(
            (tau[6 + i] - want).abs() < 1e-9 * (1.0 + want.abs()),
            "joint {i}: the free reading commanded {} against the free metric's {want}",
            tau[6 + i]
        );
    }
    // and nothing holds the machine in six directions, which is what the free reading is about
    assert!(!float.structure().braced());
}

/// A bound in SOME base directions and none in others is a machine this loop cannot read — it holds all
/// six or none — so it reads it as having no actuator at all and says so once.
#[test]
fn a_bound_in_some_base_directions_only_is_read_as_no_actuator() {
    let mut float = floating_home();
    let mut lp_g = loop_for(&float, 15.0, 0.9, Some(WELD));
    // take the bound off five of the six base rows
    for i in 1..6 {
        lp_g.u_lim[i] = 0.0;
    }
    let (cur, cq) = float.task_pose();
    let tau = lp_g.step(&mut float, cur, cq, DT, &[]);
    for i in 0..6 {
        assert_eq!(
            tau[i], 0.0,
            "row {i} was written from a base the loop had declared unreadable: {}",
            tau[i]
        );
    }
}

/// The efference copy is the same statement about the base rows: the wrench the loop commanded at the
/// base is the wrench the plant applied, so a welded run leaves no residual there either — which is what
/// says the base rows are a consistent command and not a number that happens to be there.
#[test]
fn the_efference_copy_covers_the_base_rows_too() {
    let mut float = floating_home();
    let mut lp_g = loop_for(&float, 15.0, 0.9, Some(WELD));
    let (cur, cq) = float.task_pose();
    let target = cur.add(Vec3::new(0.02, 0.0, -0.03));
    let ticks = 50;
    for _ in 0..ticks {
        let tau = lp_g.step(&mut float, target, cq, DT, &[]);
        float.step(&tau, 1);
    }
    assert_eq!(
        lp_g.eff.samples,
        ticks - 1,
        "one pairing per tick after the first"
    );
    assert_eq!(
        lp_g.eff.commanded.len(),
        12,
        "the copy runs in the machine's own coordinates, the base included"
    );
    for i in 0..12 {
        assert!(
            lp_g.eff.residual(i).abs() < 1e-6,
            "channel {i} reads an external share of {} on a plant that adds nothing",
            lp_g.eff.residual(i)
        );
    }
}

/// The whole-arm escape reads its rows through the same split: a held base's columns cannot escape
/// anything, so the solve is written in the joints and the base's rows of the readout stay zero — while
/// the weld still holds under the escape's own torque.
#[test]
fn the_whole_arm_escape_is_written_in_the_joints_a_held_base_leaves() {
    let mut plant = floating_home();
    let n = plant.nv();
    // a plane through the tip's own position, so the escape is engaged from the first tick
    let (cur, _) = plant.task_pose();
    let u_lim = {
        let mut u = vec![0.0; n];
        for x in u.iter_mut().take(6) {
            *x = WELD;
        }
        u
    };
    let mut lp = PlaneTaskLoop::new(
        n,
        PlaneTaskLoopOpts {
            gains: GainOpts {
                m: 3,
                mode: GainMode::Poles,
                wn: vec![15.0; 3],
                zeta: vec![0.9; 3],
                damp: plant.damp.clone(),
                ..Default::default()
            },
            integral: IntegralOpts {
                u_lim,
                ..Default::default()
            },
            avoid: AvoidOpts {
                full_body: true,
                keepouts: vec![Keepout::Plane(PlaneKo {
                    n: Vec3::new(0.0, 0.0, 1.0),
                    p: cur,
                })],
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (_, cq) = plant.task_pose();
    let p0 = plant.base_p;
    for _ in 0..50 {
        let tau = lp.step(&mut plant, cur, cq, DT, &[]);
        plant.step(&tau, 1);
    }
    assert_eq!(
        lp.escape_dq.len(),
        n,
        "the escape's readout is in the machine's own coordinates"
    );
    assert!(
        lp.escape_dq[..6].iter().all(|d| *d == 0.0),
        "a held base's rows of the escape are not zero: {:?}",
        &lp.escape_dq[..6]
    );
    assert!(
        lp.escape_dq[6..].iter().any(|d| d.abs() > 1e-12),
        "the escape never engaged, so this checks nothing"
    );
    assert!(
        plant.base_p.sub(p0).norm() < 1e-9,
        "the base walked {} m under the escape",
        plant.base_p.sub(p0).norm()
    );
}

/// A joint-space design spans EVERY coordinate, so a bounded base is commanded like a joint: the same
/// arithmetic makes the commanded base pose real, the coupling included.
#[test]
fn a_joint_design_holds_the_base_it_commands() {
    let mut plant = floating_home();
    let mut lp = loop_for(&plant, 15.0, 0.9, Some(WELD));
    let n = plant.nv();
    let q_des = plant.joint_positions();
    let v_des = vec![0.0; n];
    let p0 = plant.base_p;
    let mut worst: f64 = 0.0;
    for _ in 0..300 {
        let tau = lp.step_joint(&mut plant, &q_des, &v_des, DT);
        assert_eq!(tau.len(), n);
        plant.step(&tau, 1);
        worst = worst.max(plant.base_p.sub(p0).norm());
    }
    assert!(
        worst < 1e-9,
        "a joint design commanded to hold the base let it walk {worst} m"
    );
    // the joints, too, stay where they were commanded
    let q = plant.joint_positions();
    for i in 6..n {
        assert!(
            (q[i] - q_des[i]).abs() < 1e-6,
            "joint {} moved to {} against a commanded {}",
            i - 6,
            q[i],
            q_des[i]
        );
    }
}

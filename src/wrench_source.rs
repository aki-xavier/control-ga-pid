// wrench_source.rs — where the law's desired force is TURNED INTO a command. A reading forms the joint
// rows the demand asks for; what may be written on the rows the law does NOT span is a fact about the
// PLANT, and the answers are not one answer:
//
//   ActuatedJoints   a base with no actuator declared: its rows are left to the contact, and zero until
//                    a source claims them.
//   HeldBase         a bounded actuator at the base: the leading rows are the wrench the hold costs
//                    (base.rs), clamped by that same bound.
//   ContactWrench    a base whose wrench must come from the ground. NOT YET WRITTEN: this is where a
//                    legged machine's load distribution goes, and `realize` below is the ONE place that
//                    decides it, so neither reading branches on it.
//
// The last step is the plant's own: `realize` asks it to lift the command onto its actuators, because a
// plant's declared coordinates are its actuators only when it is fully actuated in them.

use crate::base::HeldBase;
use control_base::plant::Plant;
use control_math::mat::Mat;

/// What a source is handed: the base end the reading already split, and the bounds and damping that
/// decide what may be written.
pub struct RealizeCtx<'a> {
    /// Some when the plant's base is a state and `u_lim` bounds all of its rows
    pub hb: Option<HeldBase>,
    /// the loop's coordinate count, the base included
    pub n: usize,
    /// the coordinates the demand spans: `n` less a held base
    pub nj: usize,
    /// leading rows a base with no actuator leaves to the contact
    pub nfree: usize,
    /// per-coordinate output bound, `n` entries when stated and empty otherwise
    pub u_lim: &'a [f64],
    /// per-coordinate damping, the plant's own; the held-base wrench cancels the command it implies
    pub damp: &'a [f64],
    pub dt: f64,
}

/// Which force the command is written as. The plant is an argument because the bound and the base end are
/// statements about THE MACHINE, and it is `&mut` because the terms come from a plant that caches them.
pub trait WrenchSource {
    /// The bound, then the base end: what the machine is actually commanded with.
    fn settle(&self, plant: &mut dyn Plant, ctx: &RealizeCtx, tau_j: &[f64]) -> Vec<f64>;
}

/// The shipped source: the machine drives every coordinate the law spans.
pub struct ActuatedJoints;

/// A bounded actuator at the base, which is the whole of what makes a machine a fixed one.
pub struct HeldBaseSource;

impl WrenchSource for ActuatedJoints {
    fn settle(&self, _plant: &mut dyn Plant, ctx: &RealizeCtx, tau_j: &[f64]) -> Vec<f64> {
        let mut tau = clamp(ctx, tau_j);
        // a base with no actuator declared is written by nobody: its rows are left to the contact
        for i in 0..ctx.nfree {
            tau[i] = 0.0;
        }
        tau
    }
}

impl WrenchSource for HeldBaseSource {
    fn settle(&self, plant: &mut dyn Plant, ctx: &RealizeCtx, tau_j: &[f64]) -> Vec<f64> {
        let tau = clamp(ctx, tau_j);
        let h = ctx
            .hb
            .expect("a held base is this source's own precondition");
        hold_base(plant, ctx, h, &tau)
    }
}

/// The command for a structure, from the split the reading already made, in the coordinates the MACHINE
/// takes. Why the lift is here and last: a reading forms a force in the coordinates the plant DECLARED,
/// and a plant whose coordinates are a subspace of its actuators owes the difference — that step is the
/// one thing no reading can do for it. Why the source is chosen rather than held: which one applies is a
/// fact about the plant, and a plant is handed in per tick.
pub fn realize(plant: &mut dyn Plant, ctx: &RealizeCtx, tau_j: &[f64]) -> Vec<f64> {
    let tau = match ctx.hb {
        Some(_) => HeldBaseSource.settle(plant, ctx, tau_j),
        None => ActuatedJoints.settle(plant, ctx, tau_j),
    };
    plant.realize_command(&tau)
}

/// The joint rows a demand asks for: `J' f + bias + gravity`, UN-clamped, so a reading may add its own
/// secondary task before the bound applies — an escape bounded by the same actuators as the task.
pub fn joint_demand(
    plant: &mut dyn Plant,
    hb: Option<HeldBase>,
    j: &Mat,
    f: &[f64],
    nj: usize,
) -> Vec<f64> {
    let mut tau = j.transposed().mul_vec(f);
    let bias = tail_of(hb, &plant.bias_torques());
    let g = tail_of(hb, &plant.gravity_torques());
    for i in 0..nj {
        tau[i] += bias[i] + g[i];
    }
    tau
}

/// Rows `nb..` of a full-coordinate vector: the joint part of a bias, a weight, a velocity, a bound; a
/// machine with no base in its coordinates is every row.
pub fn tail_of(hb: Option<HeldBase>, v: &[f64]) -> Vec<f64> {
    match hb {
        Some(h) => h.tail(v),
        None => v.to_vec(),
    }
}

/// The bound row by row: the joints by their own entries, and a held base's by the base actuator's.
fn clamp(ctx: &RealizeCtx, tau_j: &[f64]) -> Vec<f64> {
    let mut tau = tau_j.to_vec();
    if ctx.u_lim.len() == ctx.n {
        let lim = tail_of(ctx.hb, ctx.u_lim);
        for i in 0..ctx.nj {
            if lim[i] > 0.0 {
                if tau[i] > lim[i] {
                    tau[i] = lim[i];
                }
                if tau[i] < -lim[i] {
                    tau[i] = -lim[i];
                }
            }
        }
    }
    tau
}

/// The base rows of a held machine: the wrench an actuator with NO torque bound must supply for the base
/// to stay where it is, given the joint command this tick settled on.
///
/// Why the model terms are read again: the wrench is a statement about the WHOLE machine
/// (`M_bj qdd_j + h_base`) and the law's own reads were the joint block, so the coupling is not in hand.
/// Why the acceleration is solved here rather than designed: it is what the PLANT will do — the same
/// implicit solve it integrates with, `(M_jj + dt D_j) qdd_j = tau_j - h_j - D_j v_j` — so the wrench
/// cancels the command that was actually formed and not the one the design intended.
fn hold_base(plant: &mut dyn Plant, ctx: &RealizeCtx, h: HeldBase, tau_j: &[f64]) -> Vec<f64> {
    let m = plant.mass_matrix();
    let mut hf = plant.bias_torques();
    let g = plant.gravity_torques();
    for i in 0..hf.len().min(g.len()) {
        hf[i] += g[i];
    }
    let v = plant.joint_velocities();
    let hj = h.tail(&hf);
    let mut mm = h.joint_block(&m);
    let mut r = vec![0.0; h.nj()];
    for i in 0..h.nj() {
        let d = if h.nb + i < ctx.damp.len() {
            ctx.damp[h.nb + i]
        } else {
            0.0
        };
        let vi = if h.nb + i < v.len() { v[h.nb + i] } else { 0.0 };
        r[i] = tau_j[i] - hj[i] - d * vi;
        mm.set(i, i, mm.at(i, i) + ctx.dt * d);
    }
    let qdd_j = mm.solve(&r);
    let mut w = h.wrench(&m, &hf, &qdd_j);
    // the base rows of the viscous term the plant also subtracts: zero on a base with no joint
    // damping, and stated rather than assumed because the caller's `damp` IS the plant's damping
    let vb = h.head(&v);
    let db = h.head(ctx.damp);
    for i in 0..h.nb {
        w[i] += db[i] * vb[i];
    }
    // THE ACTUATOR'S OWN BOUND, and the whole difference between this machine and a welded one: at a
    // very large bound the wrench passes through verbatim and the base cannot move; at a bound short
    // of it the base gives way, which is a floating machine again
    for i in 0..h.nb {
        let lim = if i < ctx.u_lim.len() {
            ctx.u_lim[i]
        } else {
            0.0
        };
        if w[i] > lim {
            w[i] = lim;
        }
        if w[i] < -lim {
            w[i] = -lim;
        }
    }
    h.assemble(&w, tau_j)
}

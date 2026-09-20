# control-ga-pid — the GA-PID control core

The law in plane space, the design surface that parameterizes it, and the loop that realizes it.
MIT-licensed (see `LICENSE`).

## The law

```text
num_i = k_i soft_i e_i - d_i (v_i - v_ref_i) + alpha_i i_acc_i
f     = Lambda num        (the poles reading)
f     = num               (the physical reading)
tau   = J' f + C q_dot + g
```

`law.rs` holds the `num` expression; its design inputs are the pole placement
`k = wn^2 - k_eff`, `d = 2 zeta wn - b_eff` (`design.rs`) and the metric
`Lambda = (J M^-1 J^T)^-1` (`inertia.rs`).

## The base end

A FIXED base is not a different machine: it is the floating one whose base actuator is bounded by a
number no command reaches. The base end is stated in ONE place — the leading `base_dof` rows of
`u_lim`, which on a floating machine are the base actuator's bound rather than a joint's:

```text
u_lim[..base_dof] all zero (or absent)   no actuator at the base: the free reading, and the rows
                                         the loop returns there are zero, the wrench owed to contact
u_lim[..base_dof] all positive, large    a bound nothing reaches: the base cannot accelerate, and the
                                         joint rows are exactly the welded machine's
u_lim[..base_dof] all positive, small    an actuator that gives way under the joint reaction
```

So a very large value IS the weld, and there is no second flag: the loop reads the joint block of `M`
(`base.rs`) whenever the base rows are bounded, writes the wrench the hold costs there (clamped by
those rows), and reads the free metric over the whole machine otherwise. A bound in some base
directions and none in others is a machine the loop cannot read, and it says so.

## Modules

```text
law          THE law, as one expression and no realization of it
design       the pole placement, and the settling/overshoot relations inverted
gains        PlaneGains, one plane's (kappa_p, kappa_d, kappa_i) triple
impedance    the descending-impedance channel a caller scales its gains with
inertia      Lambda = (J M^-1 J^T)^-1, its per-plane masses, the passivity floor
spec         TaskSpec, the task-class input to the design
class        TaskClass -> spec -> gains, and the window it solves
budget       MotionBudget: the per-joint actuator inequality, scanned
window       MotionWindow: what the theory alone supports (not the same flag)
opts         PlaneTaskLoopOpts, the loop's one configuration surface, and GainMode
base         the joint block a held base leaves, and the constraint wrench holding it demands
task_loop    the realization: tau = J' f + C q_dot + g, and the only file that names a plant
keepout      the convex keep-out sum type: signed distance, shaping, escape
plane_ko     the half-space primitive          \
box_ko       the axis-aligned box primitive     >  the three Keepout variants
sphere_ko    the spherical primitive           /
escape       TaskAvoidance: the goal-side projection against the keep-outs
recruit      minimal-authority-first allocation of redundancy
```

## Dependencies

`control-math` (the arithmetic) and `control-base` (the `Plant` contract and the `Efference` copy),
both siblings below this crate; `control-model` is a dev-dependency only. No `build.rs`, no engine.

## Building

```sh
mbx test
mbx clippy --all-targets -- -D warnings
mbx run --release --example plane_mode_probe
```

`tests/common/mod.rs` is the engine-free rigid-body plant the closed-loop tests drive: the same ODE
an engine-backed plant integrates, with the engine's world mirror left out. It holds both shapes of
the same z1 machine — the welded chain (*the reference*) and the floating one, which applies whatever
it is commanded and leaves the base's bound to the loop's `u_lim`. `tests/floating_base.rs` reads one
against the other: at a bound of `1e12` the floating machine commands the welded machine's joint rows
and moves like it, and at a bound short of the demand its base gives way.

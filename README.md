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
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --release --example plane_mode_probe
```

`tests/common/mod.rs` is the engine-free rigid-body plant the closed-loop tests drive: the same ODE
an engine-backed plant integrates, with the engine's world mirror left out.

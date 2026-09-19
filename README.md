# control-ga-pid — the GA-PID control core

A project of its own, and not a module inside anything that uses it: **one law,
two realizations** — the arm of `simu` drives it from a fixed base, the biped
drives the same expression through ground contact, and neither may own it.
MIT-licensed (see `LICENSE`).

## The law

One expression, in plane space:

```text
num_i = k_i soft_i e_i - d_i (v_i - v_ref_i) + alpha_i i_acc_i
f     = Lambda num        (the poles reading)
f     = num               (the physical reading)
tau   = J' f + C q_dot + g
```

`law.rs` holds the `num` expression itself. Its two design inputs are the pole
placement `k = wn^2 - k_eff`, `d = 2 zeta wn - b_eff` (`design.rs`) and the metric
`Lambda = (J M^-1 J^T)^-1` (`inertia.rs`). Everything else here is the numbers that
design is read from, the loop one realization of it runs in, or what that loop acts
on.

## Modules

```text
law          THE law, as one expression and no realization of it
design       the pole placement, and the settling/overshoot relations inverted
gains        PlaneGains, one plane's (kappa_p, kappa_d, kappa_i) triple
inertia      Lambda = (J M^-1 J^T)^-1, its per-plane masses, the passivity floor
spec         TaskSpec, the task-class input to the design
class        TaskClass -> spec -> gains, and the window it solves
budget       MotionBudget: the per-joint actuator inequality, scanned
window       MotionWindow: what the theory alone supports (not the same flag)
opts         PlaneTaskLoopOpts, the loop's one configuration surface, and GainMode
task_loop    the realization: tau = J' f + C q_dot + g, in all three readings
keepout      the convex keep-out sum type: signed distance, shaping, escape
plane_ko     the half-space primitive          \
box_ko       the axis-aligned box primitive     >  the three Keepout variants
sphere_ko    the spherical primitive           /
escape       TaskAvoidance: the goal-side projection against the keep-outs
recruit      minimal-authority-first allocation of redundancy
```

`task_loop.rs` is the **only** file that names a plant. It programs against the
`Plant` contract in [`control-base`](../control-base) and implements nothing, which
is what lets this crate's own tests run without an engine.

## Dependencies

Two, both siblings below this crate:

- [`control-math`](../control-math) — the arithmetic every type here is written in;
- [`control-base`](../control-base) — the `Plant` contract the loop drives and the
  `Efference` copy it keeps.

[`control-model`](../control-model) is a **dev**-dependency only: the URDF chain and
the PGA dynamics the tests and the probe build a plant out of. Nothing in `src/`
names a model. There is no `build.rs`, and no engine — an engine binding, a
model-based view and a test fake are all equally callers' objects.

## Where the seam sits, and why it is a crate

The law is stated once and read by two machines that share nothing else, so it may
not live under either of them:

- **A law copied per caller is as many laws as there are copies.** The expression
  above was written out by hand at every call site before `law.rs` existed.
- **Nothing here may sit above anything that uses it.** `simu`'s arm and its biped
  both consume this as a sibling path dependency
  (`{ path = "../control-ga-pid" }`) and implement `Plant` for their own plants;
  were the law stated inside either, the two could not be compared.

## Building

```sh
cargo test                                     # 20 tests, no engine needed
cargo clippy --all-targets -- -D warnings
cargo run --release --example plane_mode_probe # GA_PID_AUDIT.md #19's premise
```

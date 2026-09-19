// control-ga-pid — the GA-PID control core, as a project of its own.
//
// THE LAW is one expression, in plane space: num_i = k_i soft_i e_i - d_i (v_i - v_ref_i) + alpha_i
// i_acc_i, then f = Lambda num (poles reading) or f = num (physical reading), then tau = J' f + C q_dot
// + g. law.rs holds the num expression itself; its two design inputs are the pole placement
// k = wn^2 - k_eff, d = 2 zeta wn - b_eff (design.rs) and the metric Lambda = (J M^-1 J^T)^-1
// (inertia.rs). Everything else here is the numbers that design is read from
// (gains/spec/class/budget/window), the loop one realization of it runs in (task_loop + opts), or what
// that loop acts on (the keepout family, escape, recruit).
//
// ONE LAW, TWO REALIZATIONS: this loop drives the arm (task_loop.rs, fixed base, fully actuated, the
// desired wrench maps through J' directly) and the biped (../g1-biped's src/standing_loop.rs, floating
// base, the wrench must be produced by ground contact instead). The realization differs; the law does not.
//
// It was extracted from the simu crate's `src/ga_pid/` once the dependency graph made the order
// obvious: the directory was already closed under itself — no `crate::` outside it, no model, no
// engine, no `build.rs` — and its two outward edges are the crates below it, `control-math` for the
// arithmetic and `control-base` for the two names the loop is written against: the `Plant` contract it
// drives (task_loop.rs is the ONLY file here that names a plant) and the efference copy it keeps.
//
// WHY IT IS A PROJECT OF ITS OWN: the law is stated once and read by two machines that share nothing
// else, so it may not live under either of them. The two products (`../z1-arm`, `../g1-biped`)
// consume this as a sibling path dependency (`{ path = "../control-ga-pid" }`) and implement `Plant`
// for their own plants; this crate implements nothing and drives whatever it is handed, which is what
// lets its own tests run without an engine.

pub mod box_ko;
pub mod budget;
pub mod class;
pub mod design;
pub mod escape;
pub mod gains;
pub mod inertia;
pub mod keepout;
pub mod law;
pub mod opts;
pub mod plane_ko;
pub mod recruit;
pub mod spec;
pub mod sphere_ko;
pub mod task_loop;
pub mod window;

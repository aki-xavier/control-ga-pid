// opts.rs — PlaneTaskLoopOpts, the configuration surface of the unified
// task-space loop (task_loop.rs), plus GainMode, which selects which invariant the design
// holds. Blocks are grouped by concern (planes/gains, integral, contact, impedance, escape) and
// derive Default; the contact schedule and ref_dq are deliberately not here (set by the study).

use crate::keepout::Keepout;

/// GainMode selects which invariant the design holds, i.e. design space and metric: Poles = plane
/// (bivector) space with the inertia metric, per-plane (wn, zeta); Physical = plane space, no
/// shaping, per-plane (K, D); Joint = joint space, per-joint wn^2, driven through step_joint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GainMode {
    #[default]
    Poles,
    Physical,
    Joint,
}

/// GainOpts is the task design: what the planes are, which invariant the design holds, and its gains.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GainOpts {
    /// task planes: 3 (position) or 6 (position + world rotvec)
    pub m: usize,
    pub mode: GainMode,
    /// poles reading: per-plane bandwidth [rad/s] and damping ratio
    pub wn: Vec<f64>,
    pub zeta: Vec<f64>,
    /// physical reading: per-plane stiffness and damping [N/m, N*s/m and the rotational equivalents]
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    /// joint reading: the IDENTIFIED gravity stiffness per joint, diagnostic and NOT subtracted
    /// from wn^2 — step_joint already feeds the plant's bias + g forward, which the plant subtracts
    /// again (both signs measured, GA_PID_AUDIT.md #4; see `PlaneTaskLoop::per_plane_kd`).
    pub k_eff: Vec<f64>,
    /// joint damping [N.m.s/rad], added to the bias feedforward so the plant's dissipative torque is
    /// cancelled rather than acting as free damping. Empty means no compensation, the default.
    pub damp: Vec<f64>,
    /// passivity guard on the physical reading: raise D_i to 2 sqrt(K_i M_i)
    pub passivity: bool,
}

/// IntegralOpts is the integral tier and the command's saturation (gain, gate, anti-windup, clamp).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegralOpts {
    /// per-plane gain on the error integral, plus an optional gate that integrates only while
    /// |e_i| <= i_deadband; 0 keeps the ungated behaviour the benches publish.
    pub i_alpha: Vec<f64>,
    pub i_deadband: f64,
    /// i_anti_windup_off disables step_joint's conditional anti-windup (false = a saturated joint
    /// freezes its accumulator).
    pub i_anti_windup_off: bool,
    /// per-joint output clamp, applied after the bias and gravity terms
    pub u_lim: Vec<f64>,
}

/// ImpedanceOpts is the law's descending impedance channel (SPINAL_PROGRAM.md #4's substitute for
/// an antagonist pair): coactivation scales the spring by (1 + c) and its damping by sqrt(1 + c),
/// so the damping ratio survives; uniform across planes, so GA_PID_AUDIT.md #19 holds. 0 = shipped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImpedanceOpts {
    /// the descending activation level; 0 (the default) is the shipped loop
    pub coactivation: f64,
    /// the ramp rate [1/s] carrying the passivity bound (a stiffness that can jump injects
    /// `1/2 dk e^2` while deflected). <= 0 applies the level directly, a caller's statement.
    pub coactivation_rate: f64,
}

/// AvoidOpts is the whole-arm keep-out escape (escape.rs): the keep-out set, the link radii
/// it is inflated by, and the switch between a metric-orthogonal escape and a task-priority one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AvoidOpts {
    pub full_body: bool,
    pub body_r: f64,
    pub body_margin: f64,
    pub body_rs: Vec<f64>,
    pub keepouts: Vec<Keepout>,
    /// escape_priority switches the escape from the metric-orthogonal null space (cannot fight the
    /// task) to a task-priority mode: full joint space, the task fading as the chain sinks in.
    pub escape_priority: bool,
    /// recruit is the minimal-authority-first switch for the whole-arm escape (SPINAL_PROGRAM.md
    /// #8): each joint's effort limit is the size of its unit, and the DEMAND goes to the cheap
    /// joints first (see `Recruitment`). Default false is the shipped loop.
    pub recruit: bool,
    /// ko_margin inflates the keep-outs for the end-effector target shaping
    pub ko_margin: f64,
    /// disable_goal_shaping turns OFF the end-effector safe_target shaping (false = shaping on).
    pub disable_goal_shaping: bool,
}

/// PlaneTaskLoopOpts is the single configuration surface. Lengths must match the plane count m
/// where they are read; shorter arrays fall back to documented defaults.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlaneTaskLoopOpts {
    pub gains: GainOpts,
    pub integral: IntegralOpts,
    pub impedance: ImpedanceOpts,
    pub avoid: AvoidOpts,
}

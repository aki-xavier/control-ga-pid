// opts.rs — PlaneTaskLoopOpts: configuration surface of the unified task-space loop (task_loop.rs).

use crate::keepout::Keepout;

/// Which invariant the design holds: Poles = plane space with the inertia metric (wn, zeta),
/// Physical = plane space unshaped (K, D), Joint = joint space (wn^2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GainMode {
    #[default]
    Poles,
    Physical,
    Joint,
}

/// The task design: what the planes are, which invariant the design holds, and its gains.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GainOpts {
    /// task plane count: 3 (position) or 6 (position + world rotvec)
    pub m: usize,
    pub mode: GainMode,
    /// poles reading: per-plane bandwidth [rad/s] and damping ratio
    pub wn: Vec<f64>,
    pub zeta: Vec<f64>,
    /// physical reading: per-plane stiffness and damping (N/m, N*s/m and rotational equivalents)
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    /// joint reading: identified gravity stiffness per joint, diagnostic and NOT subtracted from wn^2.
    pub k_eff: Vec<f64>,
    /// joint damping [N.m.s/rad] added to the bias feedforward; empty = no compensation (default).
    pub damp: Vec<f64>,
    /// passivity guard on the physical reading: raise D_i to 2 sqrt(K_i M_i)
    pub passivity: bool,
}

/// The integral tier and the command's saturation: gain, gate, anti-windup, clamp.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegralOpts {
    /// per-plane integral gain; integration is gated to |e_i| <= i_deadband, and 0 keeps it ungated.
    pub i_alpha: Vec<f64>,
    pub i_deadband: f64,
    /// true disables step_joint's conditional anti-windup; false freezes a saturated joint's accumulator.
    pub i_anti_windup_off: bool,
    /// per-joint output clamp, applied after the bias and gravity terms
    pub u_lim: Vec<f64>,
}

/// The law's descending impedance channel: the spring by (1 + c), the damping by sqrt(1 + c), so the
/// damping ratio survives; uniform across planes, so the presented stiffness stays symmetric.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImpedanceOpts {
    /// the descending activation level; 0 (the default) is the shipped loop
    pub coactivation: f64,
    /// ramp rate [1/s] carrying the passivity bound; <= 0 applies the level directly.
    pub coactivation_rate: f64,
}

/// Whole-arm keep-out escape (escape.rs): the keep-out set, the link radii, and the escape mode.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AvoidOpts {
    pub full_body: bool,
    pub body_r: f64,
    pub body_margin: f64,
    pub body_rs: Vec<f64>,
    pub keepouts: Vec<Keepout>,
    /// true switches the escape to task-priority; false uses the metric-orthogonal null space.
    pub escape_priority: bool,
    /// true = minimal-authority-first: each joint's effort limit is its unit, demand goes cheapest first.
    pub recruit: bool,
    /// ko_margin inflates the keep-outs for the end-effector target shaping
    pub ko_margin: f64,
    /// disables the end-effector safe_target shaping (false = shaping on)
    pub disable_goal_shaping: bool,
}

/// The single configuration surface: array lengths must match m where read, else defaults apply.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlaneTaskLoopOpts {
    pub gains: GainOpts,
    pub integral: IntegralOpts,
    pub impedance: ImpedanceOpts,
    pub avoid: AvoidOpts,
}

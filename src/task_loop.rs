// task_loop.rs — the unified task-space loop: one law, two parameterizations.
//   tau = J' f + C q_dot + g;  num_i = k_i soft_i e_i - d_i v_task,i + alpha_i i_acc_i
//   f = Lambda_m (num) in the poles reading, f = num in the physical reading.
// J and Lambda come from one task Jacobian, so the plane count m is data, not controller identity.
// The readings are NOT interchangeable: Poles keeps the closed-loop poles and bandwidth
// configuration-invariant (the impedance K = Lambda wn^2 then varies), while Physical fixes the
// impedance across the workspace (the pole pattern follows Lambda). Task selection is by choosing
// J, never by zeroing planes' gains.
//
// THE FRAME (GA_PID_AUDIT.md #19): every task-space quantity here is in world axes ABOUT THE TIP,
// in [v; w] order — the error's first three slots are the tip's point-image difference and the last
// three the world rotvec; J's rows are the tip's linear then angular rows (pinned against FK finite
// differences in simu's tests/urdf.rs). The metric is Lambda = (J M^-1 J^T)^-1, refreshed as the
// configuration moves; a non-uniform Poles configuration is reported when built.

use crate::escape::TaskAvoidance;
use crate::gains::PlaneGains;
use crate::inertia::{effective_task_mass, fill, passivity_floor, task_space_inertia};
use crate::keepout::{Keepout, BODY_LINK_RADII, TOOL_REACH};
use crate::law;
use crate::opts::{GainMode, PlaneTaskLoopOpts};
use crate::recruit::Recruitment;
use control_base::efference::Efference;
use control_base::plant::Plant;
use control_math::mat::Mat;
use control_math::quat::Quat;
use control_math::task_space_bridge::TaskSpaceBridge;
use control_math::vec3::Vec3;

const DBG_INTERNAL: bool = false;

/// NONCONSERVATIVE_POLES_WARNED reports once per process a Poles loop whose per-plane (wn, zeta)
/// or soft schedule is not uniform: it then presents K = Lambda diag(wn^2 soft), symmetric only
/// when that diagonal is (GA_PID_AUDIT.md #19, pinned by tests/task_frame.rs). Every shipped Poles
/// path is uniform.
static NONCONSERVATIVE_POLES_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn warn_nonconservative_poles() {
    if !NONCONSERVATIVE_POLES_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "simu: a Poles loop's gains are not uniform across planes, so the stiffness it\
             presents (K = Lambda diag(wn^2 soft)) is asymmetric — non-conservative\
             (GA_PID_AUDIT.md #19). The shipped paths are uniform; the Physical reading is\
             the frame for a per-plane impedance."
        );
    }
}

/// PlaneTaskLoop is the state: gains, integral accumulator, contact fade factors, the cached shaping matrix and the refresh bookkeeping.
#[derive(Clone, Debug)]
pub struct PlaneTaskLoop {
    pub m: usize,
    pub mode: GainMode,
    pub n: usize,
    pub wn: Vec<f64>,
    pub zeta: Vec<f64>,
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub k_eff: Vec<f64>,
    pub damp: Vec<f64>,
    pub i_alpha: Vec<f64>,
    pub i_deadband: f64,
    pub i_aw_off: bool,
    pub i_acc: Vec<f64>,
    pub passivity: bool,
    pub k_ratio: f64,
    pub f_floor: f64,
    pub f_tau: f64,
    pub share: Vec<f64>,
    pub force_target: Vec<f64>,
    /// ref_dq is the Lambda cache's refresh threshold: recomputed once max|dq| since the last refresh
    /// exceeds it (<= 0 refreshes every tick). Not on the options surface.
    pub ref_dq: f64,
    pub soft: Vec<f64>,
    pub full_body: bool,
    pub body_r: f64,
    pub body_margin: f64,
    pub body_rs: Vec<f64>,
    pub keepouts: Vec<Keepout>,
    pub escape_priority: bool,
    /// recruit is the minimal-authority-first switch (SPINAL_PROGRAM.md #8): the escape's DEMAND is
    /// split by each joint's effort limit, so the cheap joints take it first.
    pub recruit: bool,
    /// escape_dq is the joint-space escape increment from the last solve, for the caller's readout.
    pub escape_dq: Vec<f64>,
    pub ko_margin: f64,
    pub shape_goal: bool,
    pub u_lim: Vec<f64>,
    /// coact_cmd is the descending impedance channel (SPINAL_PROGRAM.md Phase 1): stiffness scales
    /// by (1 + coact), damping by sqrt(1 + coact), so the equilibrium point does not move. The
    /// APPLIED level ramps toward it at coact_rate [1/s]; 0 is the shipped default.
    pub coact_cmd: f64,
    pub coact_rate: f64,
    pub coact: f64,
    pub lam: Mat,
    pub q_ref: Vec<f64>,
    /// eff is the arm's efference copy (`../control-base/src/efference.rs`): one channel per joint, the torque this
    /// loop COMMANDED against the generalized force the plant's step says it APPLIED, and their
    /// difference — the applied side reconstructed from the velocity the plant's integrator produced
    /// (simu's `CEnginePlant::step` inverts to the torque it must have applied, per joint). A COPY only:
    /// nothing here computes a number in the law from it.
    ///
    /// The caller can break the pair: the plant must be stepped by the same `dt`, once per call, with
    /// its viscous damping equal to this loop's `damp` (an empty `damp` reads it as residual).
    pub eff: Efference,
    /// The copy's ledger: the plant's model terms at the state the command was formed from.
    eff_cmd: Vec<f64>,
    eff_v: Vec<f64>,
    eff_m: Mat,
    eff_rg: Vec<f64>,
    eff_dt: f64,
}

impl PlaneTaskLoop {
    /// new normalizes an option set into a loop; m comes from the design, else from the gain array.
    pub fn new(n: usize, o: PlaneTaskLoopOpts) -> PlaneTaskLoop {
        let mut m = o.gains.m;
        if m == 0 {
            m = if !o.gains.wn.is_empty() {
                o.gains.wn.len()
            } else {
                o.gains.k.len()
            };
        }
        if m == 0 {
            m = 3;
        }
        if o.gains.mode == GainMode::Poles {
            let nonuniform = |v: &[f64]| v.len() > 1 && v.iter().any(|x| *x != v[0]);
            if nonuniform(&o.gains.wn) || nonuniform(&o.gains.zeta) {
                warn_nonconservative_poles();
            }
        }
        let mut lp = PlaneTaskLoop {
            m,
            mode: o.gains.mode,
            n,
            wn: fill(&o.gains.wn, m, 0.0),
            zeta: fill(&o.gains.zeta, m, 0.9),
            k: fill(&o.gains.k, m, 0.0),
            d: fill(&o.gains.d, m, 0.0),
            k_eff: fill(&o.gains.k_eff, n, 0.0),
            damp: fill(&o.gains.damp, n, 0.0),
            passivity: o.gains.passivity,
            i_alpha: fill(&o.integral.i_alpha, m, 0.0),
            i_deadband: o.integral.i_deadband,
            i_aw_off: o.integral.i_anti_windup_off,
            i_acc: vec![0.0; m],
            u_lim: o.integral.u_lim.clone(),
            // the contact schedule starts OPEN (k_ratio 1 = no softening); simu's bench_contact::contact_schedule sets it
            k_ratio: 1.0,
            f_floor: 0.0,
            f_tau: 0.0,
            share: vec![0.0; m],
            force_target: vec![0.0; m],
            ref_dq: 0.0,
            soft: vec![1.0; m],
            full_body: o.avoid.full_body,
            body_r: o.avoid.body_r,
            body_margin: o.avoid.body_margin,
            body_rs: o.avoid.body_rs.clone(),
            keepouts: o.avoid.keepouts.clone(),
            escape_priority: o.avoid.escape_priority,
            recruit: o.avoid.recruit,
            escape_dq: Vec::new(),
            ko_margin: o.avoid.ko_margin,
            shape_goal: !o.avoid.disable_goal_shaping,
            // the channel's own entry, so `rate <= 0` applies here too; LEVEL 0 SKIPS THE CALL
            coact_cmd: 0.0,
            coact_rate: 0.0,
            coact: 0.0,
            lam: Mat::zeros(0, 0),
            q_ref: Vec::new(),
            eff: Efference::new(),
            eff_cmd: Vec::new(),
            eff_v: Vec::new(),
            eff_m: Mat::zeros(0, 0),
            eff_rg: Vec::new(),
            eff_dt: 0.0,
        };
        // the arm's channels are joint torques, so the unit is N.m (the type's default is the legs' "N")
        lp.eff.unit = "N.m".to_string();
        if o.impedance.coactivation != 0.0 {
            lp.set_coactivation(o.impedance.coactivation, o.impedance.coactivation_rate);
        }
        lp
    }

    /// new_position is the free-space position entry: three planes, poles reading, no contact layers.
    pub fn new_position(n: usize, wn: f64, zeta: f64) -> PlaneTaskLoop {
        PlaneTaskLoop::new(
            n,
            PlaneTaskLoopOpts {
                gains: crate::opts::GainOpts {
                    m: 3,
                    mode: GainMode::Poles,
                    wn: vec![wn, wn, wn],
                    zeta: vec![zeta, zeta, zeta],
                    ..Default::default()
                },
                ..Default::default()
            },
        )
    }

    /// new_pose is the full-pose entry: six planes, poles reading by default.
    pub fn new_pose(n: usize, wn: f64, zeta: f64) -> PlaneTaskLoop {
        PlaneTaskLoop::new(
            n,
            PlaneTaskLoopOpts {
                gains: crate::opts::GainOpts {
                    m: 6,
                    mode: GainMode::Poles,
                    wn: vec![wn; 6],
                    zeta: vec![zeta; 6],
                    ..Default::default()
                },
                ..Default::default()
            },
        )
    }

    /// new_physical is the impedance entry: real K and D, read as a fixed mechanical impedance.
    pub fn new_physical(n: usize, k: &[f64], d: &[f64]) -> PlaneTaskLoop {
        PlaneTaskLoop::new(
            n,
            PlaneTaskLoopOpts {
                gains: crate::opts::GainOpts {
                    m: k.len(),
                    mode: GainMode::Physical,
                    k: k.to_vec(),
                    d: d.to_vec(),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
    }

    /// new_joint is the joint-space entry: the planes are the joints, the metric is the mass matrix's
    /// diagonal, no frame map appears, and k_eff is diagnostic (see `per_plane_kd`). Use step_joint.
    pub fn new_joint(n: usize, wn: f64, zeta: f64, k_eff: &[f64]) -> PlaneTaskLoop {
        PlaneTaskLoop::new(
            n,
            PlaneTaskLoopOpts {
                gains: crate::opts::GainOpts {
                    m: n,
                    mode: GainMode::Joint,
                    wn: vec![wn; n],
                    zeta: vec![zeta; n],
                    k_eff: k_eff.to_vec(),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
    }

    /// take_efference samples the copy of the LAST command: the torque this loop returned against the
    /// generalized force the plant's step implies it applied (no sample until both sides exist).
    fn take_efference(&mut self, plant: &mut dyn Plant) {
        if self.eff_cmd.len() != self.n
            || self.eff_v.len() != self.n
            || self.eff_rg.len() != self.n
            || self.eff_dt <= 0.0
        {
            return;
        }
        let v = plant.joint_velocities();
        if v.len() < self.n {
            return;
        }
        let dt = self.eff_dt;
        // the plant's integrator is semi-implicit Euler (v_new = v + dt qdd): a plain division
        let mut qdd = vec![0.0; self.n];
        for i in 0..self.n {
            qdd[i] = (v[i] - self.eff_v[i]) / dt;
        }
        let mut applied = vec![0.0; self.n];
        for i in 0..self.n {
            let di = if i < self.damp.len() {
                self.damp[i]
            } else {
                0.0
            };
            let mut t = self.eff_rg[i] + di * self.eff_v[i];
            for j in 0..self.n {
                let mut mij = if self.eff_m.rows == self.n {
                    self.eff_m.at(i, j)
                } else {
                    0.0
                };
                if i == j {
                    mij += dt * di;
                }
                t += mij * qdd[j];
            }
            applied[i] = t;
        }
        self.eff.observe(&self.eff_cmd, &applied);
    }

    /// record_command caches what `take_efference` needs (torque, model terms); called AFTER the clamp.
    fn record_command(&mut self, plant: &mut dyn Plant, tau: &[f64], dt: f64) {
        self.eff_cmd = tau.to_vec();
        self.eff_v = plant.joint_velocities();
        self.eff_m = plant.mass_matrix();
        self.eff_rg = plant.bias_torques();
        let g = plant.gravity_torques();
        for i in 0..self.eff_rg.len().min(g.len()) {
            self.eff_rg[i] += g[i];
        }
        self.eff_dt = dt;
    }

    /// integrate_gated advances the integral tier by the anti-windup gate: a joint integrates while
    /// its command is inside the limit or is being pulled back toward it, and freezes otherwise.
    fn integrate_gated(
        &self,
        i_acc: &[f64],
        err: &[f64],
        tau: &[f64],
        u_lim: &[f64],
        dt: f64,
    ) -> Vec<f64> {
        let mut out = vec![0.0; i_acc.len()];
        for i in 0..i_acc.len() {
            out[i] = i_acc[i];
            if i >= u_lim.len() {
                continue;
            }
            let inside = tau[i].abs() <= u_lim[i];
            let pulls_in =
                (tau[i] > u_lim[i] && err[i] < 0.0) || (tau[i] < -u_lim[i] && err[i] > 0.0);
            if inside || pulls_in {
                out[i] += err[i] * dt;
            }
        }
        out
    }

    /// step_joint runs the joint design: the error is q_des - q, the per-joint gains are
    /// (wn^2, 2 zeta wn) with k_eff diagnostic rather than a term, and the mass matrix turns the
    /// commanded acceleration into torque. No Jacobian enters, so no two inner products are mixed.
    /// v_des is the reference joint velocity; the integral tier updates AFTER the torque is known.
    pub fn step_joint(
        &mut self,
        plant: &mut dyn Plant,
        q_des: &[f64],
        v_des: &[f64],
        dt: f64,
    ) -> Vec<f64> {
        self.take_efference(plant);
        let m = self.m;
        let q = plant.joint_positions();
        let v = plant.joint_velocities();
        let mut dq = vec![0.0; m];
        for i in 0..m {
            dq[i] = q_des[i] - q[i];
        }
        let mut a = vec![0.0; m];
        for i in 0..m {
            let (kv, dv) = self.per_plane_kd(i);
            let vd = if i < v_des.len() { v_des[i] } else { 0.0 };
            // the law with the commanded joint velocity as its reference (v_ref = v_des): the joint
            // reading states the same damping on the velocity error the task readings do
            let g = PlaneGains {
                kappa_p: kv,
                kappa_d: dv,
                kappa_i: self.i_alpha[i],
            };
            a[i] = law::num(g, dq[i], v[i], vd, self.i_acc[i]);
        }
        // same implicit computed torque as step: (M + dt D) a + rg + D v lands on a exactly
        let massm = plant.mass_matrix();
        let mut mm = massm.clone();
        for i in 0..self.n {
            if i < self.damp.len() && self.damp[i] != 0.0 {
                mm.set(i, i, mm.at(i, i) + dt * self.damp[i]);
            }
        }
        let mut tau = mm.mul_vec(&a);
        let bias = plant.bias_torques();
        let g = plant.gravity_torques();
        for i in 0..self.n {
            tau[i] += bias[i] + g[i];
            if i < self.damp.len() && self.damp[i] != 0.0 {
                tau[i] += self.damp[i] * v[i];
            }
        }
        // integral tier, before the clamp but after the unclamped torque is known:
        if self.u_lim.len() == self.n && !self.i_aw_off {
            let mut err = dq.clone();
            for i in 0..m {
                // no integral gain, or an error outside the deadband: no integral input this tick
                if self.i_alpha[i] == 0.0
                    || (self.i_deadband > 0.0 && dq[i].abs() > self.i_deadband)
                {
                    err[i] = 0.0;
                }
            }
            self.i_acc = self.integrate_gated(&self.i_acc, &err, &tau, &self.u_lim, dt);
        } else {
            for i in 0..m {
                if self.i_alpha[i] == 0.0 {
                    continue;
                }
                if self.i_deadband > 0.0 && dq[i].abs() > self.i_deadband {
                    continue;
                }
                self.i_acc[i] += dq[i] * dt;
            }
        }
        if self.u_lim.len() == self.n {
            for i in 0..self.n {
                if self.u_lim[i] > 0.0 {
                    if tau[i] > self.u_lim[i] {
                        tau[i] = self.u_lim[i];
                    }
                    if tau[i] < -self.u_lim[i] {
                        tau[i] = -self.u_lim[i];
                    }
                }
            }
        }
        self.record_command(plant, &tau, dt);
        tau
    }

    /// whole_arm_escape_dqs samples the chain axis (base -> link frames -> tool flange) plus quarter
    /// points per segment, takes each sample's worst keep-out escape inflated by the per-segment
    /// link radius, and maps it (damped LS, stacked segment Jacobians) to a joint-space escape
    /// increment — the target for the null-space secondary task in step_ff, kept in `escape_dq`.
    fn whole_arm_escape_dqs(&mut self, plant: &mut dyn Plant) -> (Vec<f64>, f64) {
        let n = self.n;
        let mut vmax = 0.0;
        let mut pts: Vec<Vec3> = Vec::new();
        let mut jjs: Vec<Mat> = Vec::new();
        pts.push(Vec3::new(0.0, 0.0, 0.0));
        jjs.push(Mat::zeros(3, n));
        for i in 0..n {
            let (p, rot) = plant.link_frame(i);
            pts.push(p);
            jjs.push(plant.link_jacobian(i));
            if i == n - 1 {
                let u = rot.mul_vec3(Vec3::new(1.0, 0.0, 0.0)).normalized();
                pts.push(p.add(u.scale(TOOL_REACH)));
                jjs.push(plant.compute_jacobian());
            }
        }
        let mut rad: Vec<f64> = Vec::new();
        if n == BODY_LINK_RADII.len() {
            rad.push(BODY_LINK_RADII[0]);
            for seg in 1..n {
                rad.push(BODY_LINK_RADII[seg - 1]);
            }
            rad.push(BODY_LINK_RADII[n - 1]);
        } else {
            rad = vec![self.body_r; n + 1];
        }
        if self.body_rs.len() == n + 1 {
            rad = self.body_rs.clone();
        }
        let mut sp: Vec<Vec3> = Vec::new();
        let mut sj: Vec<Mat> = Vec::new();
        let mut sr: Vec<f64> = Vec::new();
        sp.push(pts[0]);
        sj.push(jjs[0].clone());
        sr.push(rad[0]);
        for seg in 0..pts.len() - 1 {
            let pa = pts[seg];
            let pb = pts[seg + 1];
            if pa.sub(pb).norm() > 1e-9 {
                for s in [0.25, 0.5, 0.75] {
                    sp.push(pa.add(pb.sub(pa).scale(s)));
                    sj.push(jjs[seg].scale(1.0 - s).add(&jjs[seg + 1].scale(s)));
                    sr.push(rad[seg]);
                }
            }
            sp.push(pts[seg + 1]);
            sj.push(jjs[seg + 1].clone());
            let last = seg == pts.len() - 2;
            sr.push(if last || rad[seg] > rad[seg + 1] {
                rad[seg]
            } else {
                rad[seg + 1]
            });
        }
        let mut rowmat: Vec<Vec<f64>> = Vec::new();
        let mut es: Vec<f64> = Vec::new();
        for i in 0..sp.len() {
            let mut best = Vec3::new(0.0, 0.0, 0.0);
            let mut bn = 0.0;
            for ko in &self.keepouts {
                let e = ko.escape(sp[i], sr[i], self.body_margin);
                let en = e.norm();
                if en > bn {
                    bn = en;
                    best = e;
                }
            }
            if bn > vmax {
                vmax = bn;
            }
            if bn > 1e-12 {
                let ji = &sj[i];
                let mut row = vec![0.0; n];
                for r in 0..3 {
                    for cc in 0..n {
                        row[cc] = ji.at(r, cc);
                    }
                    rowmat.push(row.clone());
                    es.push(best.x);
                    es.push(best.y);
                    es.push(best.z);
                }
            }
        }
        let mut dqs = vec![0.0; n];
        if !es.is_empty() {
            let am = Mat::from_rows(&rowmat);
            dqs = TaskSpaceBridge::new(n, 1e-6).step(&am, &es);
        }
        // Recruitment acts on the DEMAND, not through the solve's metric: the stacked Jacobian is
        // tall, so its least-squares answer is unique and a price table cannot move it. Uniform
        // prices leave this the shipped loop bit for bit.
        if self.recruit {
            let w = self.recruit_weights();
            for (i, d) in dqs.iter_mut().enumerate() {
                *d *= w.get(i).copied().unwrap_or(1.0);
            }
        }
        self.escape_dq = dqs.clone();
        (dqs, vmax)
    }

    /// recruit_weights is the escape's own price table, read from `u_lim`: a joint's effort limit is
    /// the size of its unit, which is what the size principle orders by. No limits of the right
    /// length means unit weights, i.e. the shipped solve.
    fn recruit_weights(&self) -> Vec<f64> {
        let lim: Vec<f64> = if self.u_lim.len() == self.n {
            self.u_lim.clone()
        } else {
            vec![1.0; self.n]
        };
        let mut r = Recruitment::new(lim);
        r.on = true;
        r.weights()
    }

    /// set_coactivation is the impedance channel's command entry: the level and the ramp rate that
    /// bounds how fast the stiffness may move. rate <= 0 applies the level directly; level 0 is the
    /// shipped loop and the default.
    pub fn set_coactivation(&mut self, level: f64, rate: f64) {
        self.coact_cmd = level.max(0.0);
        self.coact_rate = rate;
        if rate <= 0.0 {
            self.coact = self.coact_cmd;
        }
    }

    /// coact_scale applies the impedance channel to one (kp, kd) pair: stiffness by (1 + c), damping
    /// by sqrt(1 + c), so the damping ratio survives and K = Lambda diag(wn^2 (1+c)) stays a uniform
    /// scalar times the design's — GA_PID_AUDIT.md #19 holds because the scale is uniform.
    fn coact_scale(&self, kp: f64, kd: f64) -> (f64, f64) {
        if self.coact <= 0.0 {
            return (kp, kd);
        }
        let s = 1.0 + self.coact;
        (kp * s, kd * s.sqrt())
    }

    /// per_plane_kd returns the stiffness and damping the law uses on one plane in the current
    /// reading (poles/joint derive them from the design tuple, physical takes them as given).
    /// Public because the integration tests read it directly.
    ///
    /// The joint reading's stiffness is wn^2, NOT wn^2 - k_eff: step_joint feeds the plant's bias +
    /// gravity forward already, so subtracting k_eff removed stiffness the design asked for (with
    /// both signs over the joints, GA_PID_AUDIT.md #4/#5).
    pub fn per_plane_kd(&self, i: usize) -> (f64, f64) {
        if self.mode == GainMode::Poles || self.mode == GainMode::Joint {
            let alpha = if i < self.i_alpha.len() {
                self.i_alpha[i]
            } else {
                0.0
            };
            let g = law::gains(self.wn[i], self.zeta[i], alpha);
            return self.coact_scale(g.kappa_p, g.kappa_d);
        }
        self.coact_scale(self.k[i], self.d[i])
    }

    /// plane_masses: the cached shaping matrix's diagonal when valid, else the full-pose one.
    fn plane_masses(&self, plant: &mut dyn Plant) -> Vec<f64> {
        if self.lam.rows == self.m && self.lam.cols == self.m {
            return self.lam.diag();
        }
        let j6 = plant.compute_full_jacobian();
        effective_task_mass(&plant.mass_matrix(), &j6, self.n)
    }

    /// step_points is the multi-point task reading of the same law: n_p task points (three planes per
    /// point), with cur in the row order of the stacked task Jacobian (compute_jacobian, 3 n_p x n)
    /// and targets as their references. Free-space poles core only (see step/step_ff for the rest).
    pub fn step_points(
        &mut self,
        plant: &mut dyn Plant,
        cur: &[Vec3],
        targets: &[Vec3],
        dt: f64,
    ) -> Vec<f64> {
        self.take_efference(plant);
        let m = self.m;
        let j = plant.compute_jacobian();
        let mut e = vec![0.0; m];
        for i in 0..m {
            let p = i / 3;
            if p >= targets.len() || p >= cur.len() {
                break;
            }
            let d = targets[p].sub(cur[p]);
            e[i] = if i % 3 == 0 {
                d.x
            } else if i % 3 == 1 {
                d.y
            } else {
                d.z
            };
        }
        for i in 0..m {
            if self.i_alpha[i] == 0.0 {
                continue;
            }
            if self.i_deadband > 0.0 && e[i].abs() > self.i_deadband {
                continue;
            }
            self.i_acc[i] += e[i] * dt;
        }
        let v_task = j.mul_vec(&plant.joint_velocities());
        let mut num = vec![0.0; m];
        for i in 0..m {
            let (kv, dv) = self.per_plane_kd(i);
            // no reference velocity in this reading, so the law's v_ref term is zero here
            let g = PlaneGains {
                kappa_p: kv,
                kappa_d: dv,
                kappa_i: self.i_alpha[i],
            };
            num[i] = law::num(g, e[i], v_task[i], 0.0, self.i_acc[i]);
        }
        if DBG_INTERNAL {
            eprintln!(
                "  [dbg] e2={:.5} vt2={:.5} num2={:.4}",
                e[2], v_task[2], num[2]
            );
        }
        let massm = plant.mass_matrix();
        let lam = task_space_inertia(&massm, &j, self.n);
        let f = lam.mul_vec(&num);
        let mut tau = j.transposed().mul_vec(&f);
        let bias = plant.bias_torques();
        let g = plant.gravity_torques();
        for i in 0..self.n {
            tau[i] += bias[i] + g[i];
        }
        // null-space damping with the dynamically consistent projector: tau_null = -D q_dot +
        // J' Lambda J M^-1 (D q_dot) damps only the redundant degrees of freedom.
        if self.u_lim.len() == self.n {
            for i in 0..self.n {
                if self.u_lim[i] > 0.0 {
                    if tau[i] > self.u_lim[i] {
                        tau[i] = self.u_lim[i];
                    }
                    if tau[i] < -self.u_lim[i] {
                        tau[i] = -self.u_lim[i];
                    }
                }
            }
        }
        self.record_command(plant, &tau, dt);
        tau
    }

    /// step computes the joint torques for one control sample: step_ff with empty references.
    pub fn step(
        &mut self,
        plant: &mut dyn Plant,
        target_pos: Vec3,
        target_quat: Quat,
        dt: f64,
        contact_f: &[f64],
    ) -> Vec<f64> {
        self.step_ff(plant, target_pos, target_quat, &[], &[], dt, contact_f)
    }

    /// step_ff is the feedforward form of step: the reference's velocity and acceleration enter the
    /// design directly (F = Lambda (a_ref + wn^2 e + 2 zeta wn (v_ref - v))); empty refs = static.
    #[allow(clippy::too_many_arguments)]
    pub fn step_ff(
        &mut self,
        plant: &mut dyn Plant,
        target_pos: Vec3,
        target_quat: Quat,
        ref_vel: &[f64],
        ref_acc: &[f64],
        dt: f64,
        contact_f: &[f64],
    ) -> Vec<f64> {
        self.take_efference(plant);
        let m = self.m;
        // the applied co-activation ramps toward the command once per tick, before any gain is read
        if self.coact_rate > 0.0 && dt > 0.0 && self.coact != self.coact_cmd {
            let step = self.coact_rate * dt;
            let d = self.coact_cmd - self.coact;
            self.coact += if d.abs() <= step {
                d
            } else {
                step * d.signum()
            };
        }
        let j = if m == 6 {
            plant.compute_full_jacobian()
        } else {
            plant.compute_jacobian()
        };
        let q = plant.joint_positions();
        let needs_lam =
            self.mode == GainMode::Poles || (self.passivity && self.mode == GainMode::Physical);
        if needs_lam {
            let mut refresh = self.lam.rows != m;
            if !refresh {
                if self.ref_dq <= 0.0 {
                    refresh = true;
                } else {
                    let mut dq = 0.0;
                    for i in 0..q.len() {
                        let dd = (q[i] - self.q_ref[i]).abs();
                        if dd > dq {
                            dq = dd;
                        }
                    }
                    refresh = dq > self.ref_dq;
                }
            }
            if refresh {
                let massm = plant.mass_matrix();
                self.lam = task_space_inertia(&massm, &j, self.n);
                self.q_ref = q.clone();
            }
        }
        let (cur, cq) = plant.body_pose();
        // end-effector keep-out shaping: the target is projected out of the convex keep-out sets in
        // task space (escape.rs safe_target), the GOAL-side half of avoidance (the path-side
        // half being the null-space whole-arm escape below, full_body).
        let mut goal = target_pos;
        if self.shape_goal && !self.keepouts.is_empty() {
            let (sg, _, _) =
                TaskAvoidance.safe_target(cur, target_pos, &self.keepouts, self.ko_margin);
            goal = sg;
        }
        // THE ERROR CHANNEL IS SIX WORLD-AXIS SCALARS: the tip displacement and the world rotvec, not
        // the theory's one geometric object `B_e = -2 log(M_d ~M)` — that object's translation half is
        // the screw's MOMENT about the motor's origin, not the endpoint displacement, and feeding it
        // into these slots drives the arm off target (GA_PID_AUDIT.md #3).
        let mut e = vec![0.0; m];
        let dp = goal.sub(cur);
        e[0] = dp.x;
        e[1] = dp.y;
        e[2] = dp.z;
        if m == 6 {
            let rot = Quat::rotvec_between(target_quat, cq);
            e[3] = rot.x;
            e[4] = rot.y;
            e[5] = rot.z;
        }
        let v_task = j.mul_vec(&plant.joint_velocities());
        let vj = plant.joint_velocities();

        // contact schedule: soften the aligned planes towards k * k_ratio, faded with f_tau.
        if self.k_ratio < 1.0 {
            if self.mode == GainMode::Poles {
                // a Poles loop under the schedule softens planes independently, so its stiffness goes
                // non-conservative with the first anisotropic wrench (GA_PID_AUDIT.md #19)
                warn_nonconservative_poles();
            }
            let mut target = vec![1.0; m];
            let mut fmag = 0.0;
            if contact_f.len() >= 3 {
                fmag = (contact_f[0] * contact_f[0]
                    + contact_f[1] * contact_f[1]
                    + contact_f[2] * contact_f[2])
                    .sqrt();
            }
            if fmag > self.f_floor {
                for i in 0..3 {
                    let u = contact_f[i] / fmag;
                    target[i] = 1.0 - (1.0 - self.k_ratio) * u.abs();
                }
            }
            let alpha = if self.f_tau > 0.0 {
                (dt / self.f_tau).min(1.0)
            } else {
                1.0
            };
            for i in 0..m {
                self.soft[i] += (target[i] - self.soft[i]) * alpha;
            }
        } else {
            for i in 0..m {
                self.soft[i] = 1.0;
            }
        }

        // integral tier, gated when a deadband is set
        if self.i_deadband > 0.0 {
            for i in 0..m {
                if self.i_alpha[i] != 0.0 && e[i].abs() <= self.i_deadband {
                    self.i_acc[i] += e[i] * dt;
                }
            }
        } else {
            for i in 0..m {
                if self.i_alpha[i] != 0.0 {
                    self.i_acc[i] += e[i] * dt;
                }
            }
        }

        // per-plane damping: the configured value raised to the passivity floor, applied here rather
        // than written back into c.d so it cannot ratchet upward as the configuration moves
        let mut dv_eff = vec![0.0; m];
        let mut masses: Vec<f64> = Vec::new();
        if self.passivity && self.mode == GainMode::Physical {
            masses = self.plane_masses(plant);
        }
        for i in 0..m {
            let (_, dv) = self.per_plane_kd(i);
            let mut dd = dv;
            if masses.len() > i {
                dd = passivity_floor(self.k[i] * self.soft[i], dv, masses[i]);
            }
            dv_eff[i] = dd;
        }
        // plane-space damping mapping: corr = J M^-1 (D q_dot), added to the plane numerator before the
        // shaping, so the closed-loop task damping is the design's (a single 3x3 plane-space term).
        let mut corr = vec![0.0; m];
        if self.damp.len() == self.n {
            let mut dqv = vec![0.0; self.n];
            for i in 0..self.n {
                dqv[i] = self.damp[i] * vj[i];
            }
            let massm = plant.mass_matrix();
            let c3 = j.mul_vec(&massm.solve(&dqv));
            corr[..m].copy_from_slice(&c3[..m]);
        }
        let mut num = vec![0.0; m];
        for i in 0..m {
            let (kv, _) = self.per_plane_kd(i);
            let rv = if i < ref_vel.len() { ref_vel[i] } else { 0.0 };
            let ra = if i < ref_acc.len() { ref_acc[i] } else { 0.0 };
            // the contact schedule's soft factor and the passivity floor are folded into the gains the
            // law is handed, so the law's own expression is the one evaluated
            let g = PlaneGains {
                kappa_p: kv * self.soft[i],
                kappa_d: dv_eff[i],
                kappa_i: self.i_alpha[i],
            };
            num[i] = law::num(g, e[i], v_task[i], rv, self.i_acc[i]) + corr[i];
            if self.mode == GainMode::Poles {
                num[i] += ra;
            }
        }
        let mut f = vec![0.0; m];
        if self.mode == GainMode::Poles {
            f = self.lam.mul_vec(&num);
        } else {
            f[..m].copy_from_slice(&num[..m]);
        }
        for i in 0..m {
            let s = self.share[i];
            if s > 0.0 {
                f[i] = (1.0 - s) * f[i] + s * (self.force_target[i] - dv_eff[i] * v_task[i]);
            }
        }

        let mut tau = j.transposed().mul_vec(&f);
        let bias = plant.bias_torques();
        let g = plant.gravity_torques();
        for i in 0..self.n {
            tau[i] += bias[i] + g[i];
        }
        // Joint damping is compensated in plane space via the corr term above (the per-joint
        // feedforward forms destabilised the stiff wrist joint).
        //
        // whole-arm escape as a null-space secondary task: the escape PD's torque is projected by
        // N' = I - J' Jbar', so it holds the links clear of the keep-outs without disturbing the task:
        // tau_escape = (I - J' Lambda J M^-1) M a2.
        if self.full_body && !self.keepouts.is_empty() {
            let (dqs, vmax) = self.whole_arm_escape_dqs(plant);
            let (kv, dv) = self.per_plane_kd(0);
            let mut a2 = vec![0.0; self.n];
            for i in 0..self.n {
                a2[i] = kv * dqs[i] - dv * vj[i];
            }
            let massm = plant.mass_matrix();
            let ma2 = massm.mul_vec(&a2);
            if self.escape_priority {
                // task-priority: the escape acts in FULL joint space, the task fading as the chain
                // sinks into a keep-out, wt = 1 - vmax/0.02.
                let wt = if vmax > 0.0 {
                    (1.0 - vmax / 0.02).max(0.0)
                } else {
                    1.0
                };
                for i in 0..self.n {
                    tau[i] = wt * tau[i] + ma2[i];
                }
            } else {
                // metric-orthogonal: the escape cannot fight the task, so it only holds the boundary,
                // with no task fade.
                let mut lam_e = self.lam.clone();
                if lam_e.rows != j.rows {
                    lam_e = task_space_inertia(&massm, &j, self.n);
                }
                let la2 = lam_e.mul_vec(&j.mul_vec(&a2));
                let back = j.transposed().mul_vec(&la2);
                for i in 0..self.n {
                    tau[i] += ma2[i] - back[i];
                }
            }
        }
        if self.u_lim.len() == self.n {
            for i in 0..self.n {
                if self.u_lim[i] > 0.0 {
                    if tau[i] > self.u_lim[i] {
                        tau[i] = self.u_lim[i];
                    }
                    if tau[i] < -self.u_lim[i] {
                        tau[i] = -self.u_lim[i];
                    }
                }
            }
        }
        self.record_command(plant, &tau, dt);
        tau
    }
}

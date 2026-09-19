// task_loop.rs — one task-space law, two readings: f = Lambda num in Poles, f = num in Physical,
// then tau = J' f + C q_dot + g. `f` carries the ACCELERATION and `g` the weight; the error is
// world axes about the tip, in [v; w] order.

use crate::escape::TaskAvoidance;
use crate::gains::PlaneGains;
use crate::impedance::ImpedanceChannel;
use crate::inertia::{effective_task_mass, fill, passivity_floor, task_space_inertia};
use crate::keepout::{Keepout, BODY_LINK_RADII, TOOL_REACH};
use crate::law;
use crate::opts::{GainMode, PlaneTaskLoopOpts};
use crate::recruit::Recruitment;
use control_base::efference::Efference;
use control_base::plant::{Plant, TaskMap};
use control_math::mat::Mat;
use control_math::quat::Quat;
use control_math::lstsq::DampedLstsq;
use control_math::vec3::Vec3;

const DBG_INTERNAL: bool = false;

/// Set once per process for a Poles loop whose per-plane gains are non-uniform, so K is asymmetric.
static NONCONSERVATIVE_POLES_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn warn_nonconservative_poles() {
    if !NONCONSERVATIVE_POLES_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "simu: a Poles loop's gains are not uniform across planes, so the stiffness it\
             presents (K = Lambda diag(wn^2 soft)) is asymmetric — non-conservative.\
             The shipped paths are uniform; the Physical reading is\
             the frame for a per-plane impedance."
        );
    }
}

/// PlaneTaskLoop — task-space loop state: gains, integral, contact fade, cached shaping matrix.
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
    /// Lambda refresh threshold: recomputed once max|dq| since the last refresh exceeds it (<= 0 = every tick).
    pub ref_dq: f64,
    pub soft: Vec<f64>,
    pub full_body: bool,
    pub body_r: f64,
    pub body_margin: f64,
    pub body_rs: Vec<f64>,
    pub keepouts: Vec<Keepout>,
    pub escape_priority: bool,
    /// Minimal-authority-first escape: the demand is split by each joint's effort limit.
    pub recruit: bool,
    /// Joint-space escape increment from the last solve, for the caller's readout.
    pub escape_dq: Vec<f64>,
    pub ko_margin: f64,
    pub shape_goal: bool,
    pub u_lim: Vec<f64>,
    /// Impedance channel: kp by (1 + level), kd by sqrt(1 + level), so the equilibrium does not move; level 0 = default.
    pub impedance: ImpedanceChannel,
    pub lam: Mat,
    pub q_ref: Vec<f64>,
    /// Efference copy: the torque commanded against the force the plant's step implies it applied; a copy only.
    /// Contract: the caller steps the plant by the same `dt`, once per call, with its viscous damping equal to `damp`.
    pub eff: Efference,
    /// The copy's ledger: the plant's model terms at the state the command was formed from.
    eff_cmd: Vec<f64>,
    eff_v: Vec<f64>,
    eff_m: Mat,
    eff_rg: Vec<f64>,
    eff_dt: f64,
    /// structure_checked gates the once-per-loop plant-structure report (`check_structure`).
    structure_checked: bool,
}

impl PlaneTaskLoop {
    /// Builds a loop from an option set; m comes from the design, else from the gain array.
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
            // contact schedule starts OPEN: k_ratio 1 = no softening
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
            impedance: ImpedanceChannel::new(),
            lam: Mat::zeros(0, 0),
            q_ref: Vec::new(),
            eff: Efference::new(),
            eff_cmd: Vec::new(),
            eff_v: Vec::new(),
            eff_m: Mat::zeros(0, 0),
            eff_rg: Vec::new(),
            eff_dt: 0.0,
            structure_checked: false,
        };
        // joint torques, so the unit is N.m (the type's default is "N")
        lp.eff.unit = "N.m".to_string();
        if o.impedance.coactivation != 0.0 {
            lp.set_coactivation(o.impedance.coactivation, o.impedance.coactivation_rate);
        }
        lp
    }

    /// Free-space position entry: three planes, poles reading, no contact layers.
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

    /// Full-pose entry: six planes, poles reading.
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

    /// Impedance entry: real K and D, read as a fixed mechanical impedance.
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

    /// Joint-space entry: the planes are the joints, the metric the mass matrix's diagonal; k_eff is diagnostic.
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

    /// Reads the plant's DECLARED structure once and reports where this loop's direct map cannot hold.
    fn check_structure(&mut self, plant: &mut dyn Plant) {
        if self.structure_checked {
            return;
        }
        self.structure_checked = true;
        let s = plant.structure();
        if s.dof != self.n {
            eprintln!(
                "simu: this loop was built for {} generalized coordinates and its plant declares {} \
                 (PlantStructure::dof), so the Jacobian columns and the torque length cannot agree",
                self.n, s.dof
            );
        }
        if !s.all_driven() {
            let free = s.actuated.iter().filter(|a| !**a).count();
            eprintln!(
                "simu: the plant declares {free} undriven generalized coordinate(s), so a wrench \
                 demand is not directly realizable — tau = J' f + bias assumes every coordinate is \
                 driven, and an undriven one's row is a constraint the environment must satisfy"
            );
        }
        if s.base_is_a_state() {
            eprintln!(
                "simu: the plant's base is a STATE (base_dof = {}), so this loop's direct map is \
                 not the whole realization — the base rows must be sourced by contact, which the \
                 loop does not do",
                s.base_dof
            );
        }
        if !s.braced() {
            eprintln!(
                "simu: nothing holds the machine in six directions (no welded base and no welded \
                 contact), so a wrench demand written at a task frame has no source"
            );
        }
        // A POINT task has no rotation, so a six-plane reading asks for three numbers that are not
        // there. `task_pose` answers the identity and `task_full_jacobian` an empty matrix rather than
        // inventing rows, which is why this has to be said BEFORE either is used.
        if s.task_is_a_point() && self.m == 6 {
            eprintln!(
                "simu: the plant's task is a POINT ({}) and this loop reads six planes, so its last \
                 three are a rotation the task does not have (PlantStructure::task_map, \
                 task_is_a_point) — build a position loop (`new_position`) for it",
                s.task_map.name()
            );
        }
        if s.task_is_a_point() && self.shape_goal && !self.keepouts.is_empty() {
            eprintln!(
                "simu: the plant's task is a POINT ({}) and this loop shapes its goal out of keep-out \
                 sets, whose sampling walks the chain out to the tip — a point task is on no link and \
                 has no tip, so that last sample is dropped",
                s.task_map.name()
            );
        }
    }

    fn take_efference(&mut self, plant: &mut dyn Plant) {
        self.check_structure(plant);
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

    /// Caches what `take_efference` needs; call AFTER the clamp.
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

    /// A joint integrates while inside its limit or pulled back toward it, and freezes otherwise.
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

    /// Joint design: e = q_des - q, gains (wn^2, 2 zeta wn) with k_eff diagnostic, M turns acceleration into torque.
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
            // v_ref = v_des: the joint reading damps the velocity error like the task readings do
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
        // `f` carries the ACCELERATION and `g` the weight; on a floating base these leading rows are the CONTACT'S DEMAND
        for i in 0..self.n {
            tau[i] += bias[i] + g[i];
            if i < self.damp.len() && self.damp[i] != 0.0 {
                tau[i] += self.damp[i] * v[i];
            }
        }
        if self.u_lim.len() == self.n && !self.i_aw_off {
            let mut err = dq.clone();
            for i in 0..m {
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

    /// Samples the chain plus quarter points; the worst keep-out escape per point is mapped (damped LS) into `escape_dq`.
    fn whole_arm_escape_dqs(&mut self, plant: &mut dyn Plant) -> (Vec<f64>, f64) {
        let n = self.n;
        let mut vmax = 0.0;
        let mut pts: Vec<Vec3> = Vec::new();
        let mut jjs: Vec<Mat> = Vec::new();
        pts.push(Vec3::new(0.0, 0.0, 0.0));
        jjs.push(Mat::zeros(3, n));
        // bodies come from the plant's declared order: body i is the distal link of coordinate i
        let s = plant.structure();
        let bodies = s.bodies;
        // the TIP is only a place to sample when the task is a FRAME. A point task (the centre of mass)
        // is not on the chain and has no tip the way a link does, so this whole-body escape cannot
        // sample beyond the bodies themselves — reported once, where the structure is checked.
        let tip_frame = match &s.task_map {
            TaskMap::Frame(f) => Some(f.clone()),
            TaskMap::Point(_) => None,
        };
        let nb = bodies.len().min(n);
        for (i, name) in bodies.iter().enumerate().take(nb) {
            let (p, rot) = plant.body_frame(name);
            pts.push(p);
            jjs.push(plant.body_jacobian(name));
            if i + 1 == nb {
                if let Some(f) = &tip_frame {
                    let u = rot.mul_vec3(Vec3::new(1.0, 0.0, 0.0)).normalized();
                    pts.push(p.add(u.scale(TOOL_REACH)));
                    jjs.push(plant.frame_jacobian(f));
                }
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
            dqs = DampedLstsq::new(n, 1e-6).solve(&am, &es);
        }
        // recruitment acts on the DEMAND, not through the solve's metric: uniform prices leave it bit for bit
        if self.recruit {
            let w = self.recruit_weights();
            for (i, d) in dqs.iter_mut().enumerate() {
                *d *= w.get(i).copied().unwrap_or(1.0);
            }
        }
        self.escape_dq = dqs.clone();
        (dqs, vmax)
    }

    /// The escape's price table from `u_lim` (effort limit = unit size); a wrong-length `u_lim` gives unit weights.
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

    /// Commands the impedance level and ramp rate; rate <= 0 applies the level directly, level 0 is the default.
    pub fn set_coactivation(&mut self, level: f64, rate: f64) {
        self.impedance.set(level, rate);
    }

    /// Stiffness and damping for one plane in the current reading; the joint reading's is wn^2, NOT wn^2 - k_eff, and the impedance scale applies here.
    pub fn per_plane_kd(&self, i: usize) -> (f64, f64) {
        if self.mode == GainMode::Poles || self.mode == GainMode::Joint {
            let alpha = if i < self.i_alpha.len() {
                self.i_alpha[i]
            } else {
                0.0
            };
            let g = law::gains(self.wn[i], self.zeta[i], alpha);
            return self.impedance.scale(g.kappa_p, g.kappa_d);
        }
        self.impedance.scale(self.k[i], self.d[i])
    }

    /// The cached shaping matrix's diagonal when valid, else the full-pose one.
    fn plane_masses(&self, plant: &mut dyn Plant) -> Vec<f64> {
        if self.lam.rows == self.m && self.lam.cols == self.m {
            return self.lam.diag();
        }
        let j6 = plant.task_full_jacobian();
        effective_task_mass(&plant.mass_matrix(), &j6, self.n)
    }

    /// Multi-point reading of the same law: three planes per point, `cur` in the stacked task Jacobian's row order.
    pub fn step_points(
        &mut self,
        plant: &mut dyn Plant,
        cur: &[Vec3],
        targets: &[Vec3],
        dt: f64,
    ) -> Vec<f64> {
        self.take_efference(plant);
        let m = self.m;
        let j = plant.task_jacobian();
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
        // `f` carries the ACCELERATION and `g` the weight; on a floating base these leading rows are the CONTACT'S DEMAND
        for i in 0..self.n {
            tau[i] += bias[i] + g[i];
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

    /// One control sample: step_ff with empty references.
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

    /// Feedforward form of step: F = Lambda (a_ref + wn^2 e + 2 zeta wn (v_ref - v)); empty refs = static.
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
        let coact_cmd = self.impedance.command;
        self.impedance.advance(coact_cmd, dt);
        let j = if m == 6 {
            plant.task_full_jacobian()
        } else {
            plant.task_jacobian()
        };
        // the CONFIGURATION STAMP, not a position: all this is used for is noticing that the metric's
        // configuration has moved. Reading `joint_positions` here would require every plant to have a
        // configuration — which a stance-held reduction (whose coordinates are a velocity-level
        // subspace) does not, and which is why the contract states the weaker quantity separately
        let q = plant.configuration_stamp();
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
        let (cur, cq) = plant.task_pose();
        let mut goal = target_pos;
        if self.shape_goal && !self.keepouts.is_empty() {
            let (sg, _, _) =
                TaskAvoidance.safe_target(cur, target_pos, &self.keepouts, self.ko_margin);
            goal = sg;
        }
        // the error channel is six WORLD-AXIS scalars: tip displacement then world rotvec, not B_e = -2 log(M_d ~M)
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

        // raised to the passivity floor here, never written back into d
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
        // plane-space damping mapping: corr = J M^-1 (D q_dot), added to the numerator before shaping.
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
        // `f` carries the ACCELERATION and `g` the weight; on a floating base these leading rows are the CONTACT'S DEMAND
        for i in 0..self.n {
            tau[i] += bias[i] + g[i];
        }
        // whole-arm escape as a null-space secondary task: tau_escape = (I - J' Lambda J M^-1) M a2.
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
                // task-priority: the escape acts in FULL joint space, the task fading as wt = 1 - vmax/0.02
                let wt = if vmax > 0.0 {
                    (1.0 - vmax / 0.02).max(0.0)
                } else {
                    1.0
                };
                for i in 0..self.n {
                    tau[i] = wt * tau[i] + ma2[i];
                }
            } else {
                // metric-orthogonal: the escape cannot fight the task, so it only holds the boundary
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

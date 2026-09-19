// window.rs — MotionWindow, what the theory does give: damping from the overshoot budget, a bandwidth
// floor from the settling requirement, and a sampling ceiling (the TaskClass that solves it is class.rs).
// THIS IS NOT THE MACHINE'S FEASIBILITY DECISION, and `feasible` is the field where that could be misread: unlike
// MotionBudget (budget.rs) the ceiling here is 0.2/dt, so the flag means only "the theory's window is non-empty".

/// MotionWindow is what the theory does give: damping from the overshoot budget, a bandwidth floor from the
/// settling requirement, and a sampling ceiling. Its lower bound is the SAME number as MotionBudget's in every
/// case — both call plane_design's zeta_from_overshoot / wn_for_settling, one rule at two call sites — while the
/// ceilings are different quantities: 0.2/dt is the same for every task at a given period, the budget's is the
/// task's own step against its torque ceiling. It deliberately offers no actuator ceiling: both candidates for
/// one were falsified by measurement, so what bounds the step is a property of the loop's own torque
/// distribution, which has to be measured — simu's examples/class_budget_probe.rs locates the boundary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MotionWindow {
    pub zeta: f64,
    pub wn_lo: f64,
    pub wn_hi: f64,
    pub wn: f64,
    pub feasible: bool,
    pub reason: String,
}

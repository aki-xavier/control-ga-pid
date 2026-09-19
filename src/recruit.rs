// recruit.rs — minimal-authority-first allocation of redundancy: the size principle's
// functional equivalent (SPINAL_PROGRAM.md #8). `allocate` fills channels in ASCENDING cost order,
// each up to its own capacity (the size principle itself); `weights()` is the continuous form the
// task-space solve takes. OFF is the shipped machine: unit weights, and nothing allocated.

/// Recruitment is the rule and its price table: one cost per channel, and the switch that reads it.
#[derive(Clone, Debug)]
pub struct Recruitment {
    /// whether the rule is read; false is the shipped machine
    pub on: bool,
    /// cost[i] is the authority one unit of motion on channel i spends — the model's own effort limit
    /// where it has one, and never negative or zero (a channel with no stated limit is priced 1)
    pub cost: Vec<f64>,
}

impl Recruitment {
    /// new takes a price table as given; the costs are clamped so the weights stay finite whatever a
    /// caller sends.
    pub fn new(cost: Vec<f64>) -> Recruitment {
        Recruitment {
            on: false,
            cost: cost
                .into_iter()
                .map(|c| if c.is_finite() && c > 0.0 { c } else { 1.0 })
                .collect(),
        }
    }

    /// from_limits builds the price table from a model's per-joint effort limits.
    pub fn from_limits(lim: &[f64]) -> Recruitment {
        Recruitment::new(lim.to_vec())
    }

    /// weights is the continuous form: each channel's weight is the MEAN PRICE over its own price, so
    /// a uniform price table answers exactly 1.0 — the shipped solve — and `weight * cost` is then
    /// the same number on every channel.
    pub fn weights(&self) -> Vec<f64> {
        if !self.on {
            return vec![1.0; self.cost.len()];
        }
        let n = self.cost.len();
        if n == 0 {
            return Vec::new();
        }
        let mean = self.cost.iter().sum::<f64>() / n as f64;
        self.cost
            .iter()
            .map(|c| {
                let c = self.cost_of(*c);
                mean / c
            })
            .collect()
    }

    fn cost_of(&self, c: f64) -> f64 {
        if c.is_finite() && c > 0.0 {
            c
        } else {
            1.0
        }
    }

    /// order is the recruitment order: channel indices by ASCENDING cost, ties by index so the answer
    /// is deterministic.
    pub fn order(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.cost.len()).collect();
        idx.sort_by(|&a, &b| {
            self.cost_of(self.cost[a])
                .partial_cmp(&self.cost_of(self.cost[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx
    }

    /// allocate is the size principle: fill channels in recruitment order, each up to its own
    /// capacity, until the demand is met (a demand above the total capacity saturates every channel).
    /// The loop takes the continuous form instead — this stands as the rule's own acceptance.
    pub fn allocate(&self, demand: f64, caps: &[f64]) -> Vec<f64> {
        let n = self.cost.len();
        let mut out = vec![0.0; n];
        if !(demand.is_finite() && demand > 0.0) {
            return out;
        }
        let mut left = demand;
        for i in self.order() {
            if left <= 0.0 {
                break;
            }
            let cap = caps
                .get(i)
                .copied()
                .filter(|c| c.is_finite() && *c > 0.0)
                .unwrap_or(0.0);
            let take = left.min(cap);
            out[i] = take;
            left -= take;
        }
        out
    }

    /// report is the table and the order as one line, for the caller's readout.
    pub fn report(&self) -> String {
        let order: Vec<String> = self.order().iter().map(|i| i.to_string()).collect();
        format!(
            "recruitment{}: cost [{}] order [{}]",
            if self.on { "" } else { " (off)" },
            self.cost
                .iter()
                .map(|c| format!("{c:.3}"))
                .collect::<Vec<_>>()
                .join(", "),
            order.join(", ")
        )
    }
}

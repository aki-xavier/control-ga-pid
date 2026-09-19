// recruit.rs — minimal-authority-first recruitment: fill channels in ascending cost order, each to its capacity.

/// The rule and its price table: one cost per channel, plus the switch that reads it.
#[derive(Clone, Debug)]
pub struct Recruitment {
    /// whether the rule is read; false is the shipped default
    pub on: bool,
    /// Authority one unit of motion on channel i spends; non-finite or <= 0 is priced 1.
    pub cost: Vec<f64>,
}

impl Recruitment {
    /// Takes the price table as given; non-finite or non-positive costs are replaced by 1.0.
    pub fn new(cost: Vec<f64>) -> Recruitment {
        Recruitment {
            on: false,
            cost: cost
                .into_iter()
                .map(|c| if c.is_finite() && c > 0.0 { c } else { 1.0 })
                .collect(),
        }
    }

    pub fn from_limits(lim: &[f64]) -> Recruitment {
        Recruitment::new(lim.to_vec())
    }

    /// Each weight is mean price / own price, so a uniform price table answers exactly 1.0.
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

    /// Channel indices in ASCENDING cost order; ties by index, so the answer is deterministic.
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

    /// Fills channels in recruitment order, each up to its own capacity; a demand above total capacity saturates all.
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

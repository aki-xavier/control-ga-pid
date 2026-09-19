// impedance.rs — ImpedanceChannel: one descending level applied to a (kp, kd) pair.

/// One impedance channel: the level commanded, the level in force, and the rate [1/s] between them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImpedanceChannel {
    /// the level in force this tick — what `scale` reads and what a readout reports.
    pub level: f64,
    /// the level the caller asked for, clamped at 0; the channel only ever STIFFENS.
    pub command: f64,
    /// the ramp rate [1/s] carrying the passivity bound; <= 0 applies a command directly.
    pub rate: f64,
}

impl ImpedanceChannel {
    pub fn new() -> ImpedanceChannel {
        ImpedanceChannel::default()
    }

    /// Command a level; rate <= 0 applies it directly and a negative level clamps to 0.
    pub fn set(&mut self, level: f64, rate: f64) {
        self.command = level.max(0.0);
        self.rate = rate;
        if rate <= 0.0 {
            self.level = self.command;
        }
    }

    /// Move the applied level one tick toward `target`, by at most `rate * dt` and never past it.
    /// `dt <= 0` is no tick at all; `target` is an argument so a caller may add its own term for one tick.
    pub fn advance(&mut self, target: f64, dt: f64) {
        if dt <= 0.0 || self.level == target {
            return;
        }
        if self.rate > 0.0 {
            let lim = self.rate * dt;
            let d = target - self.level;
            self.level += if d.abs() <= lim { d } else { lim * d.signum() };
        } else {
            self.level = target;
        }
    }

    /// kp by (1 + level), kd by sqrt(1 + level); level <= 0 returns the pair unchanged, bit for bit.
    pub fn scale(&self, kp: f64, kd: f64) -> (f64, f64) {
        if self.level <= 0.0 {
            return (kp, kd);
        }
        let s = 1.0 + self.level;
        (kp * s, kd * s.sqrt())
    }
}

// The descending-impedance channel pinned at the type: OFF, the scale, the ramp, `dt <= 0`.

use control_ga_pid::impedance::ImpedanceChannel;

/// A fresh or clamped channel returns the pair unchanged, bit for bit.
#[test]
fn a_channel_nobody_commanded_scales_nothing() {
    let c = ImpedanceChannel::new();
    assert_eq!(c, ImpedanceChannel::default());
    assert_eq!((c.level, c.command, c.rate), (0.0, 0.0, 0.0));
    let (kp, kd) = (400.0, 20.0);
    assert_eq!(c.scale(kp, kd), (kp, kd), "a fresh channel moved the pair");
    let mut neg = ImpedanceChannel::new();
    neg.set(-5.0, 0.0);
    assert_eq!(neg.command, 0.0, "a negative command was not clamped");
    assert_eq!(
        neg.scale(kp, kd),
        (kp, kd),
        "a clamped channel moved the pair"
    );
    assert_eq!(
        neg, c,
        "a clamped command is a different state from no command"
    );
    // any level below 0, however it got there
    let below = ImpedanceChannel {
        level: -0.5,
        ..Default::default()
    };
    assert_eq!(below.scale(kp, kd), (kp, kd));
}

/// `(1 + level)` on the spring, `sqrt(1 + level)` on the damping: the damping ratio holds.
#[test]
fn the_scale_is_one_plus_the_level_and_its_square_root() {
    let (kp, kd) = (400.0, 20.0);
    for level in [0.5, 1.0, 2.0, 7.25] {
        let c = ImpedanceChannel {
            level,
            ..Default::default()
        };
        let (k, d) = c.scale(kp, kd);
        assert!(
            (k - kp * (1.0 + level)).abs() < 1e-12,
            "the spring scale at {level} is not (1 + level): {k}"
        );
        assert!(
            (d - kd * (1.0 + level).sqrt()).abs() < 1e-12,
            "the damping scale at {level} is not sqrt(1 + level): {d}"
        );
        assert!(
            (d / k.sqrt() - kd / kp.sqrt()).abs() < 1e-12,
            "the damping ratio moved at level {level}"
        );
    }
}

/// `rate <= 0` applies the level directly and follows a moving target.
#[test]
fn a_direct_rate_applies_the_level_and_follows_a_moving_target() {
    let mut c = ImpedanceChannel::new();
    c.set(1.0, 0.0);
    assert_eq!(
        c,
        ImpedanceChannel {
            level: 1.0,
            command: 1.0,
            rate: 0.0
        }
    );
    c.advance(1.75, 1e-3);
    assert_eq!(c.level, 1.75, "a direct rate did not follow its target");
    assert_eq!(c.command, 1.0, "following the target moved the command");
    c.advance(0.25, 1e-3);
    assert_eq!(c.level, 0.25);
    let mut n = ImpedanceChannel::new();
    n.set(0.5, -3.0);
    assert_eq!((n.level, n.rate), (0.5, -3.0));
    n.advance(2.0, 1e-3);
    assert_eq!(n.level, 2.0);
}

/// The ramp is bounded by `rate * dt`, monotone, no overshoot, and arrives on the said tick.
#[test]
fn the_ramp_is_bounded_monotone_and_arrives_on_the_tick_the_arithmetic_says() {
    const RATE: f64 = 20.0;
    const DT: f64 = 1e-3;
    let mut c = ImpedanceChannel::new();
    c.set(2.0, RATE);
    assert_eq!(
        c.level, 0.0,
        "a ramped command applied itself at set() rather than over ticks"
    );
    let mut prev = 0.0f64;
    let mut arrived = 0usize;
    for k in 1..=600usize {
        c.advance(c.command, DT);
        let ratio = (1.0 + c.level) / (1.0 + prev) - 1.0;
        assert!(
            ratio <= RATE * DT + 1e-15,
            "the spring's own relative change in one tick ({ratio}) exceeded rate * dt ({})",
            RATE * DT
        );
        assert!(c.level >= prev, "the ramp went backwards at tick {k}");
        assert!(c.level <= 2.0 + 1e-15, "the ramp overshot at tick {k}");
        prev = c.level;
        if c.level == 2.0 {
            arrived = k;
            break;
        }
    }
    let want = (2.0 / (RATE * DT)).ceil() as usize;
    assert!(
        (arrived as i64 - want as i64).abs() <= 1,
        "the ramp arrived at tick {arrived}, not on the tick the arithmetic says ({want})"
    );
    c.advance(c.command, DT);
    assert_eq!(c.level, 2.0);
}

/// `dt <= 0` is no tick at all: the level does not move.
#[test]
fn no_tick_moves_nothing() {
    let mut c = ImpedanceChannel::new();
    c.set(1.0, 20.0);
    for dt in [0.0, -1e-3] {
        c.advance(1.0, dt);
        assert_eq!(c.level, 0.0, "dt = {dt} moved the level");
    }
    c.advance(0.0, 1e-3);
    assert_eq!(c.level, 0.0);
}

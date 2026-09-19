// recruitment.rs — minimal-authority-first allocation (SPINAL_PROGRAM.md #8): channels fill in
// ascending cost order, each up to its own capacity, so below the smallest channel's capacity only
// that one acts; the escape is the weighted solve `minimize |J dq - e|^2 + lam dq' W^-1 dq`, whose
// uniform price table is exactly the shipped solve — the identity the whole channel rests on.

use control_ga_pid::recruit::Recruitment;
use control_math::mat::Mat;
use control_math::task_space_bridge::TaskSpaceBridge;

/// The small unit acts first; the big one is silent until the demand outruns the small one's capacity.
#[test]
fn the_allocation_fills_the_smallest_units_first() {
    let mut r = Recruitment::new(vec![10.0, 1.0, 5.0]);
    r.on = true;
    assert_eq!(r.order(), vec![1, 2, 0]);
    let caps = [100.0, 30.0, 100.0];
    let a = r.allocate(12.0, &caps);
    assert_eq!(a, vec![0.0, 12.0, 0.0]);
    let a = r.allocate(100.0, &caps);
    assert_eq!(a, vec![0.0, 30.0, 70.0]);
    let a = r.allocate(1.0e9, &caps);
    assert_eq!(a, vec![100.0, 30.0, 100.0]);
    for d in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(r.allocate(d, &caps), vec![0.0; 3], "demand {d}");
    }
    // the order is deterministic on ties (a machine's runs have to be reproducible)
    let t = Recruitment::new(vec![1.0; 4]);
    assert_eq!(t.order(), vec![0, 1, 2, 3]);
    let mut prev = vec![0.0; 3];
    for k in 0..200 {
        let a = r.allocate(k as f64, &caps);
        for i in 0..3 {
            assert!(
                a[i] >= prev[i] - 1e-12,
                "channel {i} went backwards when the demand grew"
            );
        }
        prev = a;
    }
}

/// A joint's own effort limit is the size of its unit, and the weights are the mean over the prices
/// (so uniform prices are exactly 1); a broken price is a price of 1, not a NaN weight.
#[test]
fn the_weights_are_the_mean_price_over_each_price() {
    let off = Recruitment::from_limits(&[30.0, 60.0, 30.0]);
    assert_eq!(off.weights(), vec![1.0; 3], "off is not the shipped solve");
    let mut on = off.clone();
    on.on = true;
    let w = on.weights();
    // mean price 40 against [30, 60, 30]
    assert!((w[0] - 40.0 / 30.0).abs() < 1e-12);
    assert!((w[1] - 40.0 / 60.0).abs() < 1e-12);
    for i in 0..3 {
        assert!((w[i] * on.cost[i] - 40.0).abs() < 1e-12);
    }
    let mut u = Recruitment::new(vec![7.0; 5]);
    u.on = true;
    assert_eq!(u.weights(), vec![1.0; 5]);
    let mut b = Recruitment::new(vec![0.0, -2.0, f64::NAN, f64::INFINITY, 4.0]);
    b.on = true;
    let wb = b.weights();
    assert!(wb.iter().all(|w| w.is_finite() && *w > 0.0));
    assert_eq!(b.cost, vec![1.0, 1.0, 1.0, 1.0, 4.0]);
    for i in 0..5 {
        assert!(
            (wb[i] * b.cost[i] - 1.6).abs() < 1e-12,
            "the price-weight product is not the mean price on channel {i}"
        );
    }
    assert!(b.report().contains("recruitment"));
}

/// Uniform weights are the shipped answer bit for bit, and a price spread moves the task onto the
/// cheap joint while the task itself stays satisfied.
#[test]
fn the_weighted_solve_spends_the_cheap_joint_first() {
    // a redundant 2 x 3 task: channels 0 and 1 both move the first row, channel 2 owns the second
    let j = Mat::from_rows(&[vec![1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);
    let e = [1.0, 0.0];
    let b = TaskSpaceBridge::new(3, 1e-9);
    assert_eq!(b.step_weighted(&j, &e, &[1.0, 1.0, 1.0]), b.step(&j, &e));
    let flat = b.step(&j, &e);
    assert!(
        (flat[0] - 0.5).abs() < 1e-6 && (flat[1] - 0.5).abs() < 1e-6,
        "the unweighted split is not even: {flat:?}"
    );
    let mut r = Recruitment::new(vec![1.0, 1e6, 1.0]);
    r.on = true;
    let dq = b.step_weighted(&j, &e, &r.weights());
    assert!(
        dq[1].abs() < 1e-5,
        "the expensive joint moved {} of the task",
        dq[1]
    );
    assert!(
        (dq[0] - 1.0).abs() < 1e-5,
        "the cheap joint did not take the task: {}",
        dq[0]
    );
    assert!(dq[2].abs() < 1e-12);
    let row = j.mul_vec(&dq);
    assert!(
        (row[0] - 1.0).abs() < 1e-6 && row[1].abs() < 1e-12,
        "the task was not met: {row:?}"
    );
    let dq = b.step_weighted(&j, &e, &[f64::NAN, 0.0, -1.0]);
    assert_eq!(dq, b.step(&j, &e));
}

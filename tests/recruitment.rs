// Minimal-authority-first allocation; uniform prices make `weights()` exactly the shipped solve.

use control_ga_pid::recruit::Recruitment;

/// Small units fill first, up to their capacity; ties break deterministically.
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

/// Weights are the mean price over each price; a broken price counts as 1.
#[test]
fn the_weights_are_the_mean_price_over_each_price() {
    let off = Recruitment::from_limits(&[30.0, 60.0, 30.0]);
    assert_eq!(off.weights(), vec![1.0; 3], "off is not the shipped solve");
    let mut on = off.clone();
    on.on = true;
    let w = on.weights();
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

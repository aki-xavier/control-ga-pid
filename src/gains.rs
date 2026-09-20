// gains.rs — PlaneGains, one plane's gain triple; the placement that makes it is design.rs.

/// The law reads this triple as given: a non-positive `kappa_p` arrives unclamped, so a plane with no
/// authority over its own response shows it in the response rather than hidden as a small gain —
/// `design.rs` reports it, once per process.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaneGains {
    pub kappa_p: f64,
    pub kappa_d: f64,
    pub kappa_i: f64,
}

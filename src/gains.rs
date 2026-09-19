// gains.rs — PlaneGains, one plane's gain triple; the placement that makes it is design.rs.

/// PlaneGains is one plane's (kappa_p, kappa_d, kappa_i) triple.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaneGains {
    pub kappa_p: f64,
    pub kappa_d: f64,
    pub kappa_i: f64,
}

//! Bounds on physical state that keep NaN
//!
//! `f64::min` and `f64::max` return the other operand when one is NaN,
//! which would turn a failed calculation into a plausible value such as the temperature cap.

use rscm_core::timeseries::FloatValue;

pub(crate) trait Bounded {
    /// `self.min(bound)`, except that NaN stays NaN.
    fn at_most(self, bound: FloatValue) -> FloatValue;
    /// `self.max(bound)`, except that NaN stays NaN.
    fn at_least(self, bound: FloatValue) -> FloatValue;
}

impl Bounded for FloatValue {
    fn at_most(self, bound: FloatValue) -> FloatValue {
        if self.is_nan() {
            self
        } else {
            self.min(bound)
        }
    }

    fn at_least(self, bound: FloatValue) -> FloatValue {
        if self.is_nan() {
            self
        } else {
            self.max(bound)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_is_kept() {
        assert!(FloatValue::NAN.at_most(25.0).is_nan());
        assert!(FloatValue::NAN.at_least(0.0).is_nan());
    }

    #[test]
    fn finite_values_match_min_and_max() {
        assert_eq!(30.0_f64.at_most(25.0), 25.0);
        assert_eq!(20.0_f64.at_most(25.0), 20.0);
        assert_eq!((-1.0_f64).at_least(0.0), 0.0);
        assert_eq!(2.0_f64.at_least(0.0), 2.0);
    }
}

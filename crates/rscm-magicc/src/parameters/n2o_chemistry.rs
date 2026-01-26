//! N2O Chemistry Parameters
//!
//! Parameters for atmospheric nitrous oxide chemistry calculations.
//!
//! # Reference
//!
//! Based on MAGICC7 Module 02 (N2O Chemistry) which solves the N2O mass
//! balance with concentration-dependent lifetime feedback and stratospheric
//! transport delay.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for N2O chemistry calculations
///
/// Controls the nitrous oxide atmospheric lifetime calculation including:
/// - Base stratospheric photolysis/chemical destruction lifetime
/// - Concentration-dependent lifetime feedback
/// - Stratospheric transport delay for the sink term
///
/// # Lifetime Calculation
///
/// The effective N2O lifetime is calculated as:
///
/// $$\tau = \tau_{init} \cdot \max\left(1, \frac{\bar{B}}{B_0}\right)^S$$
///
/// where:
/// - $\tau_{init}$ is the base lifetime
/// - $\bar{B}$ is the mid-year burden estimate
/// - $B_0$ is the reference burden at pre-industrial
/// - $S$ is the feedback exponent (negative, so lifetime decreases slightly with higher burden)
///
/// # Stratospheric Delay
///
/// Unlike CH4 which uses instantaneous burden for the sink, N2O uses a lagged
/// average burden to account for the time required for tropospheric N2O to
/// mix into the stratosphere where destruction occurs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct N2OChemistryParameters {
    /// Pre-industrial N2O concentration used as reference for feedbacks
    /// unit: ppb
    /// default: 270.0
    pub n2o_pi: FloatValue,

    /// Natural N2O emissions (soils, oceans, etc.)
    /// unit: Tg N/yr
    /// default: 11.0
    pub natural_emissions: FloatValue,

    /// Base atmospheric lifetime (τ_init)
    /// This is the reference lifetime before concentration feedback
    /// unit: years
    /// default: 139.275 (MAGICC7 default)
    pub tau_n2o: FloatValue,

    /// Lifetime feedback exponent (S)
    /// With S < 0, higher concentrations give slightly shorter lifetime:
    /// tau = tau_init * max(1, ratio)^S, and ratio^(-0.04) < 1 when ratio > 1.
    /// This is a very weak feedback compared to CH4 (S = -0.04 vs CH4's ~-0.23).
    /// unit: dimensionless
    /// default: -0.04
    pub lifetime_feedback: FloatValue,

    /// Stratospheric mixing delay
    /// Time lag for tropospheric N2O to reach the stratosphere for destruction
    /// unit: years
    /// default: 1
    pub strat_delay: usize,

    /// Conversion factor from ppb to Tg N
    /// Based on molecular weight (28 g/mol for N in N2O) and atmospheric moles
    /// unit: Tg N/ppb
    /// default: 4.79
    pub ppb_to_tg: FloatValue,
}

impl Default for N2OChemistryParameters {
    fn default() -> Self {
        Self {
            n2o_pi: 270.0,
            natural_emissions: 11.0,
            tau_n2o: 139.275,
            lifetime_feedback: -0.04,
            strat_delay: 1,
            ppb_to_tg: 4.79,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = N2OChemistryParameters::default();
        assert!((params.n2o_pi - 270.0).abs() < 1e-10);
        assert!((params.tau_n2o - 139.275).abs() < 1e-10);
        assert!((params.ppb_to_tg - 4.79).abs() < 1e-10);
    }

    #[test]
    fn test_lifetime_feedback_sign() {
        let params = N2OChemistryParameters::default();
        // Feedback should be negative (lifetime increases with concentration)
        assert!(
            params.lifetime_feedback < 0.0,
            "N2O lifetime feedback should be negative"
        );
    }

    #[test]
    fn test_stratospheric_delay() {
        let params = N2OChemistryParameters::default();
        // Delay should be at least 1 year
        assert!(
            params.strat_delay >= 1,
            "Stratospheric delay should be >= 1"
        );
    }

    #[test]
    fn test_natural_emissions_reasonable() {
        let params = N2OChemistryParameters::default();
        // Natural emissions should be in reasonable range (9-13 Tg N/yr)
        assert!(
            params.natural_emissions > 5.0 && params.natural_emissions < 20.0,
            "Natural emissions should be in reasonable range"
        );
    }
}

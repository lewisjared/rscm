//! N2O Chemistry Parameters
//!
//! Parameters for atmospheric nitrous oxide chemistry calculations.
//!
//! # Reference
//!
//! Based on MAGICC7's N2O chemistry, which solves the N2O mass
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
/// - $B_0$ is the reference burden in the year before `feedback_start_year`
/// - $S$ is the feedback exponent (negative, so lifetime decreases slightly with higher burden)
///
/// # Stratospheric Delay
///
/// Unlike CH4, whose sink uses the current burden, the N2O sink uses a delayed
/// average burden to account for the time tropospheric N2O takes to
/// mix into the stratosphere where destruction occurs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct N2OChemistryParameters {
    /// Year from which the concentration feedback is measured (MAGICC7 `N2O_FEED_YRSTART`).
    ///
    /// unit: yr
    ///
    /// The reference concentration is taken from the year before.
    pub feedback_start_year: FloatValue,

    /// Natural N2O emissions from soils, oceans and other sources.
    ///
    /// unit: Mt N2ON/yr
    ///
    /// MAGICC7 closes the 1982-1991 budget against observed concentrations.
    /// The default is the value it derives for its standard historical inputs.
    pub natural_emissions: FloatValue,

    /// Base atmospheric lifetime before the concentration feedback, $\tau_{init}$
    ///
    /// unit: yr
    pub initial_n2o_lifetime: FloatValue,

    /// Lifetime feedback exponent, $S$.
    ///
    /// unit: dimensionless
    ///
    /// With $S < 0$, higher concentrations give a slightly shorter lifetime,
    /// because $\tau = \tau_{init} \max(1, B/B_0)^S$.
    pub lifetime_feedback: FloatValue,

    /// Delay for tropospheric N2O to reach the stratosphere and be destroyed (MAGICC7 `N2O_STRATMIXDELAY`)
    ///
    /// unit: yr
    pub stratospheric_delay: usize,

    /// Conversion factor from ppb to Tg N (MAGICC7 `N2O_PPB2TGN`)
    ///
    /// unit: Tg N/ppb
    pub tg_per_ppb: FloatValue,

    /// Apply the stratospheric lifetime response to meridional flux changes (MAGICC7 `N2O_USE_TAUSTRAT_VAR`)
    pub include_meridional_flux_feedback: bool,

    /// Fractional change in meridional flux per degree of warming (MAGICC7 `GEN_MERIDFLUX_CHNGPERDEG`)
    ///
    /// unit: K^-1
    pub meridional_flux_change_per_degree: FloatValue,

    /// Lifetime sensitivity to meridional flux changes (MAGICC7 `N2O_TAUSTRAT_SENS2MERIDFLUX`)
    ///
    /// unit: dimensionless
    pub lifetime_sensitivity_to_meridional_flux: FloatValue,

    /// Year from which warming changes the meridional flux (MAGICC7 `GEN_CHANGE_MERIDIONALFLUX_YR`)
    ///
    /// unit: yr
    pub meridional_flux_reference_year: FloatValue,

    /// Last year of concentrations prescribed by `Prescribed Atmospheric Concentrations|N2O`
    /// (MAGICC7 `N2O_SWITCHFROMCONC2EMIS_YEAR`).
    ///
    /// unit: yr
    ///
    /// `None` calculates concentrations from emissions throughout.
    pub concentration_switch_year: Option<FloatValue>,
}

impl Default for N2OChemistryParameters {
    fn default() -> Self {
        Self {
            feedback_start_year: 1925.0,
            natural_emissions: 8.0345017,
            initial_n2o_lifetime: 139.275,
            lifetime_feedback: -0.04,
            stratospheric_delay: 1,
            tg_per_ppb: 4.79,
            include_meridional_flux_feedback: true,
            meridional_flux_change_per_degree: 0.15,
            lifetime_sensitivity_to_meridional_flux: 0.04,
            meridional_flux_reference_year: 1980.0,
            concentration_switch_year: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = N2OChemistryParameters::default();
        assert!((params.feedback_start_year - 1925.0).abs() < 1e-10);
        assert!((params.initial_n2o_lifetime - 139.275).abs() < 1e-10);
        assert!((params.tg_per_ppb - 4.79).abs() < 1e-10);
    }

    #[test]
    fn test_lifetime_feedback_sign() {
        let params = N2OChemistryParameters::default();
        // Feedback should be negative (lifetime decreases with concentration)
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
            params.stratospheric_delay >= 1,
            "Stratospheric delay should be >= 1"
        );
    }

    #[test]
    fn test_natural_emissions_reasonable() {
        let params = N2OChemistryParameters::default();
        assert!(
            params.natural_emissions > 5.0 && params.natural_emissions < 20.0,
            "Natural emissions should be in reasonable range"
        );
    }
}

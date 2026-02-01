//! CH4 Chemistry Parameters
//!
//! Parameters for atmospheric methane chemistry calculations, implementing
//! the Prather iteration method with OH feedback.
//!
//! # Reference
//!
//! Based on MAGICC7 Module 01 (CH4 Chemistry) which implements Prather's
//! method from IPCC TAR Table 4.11 for solving the nonlinear methane mass
//! balance equation.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for CH4 chemistry calculations
///
/// Controls the methane atmospheric lifetime calculation including:
/// - OH sink with concentration-dependent feedback (Prather method)
/// - Non-OH sinks (soil, stratosphere, tropospheric Cl)
/// - Temperature feedback on OH oxidation rates
/// - Emissions feedback from co-emitted species (NOx, CO, NMVOC)
///
/// # Lifetime Calculation
///
/// The effective CH4 lifetime is calculated as:
///
/// $$\frac{1}{\tau_{total}} = \frac{1}{\tau_{OH}} + \frac{1}{\tau_{soil}} + \frac{1}{\tau_{strat}} + \frac{1}{\tau_{trop\_cl}}$$
///
/// where ${\tau_{OH}}$ varies with CH4 concentration and co-emitted species.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CH4ChemistryParameters {
    /// Pre-industrial CH4 concentration used as reference for feedbacks
    /// unit: ppb
    /// default: 722.0
    pub ch4_pi: FloatValue,

    /// Natural CH4 emissions (wetlands, termites, oceans, etc.)
    /// unit: Tg CH4/yr
    /// default: 209.0
    pub natural_emissions: FloatValue,

    /// Base tropospheric OH sink lifetime
    /// This is the initial/reference OH lifetime before feedbacks
    /// unit: years
    /// default: 9.3
    pub tau_oh: FloatValue,

    /// Soil uptake sink lifetime
    /// unit: years
    /// default: 150.0
    pub tau_soil: FloatValue,

    /// Stratospheric sink lifetime
    /// unit: years
    /// default: 120.0
    pub tau_strat: FloatValue,

    /// Tropospheric Cl sink lifetime
    /// unit: years
    /// default: 200.0
    pub tau_trop_cl: FloatValue,

    /// CH4 self-feedback coefficient (S in TAR Table 4.11)
    /// Represents how CH4 concentration affects OH and hence its own lifetime
    /// Negative value: higher CH4 → lower OH → longer lifetime (positive feedback)
    /// unit: dimensionless
    /// default: -0.32
    pub ch4_self_feedback: FloatValue,

    /// OH sensitivity scaling factor (γ in the equations)
    /// Scales the OH-lifetime feedback response
    /// unit: dimensionless
    /// default: 0.72
    pub oh_sensitivity_scale: FloatValue,

    /// NOx emissions effect on OH (A_NOx)
    /// Positive: more NOx → more OH → shorter CH4 lifetime
    /// unit: (Tg N/yr)^-1
    /// default: 0.0042
    pub oh_nox_sensitivity: FloatValue,

    /// CO emissions effect on OH (A_CO)
    /// Negative: more CO → less OH → longer CH4 lifetime
    /// unit: (Tg CO/yr)^-1
    /// default: -0.000105
    pub oh_co_sensitivity: FloatValue,

    /// NMVOC emissions effect on OH (A_VOC)
    /// Negative: more VOC → less OH → longer CH4 lifetime
    /// unit: (Tg NMVOC/yr)^-1
    /// default: -0.000315
    pub oh_nmvoc_sensitivity: FloatValue,

    /// Temperature sensitivity of lifetime
    /// Positive: warmer → shorter lifetime (more OH activity)
    /// unit: K^-1
    /// default: 0.0316
    pub temp_sensitivity: FloatValue,

    /// Enable temperature feedback on lifetime
    /// default: true
    pub include_temp_feedback: bool,

    /// Enable NOx/CO/NMVOC emissions feedback on OH
    /// default: true
    pub include_emissions_feedback: bool,

    /// Conversion factor from ppb to Tg CH4
    /// Derived from molecular weight (16 g/mol) and atmospheric moles
    /// unit: Tg CH4/ppb
    /// default: 2.75
    pub ppb_to_tg: FloatValue,

    /// Reference year NOx emissions for delta calculation
    /// unit: Tg N/yr
    /// default: 0.0
    pub nox_reference: FloatValue,

    /// Reference year CO emissions for delta calculation
    /// unit: Tg CO/yr
    /// default: 0.0
    pub co_reference: FloatValue,

    /// Reference year NMVOC emissions for delta calculation
    /// unit: Tg NMVOC/yr
    /// default: 0.0
    pub nmvoc_reference: FloatValue,
}

impl Default for CH4ChemistryParameters {
    fn default() -> Self {
        Self {
            ch4_pi: 722.0,
            natural_emissions: 209.0,
            tau_oh: 9.3,
            tau_soil: 150.0,
            tau_strat: 120.0,
            tau_trop_cl: 200.0,
            ch4_self_feedback: -0.32,
            oh_sensitivity_scale: 0.72,
            oh_nox_sensitivity: 0.0042,
            oh_co_sensitivity: -0.000105,
            oh_nmvoc_sensitivity: -0.000315,
            temp_sensitivity: 0.0316,
            include_temp_feedback: true,
            include_emissions_feedback: true,
            ppb_to_tg: 2.75,
            nox_reference: 0.0,
            co_reference: 0.0,
            nmvoc_reference: 0.0,
        }
    }
}

impl CH4ChemistryParameters {
    /// Calculate the combined non-OH lifetime
    ///
    /// $$\frac{1}{\tau_{other}} = \frac{1}{\tau_{soil}} + \frac{1}{\tau_{strat}} + \frac{1}{\tau_{trop\_cl}}$$
    pub fn tau_other(&self) -> FloatValue {
        1.0 / (1.0 / self.tau_soil + 1.0 / self.tau_strat + 1.0 / self.tau_trop_cl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = CH4ChemistryParameters::default();
        assert!((params.ch4_pi - 722.0).abs() < 1e-10);
        assert!((params.tau_oh - 9.3).abs() < 1e-10);
        assert!(params.include_temp_feedback);
    }

    #[test]
    fn test_tau_other_calculation() {
        let params = CH4ChemistryParameters::default();
        let tau_other = params.tau_other();

        // Expected: 1/(1/150 + 1/120 + 1/200) ≈ 48.65
        let expected = 1.0 / (1.0 / 150.0 + 1.0 / 120.0 + 1.0 / 200.0);
        assert!(
            (tau_other - expected).abs() < 1e-10,
            "tau_other should be {:.2}, got {:.2}",
            expected,
            tau_other
        );
    }

    #[test]
    fn test_ch4_self_feedback_sign() {
        let params = CH4ChemistryParameters::default();
        // Self-feedback should be negative (positive feedback loop)
        assert!(
            params.ch4_self_feedback < 0.0,
            "CH4 self-feedback should be negative"
        );
    }

    #[test]
    fn test_emissions_feedback_signs() {
        let params = CH4ChemistryParameters::default();
        // NOx increases OH (positive coefficient)
        assert!(params.oh_nox_sensitivity > 0.0);
        // CO and NMVOC decrease OH (negative coefficients)
        assert!(params.oh_co_sensitivity < 0.0);
        assert!(params.oh_nmvoc_sensitivity < 0.0);
    }

    #[test]
    fn test_partial_deserialization() {
        // Test that #[serde(default)] allows partial deserialization
        let json = r#"{"tau_oh": 10.5, "include_temp_feedback": false}"#;
        let params: CH4ChemistryParameters =
            serde_json::from_str(json).expect("Partial deserialization failed");

        // Specified fields from JSON
        assert!((params.tau_oh - 10.5).abs() < 1e-10);
        assert!(!params.include_temp_feedback);

        // Other fields should be defaults
        assert!((params.ch4_pi - 722.0).abs() < 1e-10);
        assert!((params.tau_soil - 150.0).abs() < 1e-10);
    }
}

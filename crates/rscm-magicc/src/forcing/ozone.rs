//! Ozone Radiative Forcing Component
//!
//! Calculates radiative forcing from stratospheric and tropospheric ozone
//! changes following MAGICC7 Module 04.
//!
//! # What This Component Does
//!
//! 1. Calculates stratospheric ozone forcing from EESC (Equivalent Effective
//!    Stratospheric Chlorine) using a power-law relationship. This is negative
//!    (cooling) forcing because ozone depletion reduces shortwave absorption.
//!
//! 2. Calculates tropospheric ozone forcing from:
//!    - CH4 concentration (logarithmic relationship)
//!    - Ozone precursor emissions: NOx, CO, NMVOC (linear relationships)
//!
//! 3. Calculates temperature feedback on ozone (optional, negative feedback).
//!
//! # Inputs
//!
//! - `EESC` (ppt) - Equivalent Effective Stratospheric Chlorine from halocarbon chemistry
//! - `Atmospheric Concentration|CH4` (ppb) - Methane concentration
//! - `Emissions|NOx` (Mt N/yr) - NOx emissions
//! - `Emissions|CO` (Mt CO/yr) - CO emissions
//! - `Emissions|NMVOC` (Mt NMVOC/yr) - NMVOC emissions
//! - `Surface Temperature` (K) - Global mean temperature anomaly
//!
//! # Outputs
//!
//! - `Effective Radiative Forcing|O3|Stratospheric` (W/m²) - Stratospheric ozone forcing
//! - `Effective Radiative Forcing|O3|Tropospheric` (W/m²) - Tropospheric ozone forcing
//! - `Effective Radiative Forcing|O3|Temperature Feedback` (W/m²) - Temperature feedback
//!
//! Note: Total ozone forcing should be computed downstream by summing components.
//!
//! # Differences from MAGICC7 Module 04
//!
//! This is a simplified implementation:
//!
//! - **Regional forcing**: MAGICC7 distributes forcing across 4 boxes with different
//!   patterns for stratospheric vs tropospheric. Here, forcing is scalar (global mean).
//! - **N2O ozone effect**: MAGICC7 includes a separate N2O-induced ozone forcing term.
//!   Not implemented here (considered part of N2O's direct forcing).
//! - **Threshold year**: MAGICC7 sets stratospheric forcing to zero before a threshold
//!   year (default 1979). Here, forcing is calculated for any EESC above reference.
//! - **Aviation NOx**: MAGICC7 has special treatment for aviation sector NOx with
//!   different effectiveness. Here, all NOx is treated equally.
//! - **First-year offset**: MAGICC7 subtracts first-year forcing. Not implemented.
//! - **Constant-after-year**: MAGICC7 can hold forcing constant after specified year.
//!   Not implemented.

use crate::parameters::OzoneForcingParameters;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Ozone radiative forcing component
///
/// Calculates stratospheric and tropospheric ozone effective radiative forcing
/// from EESC (halocarbons), CH4 concentration, and ozone precursor emissions.
///
/// # Stratospheric Ozone
///
/// Uses a power-law relationship with EESC:
///
/// $$ERF_{strat} = \alpha \cdot \max(0, EESC - EESC_{ref})^{\beta} / 100^{\beta}$$
///
/// The division by $100^{\beta}$ normalises the EESC delta to be in units of
/// 100 ppt, matching the MAGICC7 parameterisation.
///
/// Note: $\alpha$ is negative because ozone depletion causes cooling.
///
/// # Tropospheric Ozone
///
/// Combines CH4 and precursor emission effects:
///
/// $$ERF_{trop} = \eta \cdot \left[ OZCH4 \cdot \ln\left(\frac{CH4}{CH4_{pi}}\right)
///     + \alpha_{NOx} \cdot \Delta NOx + \alpha_{CO} \cdot \Delta CO
///     + \alpha_{VOC} \cdot \Delta NMVOC \right]$$
///
/// # Temperature Feedback
///
/// $$ERF_{temp} = \gamma \cdot \Delta T$$
///
/// This is typically negative (warming destroys ozone, reducing forcing).
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["forcing", "ozone", "magicc"], category = "Radiative Forcing")]
#[inputs(
    eesc { name = "EESC", unit = "ppt" },
    ch4_concentration { name = "Atmospheric Concentration|CH4", unit = "ppb" },
    nox_emissions { name = "Emissions|NOx", unit = "Mt N/yr" },
    co_emissions { name = "Emissions|CO", unit = "Mt CO/yr" },
    nmvoc_emissions { name = "Emissions|NMVOC", unit = "Mt NMVOC/yr" },
    temperature { name = "Surface Temperature|Global", unit = "K" },
)]
#[outputs(
    strat_o3_erf { name = "Effective Radiative Forcing|O3|Stratospheric", unit = "W/m^2" },
    trop_o3_erf { name = "Effective Radiative Forcing|O3|Tropospheric", unit = "W/m^2" },
    temp_feedback_erf { name = "Effective Radiative Forcing|O3|Temperature Feedback", unit = "W/m^2" },
)]
pub struct OzoneForcing {
    parameters: OzoneForcingParameters,
}

impl OzoneForcing {
    /// Create a new ozone forcing component with default parameters
    pub fn new() -> Self {
        Self::from_parameters(OzoneForcingParameters::default())
    }

    /// Create a new ozone forcing component from parameters
    pub fn from_parameters(parameters: OzoneForcingParameters) -> Self {
        Self { parameters }
    }

    /// Get the parameters
    pub fn parameters(&self) -> &OzoneForcingParameters {
        &self.parameters
    }

    /// Calculate stratospheric ozone forcing from EESC
    ///
    /// Uses power-law relationship:
    /// RF = scale * max(0, (EESC - EESC_ref) / 100)^exponent
    ///
    /// The scale factor is negative (cooling from ozone depletion).
    pub fn calculate_strat_forcing(&self, eesc: FloatValue) -> FloatValue {
        let delta_eesc = eesc - self.parameters.eesc_reference;

        if delta_eesc <= 0.0 {
            // Below reference EESC: no stratospheric ozone depletion
            0.0
        } else {
            // Power-law relationship, normalised by 100 ppt
            // Note: scale is negative, so forcing is negative (cooling)
            self.parameters.strat_o3_scale
                * (delta_eesc / 100.0).powf(self.parameters.strat_cl_exponent)
        }
    }

    /// Calculate tropospheric ozone forcing
    ///
    /// Combines:
    /// - CH4 contribution (logarithmic in concentration)
    /// - Precursor emissions (linear: NOx, CO, NMVOC)
    pub fn calculate_trop_forcing(
        &self,
        ch4: FloatValue,
        nox: FloatValue,
        co: FloatValue,
        nmvoc: FloatValue,
    ) -> FloatValue {
        let p = &self.parameters;

        // CH4 contribution (logarithmic)
        // Ozone change in DU from CH4
        let ch4_term = if ch4 > 0.0 && p.ch4_pi > 0.0 {
            p.trop_oz_ch4 * (ch4 / p.ch4_pi).ln()
        } else {
            0.0
        };

        // Precursor emissions contribution (linear, relative to pre-industrial)
        let delta_nox = nox - p.nox_pi;
        let delta_co = co - p.co_pi;
        let delta_nmvoc = nmvoc - p.nmvoc_pi;

        let precursor_term =
            p.trop_oz_nox * delta_nox + p.trop_oz_co * delta_co + p.trop_oz_voc * delta_nmvoc;

        // Total ozone change in DU, converted to forcing via radiative efficiency
        p.trop_radeff * (ch4_term + precursor_term)
    }

    /// Calculate temperature feedback on ozone forcing
    ///
    /// Returns negative forcing for positive temperature (negative feedback).
    pub fn calculate_temp_feedback(&self, temperature: FloatValue) -> FloatValue {
        self.parameters.temp_feedback_scale * temperature
    }

    /// Calculate all ozone forcing components
    pub fn calculate_forcings(
        &self,
        eesc: FloatValue,
        ch4: FloatValue,
        nox: FloatValue,
        co: FloatValue,
        nmvoc: FloatValue,
        temperature: FloatValue,
    ) -> OzoneForcingResult {
        OzoneForcingResult {
            strat_o3_erf: self.calculate_strat_forcing(eesc),
            trop_o3_erf: self.calculate_trop_forcing(ch4, nox, co, nmvoc),
            temp_feedback_erf: self.calculate_temp_feedback(temperature),
        }
    }
}

impl Default for OzoneForcing {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of ozone forcing calculations
#[derive(Debug, Clone, Copy)]
pub struct OzoneForcingResult {
    /// Stratospheric ozone forcing (negative, from EESC-driven depletion)
    pub strat_o3_erf: FloatValue,
    /// Tropospheric ozone forcing (positive, from CH4 and precursors)
    pub trop_o3_erf: FloatValue,
    /// Temperature feedback forcing (typically negative)
    pub temp_feedback_erf: FloatValue,
}

impl OzoneForcingResult {
    /// Total ozone forcing from all components
    pub fn total(&self) -> FloatValue {
        self.strat_o3_erf + self.trop_o3_erf + self.temp_feedback_erf
    }
}

#[typetag::serde]
impl Component for OzoneForcing {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = OzoneForcingInputs::from_input_state(input_state);

        let eesc = inputs.eesc.at_start();
        let ch4 = inputs.ch4_concentration.at_start();
        let nox = inputs.nox_emissions.at_start();
        let co = inputs.co_emissions.at_start();
        let nmvoc = inputs.nmvoc_emissions.at_start();
        let temperature = inputs.temperature.at_start();

        let result = self.calculate_forcings(eesc, ch4, nox, co, nmvoc, temperature);

        let outputs = OzoneForcingOutputs {
            strat_o3_erf: result.strat_o3_erf,
            trop_o3_erf: result.trop_o3_erf,
            temp_feedback_erf: result.temp_feedback_erf,
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rscm_core::component::RequirementType;

    fn default_component() -> OzoneForcing {
        OzoneForcing::from_parameters(OzoneForcingParameters::default())
    }

    // ===== Stratospheric Ozone Tests =====

    #[test]
    fn test_strat_forcing_zero_at_reference() {
        let component = default_component();
        let eesc_ref = component.parameters().eesc_reference;

        let forcing = component.calculate_strat_forcing(eesc_ref);
        assert!(
            forcing.abs() < 1e-10,
            "Stratospheric forcing at reference EESC should be zero, got {}",
            forcing
        );
    }

    #[test]
    fn test_strat_forcing_zero_below_reference() {
        let component = default_component();
        let eesc_ref = component.parameters().eesc_reference;

        // Below reference EESC (pre-industrial conditions)
        let forcing = component.calculate_strat_forcing(eesc_ref - 500.0);
        assert!(
            forcing.abs() < 1e-10,
            "Stratospheric forcing below reference EESC should be zero, got {}",
            forcing
        );
    }

    #[test]
    fn test_strat_forcing_negative_above_reference() {
        let component = default_component();
        let eesc_ref = component.parameters().eesc_reference;

        // Above reference EESC (ozone depletion conditions)
        let forcing = component.calculate_strat_forcing(eesc_ref + 500.0);
        assert!(
            forcing < 0.0,
            "Stratospheric forcing above reference EESC should be negative (cooling), got {}",
            forcing
        );
    }

    #[test]
    fn test_strat_forcing_scales_with_eesc() {
        let component = default_component();
        let eesc_ref = component.parameters().eesc_reference;

        let forcing_low = component.calculate_strat_forcing(eesc_ref + 200.0);
        let forcing_high = component.calculate_strat_forcing(eesc_ref + 400.0);

        // Higher EESC should give stronger (more negative) forcing
        assert!(
            forcing_high < forcing_low,
            "Higher EESC should give more negative forcing: {} vs {}",
            forcing_high,
            forcing_low
        );

        // Check power-law scaling: ratio should be (400/200)^1.7 = 2^1.7 ≈ 3.25
        let expected_ratio = 2.0_f64.powf(1.7);
        let actual_ratio = forcing_high / forcing_low;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "Power-law scaling: expected ratio {:.3}, got {:.3}",
            expected_ratio,
            actual_ratio
        );
    }

    #[test]
    fn test_strat_forcing_realistic_magnitude() {
        let component = default_component();

        // Peak EESC circa 2000: ~2000 ppt
        // Reference (1979): ~1420 ppt
        // Delta: ~580 ppt
        let peak_eesc = 2000.0;
        let forcing = component.calculate_strat_forcing(peak_eesc);

        // AR6 estimate: -0.02 [-0.10 to 0.03] W/m²
        // Our calculation should be in this range
        assert!(
            forcing > -0.15 && forcing < 0.0,
            "Peak stratospheric forcing should be in realistic range, got {} W/m²",
            forcing
        );
    }

    // ===== Tropospheric Ozone Tests =====

    #[test]
    fn test_trop_forcing_zero_at_preindustrial() {
        let component = default_component();
        let p = component.parameters();

        // At pre-industrial conditions
        let forcing = component.calculate_trop_forcing(p.ch4_pi, p.nox_pi, p.co_pi, p.nmvoc_pi);

        assert!(
            forcing.abs() < 1e-10,
            "Tropospheric forcing at pre-industrial should be zero, got {}",
            forcing
        );
    }

    #[test]
    fn test_trop_forcing_positive_above_preindustrial() {
        let component = default_component();

        // Elevated CH4 and precursor emissions
        let forcing = component.calculate_trop_forcing(
            1800.0, // Modern CH4 ~1900 ppb
            40.0,   // Modern NOx ~40 Mt N/yr
            500.0,  // Modern CO ~500 Mt/yr
            100.0,  // Modern NMVOC ~100 Mt/yr
        );

        assert!(
            forcing > 0.0,
            "Tropospheric forcing above pre-industrial should be positive, got {}",
            forcing
        );
    }

    #[test]
    fn test_trop_forcing_ch4_logarithmic() {
        let component = default_component();
        let p = component.parameters();

        // Test logarithmic scaling with CH4
        let forcing_2x = component.calculate_trop_forcing(p.ch4_pi * 2.0, 0.0, 0.0, 0.0);
        let forcing_4x = component.calculate_trop_forcing(p.ch4_pi * 4.0, 0.0, 0.0, 0.0);

        // Doubling again should add same increment (log scale)
        // ln(4x) - ln(2x) = ln(2), same as ln(2x) - ln(1x)
        let increment_2_to_4 = forcing_4x - forcing_2x;
        let increment_1_to_2 = forcing_2x;

        assert!(
            (increment_2_to_4 - increment_1_to_2).abs() < 1e-10,
            "CH4 forcing should be logarithmic: increments {} and {} should match",
            increment_1_to_2,
            increment_2_to_4
        );
    }

    #[test]
    fn test_trop_forcing_nox_linear() {
        let component = default_component();

        let forcing_10 = component.calculate_trop_forcing(700.0, 10.0, 0.0, 0.0);
        let forcing_20 = component.calculate_trop_forcing(700.0, 20.0, 0.0, 0.0);

        // Linear: doubling NOx should double the NOx contribution
        assert!(
            (forcing_20 - 2.0 * forcing_10).abs() < 1e-10,
            "NOx forcing should be linear: F(20)={}, 2*F(10)={}",
            forcing_20,
            2.0 * forcing_10
        );
    }

    #[test]
    fn test_trop_forcing_realistic_magnitude() {
        let component = default_component();

        // Modern conditions (circa 2020)
        let forcing = component.calculate_trop_forcing(
            1900.0, // CH4 ppb
            42.0,   // NOx Mt N/yr
            550.0,  // CO Mt/yr
            120.0,  // NMVOC Mt/yr
        );

        // AR6 estimate: 0.47 [0.24 to 0.70] W/m²
        assert!(
            forcing > 0.2 && forcing < 0.8,
            "Modern tropospheric forcing should be in realistic range, got {} W/m²",
            forcing
        );
    }

    // ===== Temperature Feedback Tests =====

    #[test]
    fn test_temp_feedback_zero_at_baseline() {
        let component = default_component();
        let forcing = component.calculate_temp_feedback(0.0);

        assert!(
            forcing.abs() < 1e-10,
            "Temperature feedback at zero anomaly should be zero, got {}",
            forcing
        );
    }

    #[test]
    fn test_temp_feedback_negative_for_warming() {
        let component = default_component();
        let forcing = component.calculate_temp_feedback(2.0);

        assert!(
            forcing < 0.0,
            "Temperature feedback for warming should be negative, got {}",
            forcing
        );

        // Expected: -0.037 * 2.0 = -0.074 W/m²
        let expected = -0.037 * 2.0;
        assert!(
            (forcing - expected).abs() < 1e-10,
            "Temperature feedback: expected {}, got {}",
            expected,
            forcing
        );
    }

    #[test]
    fn test_temp_feedback_linear() {
        let component = default_component();

        let forcing_1 = component.calculate_temp_feedback(1.0);
        let forcing_2 = component.calculate_temp_feedback(2.0);

        assert!(
            (forcing_2 - 2.0 * forcing_1).abs() < 1e-10,
            "Temperature feedback should be linear"
        );
    }

    // ===== Combined Forcing Tests =====

    #[test]
    fn test_total_forcing_is_sum_of_components() {
        let component = default_component();

        let result = component.calculate_forcings(
            1800.0, // EESC above reference
            1900.0, // CH4 ppb
            40.0,   // NOx
            500.0,  // CO
            100.0,  // NMVOC
            1.5,    // Temperature anomaly
        );

        let expected_total = result.strat_o3_erf + result.trop_o3_erf + result.temp_feedback_erf;

        assert!(
            (result.total() - expected_total).abs() < 1e-10,
            "Total should be sum of components"
        );
    }

    #[test]
    fn test_component_signs() {
        let component = default_component();

        let result = component.calculate_forcings(
            2000.0, // EESC above reference (should give negative)
            1900.0, // CH4 above PI (should give positive)
            40.0,   // NOx
            500.0,  // CO
            100.0,  // NMVOC
            1.0,    // Positive temperature anomaly (should give negative)
        );

        assert!(
            result.strat_o3_erf < 0.0,
            "Stratospheric forcing should be negative (cooling)"
        );
        assert!(
            result.trop_o3_erf > 0.0,
            "Tropospheric forcing should be positive (warming)"
        );
        assert!(
            result.temp_feedback_erf < 0.0,
            "Temperature feedback should be negative"
        );
    }

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        assert_eq!(defs.len(), 9); // 6 inputs + 3 outputs

        let input_names: Vec<_> = defs
            .iter()
            .filter(|d| d.requirement_type == RequirementType::Input)
            .map(|d| d.name.as_str())
            .collect();

        assert!(input_names.contains(&"EESC"));
        assert!(input_names.contains(&"Atmospheric Concentration|CH4"));
        assert!(input_names.contains(&"Emissions|NOx"));
        assert!(input_names.contains(&"Emissions|CO"));
        assert!(input_names.contains(&"Emissions|NMVOC"));
        assert!(input_names.contains(&"Surface Temperature|Global"));

        let output_names: Vec<_> = defs
            .iter()
            .filter(|d| d.requirement_type == RequirementType::Output)
            .map(|d| d.name.as_str())
            .collect();

        assert!(output_names.contains(&"Effective Radiative Forcing|O3|Stratospheric"));
        assert!(output_names.contains(&"Effective Radiative Forcing|O3|Tropospheric"));
        assert!(output_names.contains(&"Effective Radiative Forcing|O3|Temperature Feedback"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let component = OzoneForcing::from_parameters(OzoneForcingParameters {
            eesc_reference: 1500.0,
            strat_o3_scale: -0.005,
            ..OzoneForcingParameters::default()
        });

        let json = serde_json::to_string(&component).unwrap();
        let restored: OzoneForcing = serde_json::from_str(&json).unwrap();

        assert!(
            (component.parameters().eesc_reference - restored.parameters().eesc_reference).abs()
                < 1e-10
        );
        assert!(
            (component.parameters().strat_o3_scale - restored.parameters().strat_o3_scale).abs()
                < 1e-10
        );
    }
}

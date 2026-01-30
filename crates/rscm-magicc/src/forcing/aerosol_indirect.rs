//! Aerosol Indirect Radiative Forcing Component
//!
//! Calculates radiative forcing from aerosol effects on cloud properties,
//! following a simplified version of MAGICC7 Module 06.
//!
//! # What This Component Does
//!
//! 1. Calculates aerosol burden from SOx and OC emissions (primary CCN sources)
//! 2. Applies logarithmic relationship to cloud droplet number concentration
//! 3. Computes cloud albedo effect forcing (first indirect effect)
//!
//! # Inputs
//!
//! - `Emissions|SOx` (Tg S/yr) - Sulfur dioxide emissions (primary CCN source)
//! - `Emissions|OC` (Tg OC/yr) - Organic carbon emissions (secondary CCN source)
//!
//! # Outputs
//!
//! - `Effective Radiative Forcing|Aerosol|Indirect` (W/m²) - Cloud indirect forcing
//!
//! # Differences from MAGICC7 Module 06
//!
//! This is a simplified implementation:
//!
//! - **Species weights**: MAGICC7 uses detailed species weights for CCN contribution
//!   (SOx, OC, BC, nitrate, sea salt with different industrial/biomass/natural splits).
//!   Here, only SOx and OC are considered as the dominant CCN sources.
//! - **Regional patterns**: MAGICC7 applies detailed regional forcing patterns from
//!   Hansen et al. (2005). Here, forcing is scalar (global mean).
//! - **Cloud lifetime effect**: MAGICC7 has separate cloud albedo and lifetime
//!   effect parameters. Here, only cloud albedo effect is implemented
//!   (lifetime effect is often set to zero in default configs anyway).
//! - **Optical thickness proxy**: MAGICC7 uses aerosol optical thickness as a proxy
//!   for particle number. Here, emissions are used directly.
//! - **CDNC parameterisation**: MAGICC7 uses log10(aerosol number index).
//!   Here, a simpler ln(1 + B/B0) relationship is used.
//! - **Normalisation**: MAGICC7 normalises to a specific year with complex
//!   sector handling. Here, forcing is relative to pre-industrial emissions.

use crate::parameters::AerosolIndirectParameters;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Aerosol indirect radiative forcing component
///
/// Calculates cloud albedo effect forcing from aerosol emissions.
/// Uses a logarithmic relationship between aerosol burden and forcing,
/// reflecting the saturation of CCN effects at high aerosol loadings.
///
/// # Formulation
///
/// The forcing is calculated as:
///
/// $$RF = \alpha \cdot \ln\left(1 + \frac{B - B_{pi}}{B_0}\right)$$
///
/// where:
/// - $\alpha$ is the cloud albedo coefficient (negative, cooling)
/// - $B = w_{SOx} \cdot E_{SOx} + w_{OC} \cdot E_{OC}$ is the aerosol burden proxy
/// - $B_{pi}$ is the pre-industrial burden
/// - $B_0$ is a reference burden scale
///
/// The logarithmic form captures the saturation of the CCN effect: doubling
/// aerosol from low levels has a larger effect than doubling from high levels.
///
/// # Physical Interpretation
///
/// More aerosols provide more cloud condensation nuclei (CCN), which leads to:
/// 1. More cloud droplets per unit water content
/// 2. Smaller droplet sizes
/// 3. Higher cloud reflectivity (albedo)
/// 4. Negative (cooling) radiative forcing
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["forcing", "aerosol", "indirect", "cloud", "magicc"], category = "Radiative Forcing")]
#[inputs(
    sox_emissions { name = "Emissions|SOx", unit = "Tg S/yr" },
    oc_emissions { name = "Emissions|OC", unit = "Tg OC/yr" },
)]
#[outputs(
    indirect_erf { name = "Effective Radiative Forcing|Aerosol|Indirect", unit = "W/m^2" },
)]
pub struct AerosolIndirect {
    parameters: AerosolIndirectParameters,
}

impl AerosolIndirect {
    /// Create a new aerosol indirect forcing component with default parameters
    pub fn new() -> Self {
        Self::from_parameters(AerosolIndirectParameters::default())
    }

    /// Create a new aerosol indirect forcing component from parameters
    pub fn from_parameters(parameters: AerosolIndirectParameters) -> Self {
        Self { parameters }
    }

    /// Get the parameters
    pub fn parameters(&self) -> &AerosolIndirectParameters {
        &self.parameters
    }

    /// Calculate aerosol burden proxy from emissions
    ///
    /// The burden is a weighted sum of SOx and OC emissions, with SOx
    /// being the primary CCN source.
    pub fn calculate_burden(&self, sox: FloatValue, oc: FloatValue) -> FloatValue {
        let p = &self.parameters;
        p.sox_weight * sox + p.oc_weight * oc
    }

    /// Calculate pre-industrial aerosol burden
    pub fn preindustrial_burden(&self) -> FloatValue {
        let p = &self.parameters;
        self.calculate_burden(p.sox_pi, p.oc_pi)
    }

    /// Calculate cloud albedo effect forcing
    ///
    /// Uses logarithmic relationship:
    /// RF = alpha * ln(1 + (B - B_pi) / B0)
    ///
    /// The forcing is relative to pre-industrial to ensure RF=0 at PI.
    pub fn calculate_cloud_albedo(&self, sox: FloatValue, oc: FloatValue) -> FloatValue {
        let p = &self.parameters;

        let burden = self.calculate_burden(sox, oc);
        let burden_pi = self.preindustrial_burden();
        let delta_burden = burden - burden_pi;

        if delta_burden <= 0.0 {
            // At or below pre-industrial: no forcing
            return 0.0;
        }

        // Logarithmic relationship
        // The (1 + ...) ensures well-behaved log at small deltas
        p.cloud_albedo_coefficient * (1.0 + delta_burden / p.reference_burden).ln()
    }

    /// Calculate total indirect aerosol forcing
    ///
    /// Currently only cloud albedo effect is implemented.
    /// Cloud lifetime effect would be added here if implemented.
    pub fn calculate_forcing(&self, sox: FloatValue, oc: FloatValue) -> FloatValue {
        self.calculate_cloud_albedo(sox, oc)
    }
}

impl Default for AerosolIndirect {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for AerosolIndirect {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = AerosolIndirectInputs::from_input_state(input_state);

        let sox = inputs.sox_emissions.at_start();
        let oc = inputs.oc_emissions.at_start();

        let forcing = self.calculate_forcing(sox, oc);

        let outputs = AerosolIndirectOutputs {
            indirect_erf: forcing,
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rscm_core::component::RequirementType;

    fn default_component() -> AerosolIndirect {
        AerosolIndirect::from_parameters(AerosolIndirectParameters::default())
    }

    // ===== Burden Calculation Tests =====

    #[test]
    fn test_burden_calculation() {
        let component = default_component();
        let p = component.parameters();

        let burden = component.calculate_burden(10.0, 20.0);
        let expected = p.sox_weight * 10.0 + p.oc_weight * 20.0;

        assert!(
            (burden - expected).abs() < 1e-10,
            "Burden calculation: expected {}, got {}",
            expected,
            burden
        );
    }

    #[test]
    fn test_sox_dominates_burden() {
        let component = default_component();

        // Equal mass emissions
        let burden_sox = component.calculate_burden(100.0, 0.0);
        let burden_oc = component.calculate_burden(0.0, 100.0);

        assert!(
            burden_sox > burden_oc,
            "SOx should contribute more to burden than OC (equal mass): SOx={}, OC={}",
            burden_sox,
            burden_oc
        );
    }

    // ===== Forcing Calculation Tests =====

    #[test]
    fn test_zero_forcing_at_preindustrial() {
        let component = default_component();
        let p = component.parameters();

        let forcing = component.calculate_forcing(p.sox_pi, p.oc_pi);

        assert!(
            forcing.abs() < 1e-10,
            "Forcing at pre-industrial should be zero, got {}",
            forcing
        );
    }

    #[test]
    fn test_forcing_is_negative_above_preindustrial() {
        let component = default_component();
        let p = component.parameters();

        // Elevated emissions above pre-industrial
        let forcing = component.calculate_forcing(p.sox_pi + 50.0, p.oc_pi + 20.0);

        assert!(
            forcing < 0.0,
            "Indirect forcing above pre-industrial should be negative (cooling), got {}",
            forcing
        );
    }

    #[test]
    fn test_forcing_is_zero_below_preindustrial() {
        let component = default_component();
        let p = component.parameters();

        // Below pre-industrial emissions (hypothetical clean scenario)
        let forcing = component.calculate_forcing(p.sox_pi * 0.5, p.oc_pi * 0.5);

        assert!(
            forcing.abs() < 1e-10,
            "Forcing below pre-industrial should be zero, got {}",
            forcing
        );
    }

    #[test]
    fn test_forcing_logarithmic_relationship() {
        let component = default_component();
        let p = component.parameters();

        // Test that forcing shows logarithmic saturation
        let forcing_low = component.calculate_forcing(p.sox_pi + 25.0, p.oc_pi);
        let forcing_mid = component.calculate_forcing(p.sox_pi + 50.0, p.oc_pi);
        let forcing_high = component.calculate_forcing(p.sox_pi + 100.0, p.oc_pi);

        // First increment should have larger effect than second
        // (characteristic of log relationship)
        let increment_1 = forcing_mid - forcing_low; // Both negative, so this is negative
        let increment_2 = forcing_high - forcing_mid;

        // For log relationship, |increment_1/25| > |increment_2/50|
        // i.e., per-unit forcing decreases at higher levels
        let per_unit_1 = (increment_1 / 25.0).abs();
        let per_unit_2 = (increment_2 / 50.0).abs();

        assert!(
            per_unit_1 > per_unit_2,
            "Logarithmic saturation: per-unit forcing should decrease ({} > {})",
            per_unit_1,
            per_unit_2
        );
    }

    #[test]
    fn test_forcing_scales_with_sox() {
        let component = default_component();
        let p = component.parameters();

        let forcing_0 = component.calculate_forcing(p.sox_pi, p.oc_pi);
        let forcing_50 = component.calculate_forcing(p.sox_pi + 50.0, p.oc_pi);
        let forcing_100 = component.calculate_forcing(p.sox_pi + 100.0, p.oc_pi);

        // Higher SOx should give stronger (more negative) forcing
        assert!(
            forcing_50 < forcing_0,
            "Higher SOx should give more negative forcing"
        );
        assert!(
            forcing_100 < forcing_50,
            "Even higher SOx should give even more negative forcing"
        );
    }

    // ===== Realistic Magnitude Tests =====

    #[test]
    fn test_realistic_forcing_magnitude() {
        let component = default_component();

        // Approximate modern emissions (circa 2019)
        // SOx: ~50-60 Tg S/yr anthropogenic
        // OC: ~30-40 Tg OC/yr
        let forcing = component.calculate_forcing(60.0, 40.0);

        // AR6 cloud albedo (ERFaci): -0.89 [-1.68 to -0.09] W/m²
        // Our simplified model should be in roughly the right range
        assert!(
            forcing > -2.0 && forcing < 0.0,
            "Modern indirect forcing should be negative and plausible, got {} W/m²",
            forcing
        );
    }

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        assert_eq!(defs.len(), 3); // 2 inputs + 1 output

        let input_names: Vec<_> = defs
            .iter()
            .filter(|d| d.requirement_type == RequirementType::Input)
            .map(|d| d.name.as_str())
            .collect();

        assert!(input_names.contains(&"Emissions|SOx"));
        assert!(input_names.contains(&"Emissions|OC"));

        let output_names: Vec<_> = defs
            .iter()
            .filter(|d| d.requirement_type == RequirementType::Output)
            .map(|d| d.name.as_str())
            .collect();

        assert!(output_names.contains(&"Effective Radiative Forcing|Aerosol|Indirect"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let component = AerosolIndirect::from_parameters(AerosolIndirectParameters {
            cloud_albedo_coefficient: -1.2,
            reference_burden: 75.0,
            ..AerosolIndirectParameters::default()
        });

        let json = serde_json::to_string(&component).unwrap();
        let restored: AerosolIndirect = serde_json::from_str(&json).unwrap();

        assert!(
            (component.parameters().cloud_albedo_coefficient
                - restored.parameters().cloud_albedo_coefficient)
                .abs()
                < 1e-10
        );
        assert!(
            (component.parameters().reference_burden - restored.parameters().reference_burden)
                .abs()
                < 1e-10
        );
    }

    // ===== Direct + Indirect Sum Test =====

    #[test]
    fn test_direct_plus_indirect_can_be_summed() {
        // This test verifies the design: direct (FourBox) and indirect (scalar)
        // can be combined downstream. The indirect forcing applies uniformly
        // to all regions, while direct forcing is regionally distributed.

        let indirect = default_component();

        // Modern-ish emissions
        let indirect_forcing = indirect.calculate_forcing(60.0, 40.0);

        // Direct forcing would be computed separately by AerosolDirect
        // This test just verifies indirect forcing is a sensible scalar
        // that can be broadcast to regions if needed

        assert!(
            indirect_forcing.is_finite(),
            "Indirect forcing should be finite"
        );
        assert!(
            indirect_forcing < 0.0,
            "Indirect forcing should be negative at modern emissions"
        );
    }
}

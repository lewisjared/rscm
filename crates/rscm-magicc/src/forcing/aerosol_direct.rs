//! Aerosol Direct Radiative Forcing Component
//!
//! Calculates radiative forcing from aerosol particles that directly interact
//! with solar radiation through scattering (cooling) and absorption (warming).
//!
//! # What This Component Does
//!
//! 1. Calculates forcing from sulfate aerosols (SOx) - cooling via scattering
//! 2. Calculates forcing from black carbon (BC) - warming via absorption
//! 3. Calculates forcing from organic carbon (OC) - cooling via scattering
//! 4. Calculates forcing from nitrate aerosols (from NOx) - cooling via scattering
//! 5. Distributes total forcing across four regional boxes
//!
//! # Inputs
//!
//! - `Emissions|SOx` (Tg S/yr) - Sulfur dioxide emissions
//! - `Emissions|BC` (Tg BC/yr) - Black carbon emissions
//! - `Emissions|OC` (Tg OC/yr) - Organic carbon emissions
//! - `Emissions|NOx` (Tg N/yr) - NOx emissions (for nitrate formation)
//!
//! # Outputs
//!
//! - `Effective Radiative Forcing|Aerosol|Direct` (W/m², FourBox) - Regional direct aerosol forcing
//!
//! # Differences from MAGICC7 Module 05
//!
//! This is a simplified implementation:
//!
//! - **Source separation**: MAGICC7 separates industrial and biomass sources for
//!   each species with different treatment. Here, all sources are combined.
//! - **Optical thickness**: MAGICC7 can use historical optical thickness data
//!   and extrapolate with emissions. Here, forcing scales linearly with emissions.
//! - **Dust and BC on snow**: MAGICC7 includes mineral dust and BC-on-snow forcing.
//!   Not implemented here.
//! - **Harmonisation modes**: MAGICC7 has multiple harmonisation modes (APPLY=0,1,2,3).
//!   Here, only simple scaling to target is implemented.
//! - **Nitrate chemistry**: MAGICC7 uses Hauglustaine parameterisation with
//!   SOx-ammonia competition. Here, nitrate forcing scales linearly with NOx.
//! - **Efficacy**: MAGICC7 applies species-specific efficacy factors.
//!   Not implemented here.

use crate::parameters::AerosolDirectParameters;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{FourBoxSlice, ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Aerosol direct radiative forcing component
///
/// Calculates direct aerosol forcing from emissions of SOx, BC, OC, and NOx.
/// Each species contributes to forcing proportional to its emissions above
/// pre-industrial levels, with species-specific radiative efficiencies.
///
/// # Forcing Calculation
///
/// For each species $i$:
///
/// $$\Delta E_i = E_i - E_{i,pi}$$
/// $$RF_i = \alpha_i \cdot \Delta E_i$$
///
/// Total global forcing is the sum of species contributions.
///
/// # Regional Distribution
///
/// The global forcing is distributed across four boxes using emission-weighted
/// regional patterns:
///
/// $$RF_{region} = RF_{global} \cdot \sum_i w_{i,region} \cdot |RF_i| / \sum_i |RF_i|$$
///
/// This weights the regional pattern by each species' contribution to total
/// forcing magnitude.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["forcing", "aerosol", "direct", "magicc"], category = "Radiative Forcing")]
#[inputs(
    sox_emissions { name = "Emissions|SOx", unit = "Tg S/yr" },
    bc_emissions { name = "Emissions|BC", unit = "Tg BC/yr" },
    oc_emissions { name = "Emissions|OC", unit = "Tg OC/yr" },
    nox_emissions { name = "Emissions|NOx", unit = "Tg N/yr" },
)]
#[outputs(
    direct_erf { name = "Effective Radiative Forcing|Aerosol|Direct", unit = "W/m^2", grid = "FourBox" },
)]
pub struct AerosolDirect {
    parameters: AerosolDirectParameters,
}

impl AerosolDirect {
    /// Create a new aerosol direct forcing component with default parameters
    pub fn new() -> Self {
        Self::from_parameters(AerosolDirectParameters::default())
    }

    /// Create a new aerosol direct forcing component from parameters
    pub fn from_parameters(parameters: AerosolDirectParameters) -> Self {
        Self { parameters }
    }

    /// Get the parameters
    pub fn parameters(&self) -> &AerosolDirectParameters {
        &self.parameters
    }

    /// Calculate forcing for each species (internal calculation)
    ///
    /// Returns (sox_forcing, bc_forcing, oc_forcing, nitrate_forcing)
    pub fn calculate_species_forcing(
        &self,
        sox: FloatValue,
        bc: FloatValue,
        oc: FloatValue,
        nox: FloatValue,
    ) -> SpeciesForcing {
        let p = &self.parameters;

        // Calculate emissions relative to pre-industrial
        let delta_sox = sox - p.sox_pi;
        let delta_bc = bc - p.bc_pi;
        let delta_oc = oc - p.oc_pi;
        let delta_nox = nox - p.nox_pi;

        SpeciesForcing {
            sox: p.sox_coefficient * delta_sox,
            bc: p.bc_coefficient * delta_bc,
            oc: p.oc_coefficient * delta_oc,
            nitrate: p.nitrate_coefficient * delta_nox,
        }
    }

    /// Calculate total global direct aerosol forcing
    pub fn calculate_global_forcing(
        &self,
        sox: FloatValue,
        bc: FloatValue,
        oc: FloatValue,
        nox: FloatValue,
    ) -> FloatValue {
        let species = self.calculate_species_forcing(sox, bc, oc, nox);
        species.total()
    }

    /// Distribute forcing regionally using species-weighted patterns
    ///
    /// Each species has its own regional distribution pattern. The final
    /// regional distribution is weighted by each species' contribution to
    /// total forcing magnitude.
    pub fn distribute_regional(&self, species: &SpeciesForcing) -> FourBoxSlice {
        let p = &self.parameters;
        let total_forcing = species.total();

        if total_forcing.abs() < 1e-15 {
            // No significant forcing, return zeros
            return FourBoxSlice::uniform(0.0);
        }

        // Calculate absolute forcing weights for each species
        let total_abs =
            species.sox.abs() + species.bc.abs() + species.oc.abs() + species.nitrate.abs();

        if total_abs < 1e-15 {
            // All species near zero, distribute uniformly
            return FourBoxSlice::uniform(total_forcing / 4.0);
        }

        // Calculate weighted regional pattern
        let mut regional = [0.0; 4];
        for i in 0..4 {
            let weighted_pattern = (species.sox.abs() * p.sox_regional[i]
                + species.bc.abs() * p.bc_regional[i]
                + species.oc.abs() * p.oc_regional[i]
                + species.nitrate.abs() * p.nitrate_regional[i])
                / total_abs;

            regional[i] = total_forcing * weighted_pattern;
        }

        FourBoxSlice::from_array(regional)
    }

    /// Calculate all forcing components and distribute regionally
    pub fn calculate_forcing(
        &self,
        sox: FloatValue,
        bc: FloatValue,
        oc: FloatValue,
        nox: FloatValue,
    ) -> FourBoxSlice {
        let species = self.calculate_species_forcing(sox, bc, oc, nox);
        self.distribute_regional(&species)
    }
}

impl Default for AerosolDirect {
    fn default() -> Self {
        Self::new()
    }
}

/// Species-level forcing breakdown
#[derive(Debug, Clone, Copy)]
pub struct SpeciesForcing {
    /// Sulfate aerosol forcing (negative, cooling)
    pub sox: FloatValue,
    /// Black carbon forcing (positive, warming)
    pub bc: FloatValue,
    /// Organic carbon forcing (negative, cooling)
    pub oc: FloatValue,
    /// Nitrate aerosol forcing (negative, cooling)
    pub nitrate: FloatValue,
}

impl SpeciesForcing {
    /// Total forcing from all species
    pub fn total(&self) -> FloatValue {
        self.sox + self.bc + self.oc + self.nitrate
    }
}

#[typetag::serde]
impl Component for AerosolDirect {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = AerosolDirectInputs::from_input_state(input_state);

        let sox = inputs.sox_emissions.at_start();
        let bc = inputs.bc_emissions.at_start();
        let oc = inputs.oc_emissions.at_start();
        let nox = inputs.nox_emissions.at_start();

        let regional_forcing = self.calculate_forcing(sox, bc, oc, nox);

        let outputs = AerosolDirectOutputs {
            direct_erf: regional_forcing,
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rscm_core::component::RequirementType;
    use rscm_core::spatial::FourBoxRegion;

    fn default_component() -> AerosolDirect {
        AerosolDirect::from_parameters(AerosolDirectParameters::default())
    }

    // ===== Species Forcing Tests =====

    #[test]
    fn test_sox_causes_cooling() {
        let component = default_component();
        let p = component.parameters();

        // Elevated SOx emissions above pre-industrial
        let sox = p.sox_pi + 50.0; // +50 Tg S/yr
        let species = component.calculate_species_forcing(sox, p.bc_pi, p.oc_pi, p.nox_pi);

        assert!(
            species.sox < 0.0,
            "SOx should cause negative (cooling) forcing, got {}",
            species.sox
        );
        assert!(
            species.bc.abs() < 1e-10,
            "BC forcing should be zero at pre-industrial"
        );
        assert!(
            species.oc.abs() < 1e-10,
            "OC forcing should be zero at pre-industrial"
        );
    }

    #[test]
    fn test_bc_causes_warming() {
        let component = default_component();
        let p = component.parameters();

        // Elevated BC emissions above pre-industrial
        let bc = p.bc_pi + 5.0; // +5 Tg BC/yr
        let species = component.calculate_species_forcing(p.sox_pi, bc, p.oc_pi, p.nox_pi);

        assert!(
            species.bc > 0.0,
            "BC should cause positive (warming) forcing, got {}",
            species.bc
        );
    }

    #[test]
    fn test_oc_causes_cooling() {
        let component = default_component();
        let p = component.parameters();

        // Elevated OC emissions above pre-industrial
        let oc = p.oc_pi + 20.0; // +20 Tg OC/yr
        let species = component.calculate_species_forcing(p.sox_pi, p.bc_pi, oc, p.nox_pi);

        assert!(
            species.oc < 0.0,
            "OC should cause negative (cooling) forcing, got {}",
            species.oc
        );
    }

    #[test]
    fn test_nitrate_causes_cooling() {
        let component = default_component();
        let p = component.parameters();

        // Elevated NOx emissions above pre-industrial
        let nox = p.nox_pi + 30.0; // +30 Tg N/yr
        let species = component.calculate_species_forcing(p.sox_pi, p.bc_pi, p.oc_pi, nox);

        assert!(
            species.nitrate < 0.0,
            "Nitrate should cause negative (cooling) forcing, got {}",
            species.nitrate
        );
    }

    #[test]
    fn test_zero_forcing_at_preindustrial() {
        let component = default_component();
        let p = component.parameters();

        // At pre-industrial levels
        let species = component.calculate_species_forcing(p.sox_pi, p.bc_pi, p.oc_pi, p.nox_pi);

        assert!(
            species.total().abs() < 1e-10,
            "Total forcing at pre-industrial should be zero, got {}",
            species.total()
        );
    }

    #[test]
    fn test_forcing_scales_linearly() {
        let component = default_component();
        let p = component.parameters();

        let sox_10 = p.sox_pi + 10.0;
        let sox_20 = p.sox_pi + 20.0;

        let forcing_10 = component.calculate_species_forcing(sox_10, p.bc_pi, p.oc_pi, p.nox_pi);
        let forcing_20 = component.calculate_species_forcing(sox_20, p.bc_pi, p.oc_pi, p.nox_pi);

        // Doubling the delta should double the forcing
        assert!(
            (forcing_20.sox - 2.0 * forcing_10.sox).abs() < 1e-10,
            "SOx forcing should scale linearly"
        );
    }

    // ===== Regional Distribution Tests =====

    #[test]
    fn test_regional_forcing_sums_to_global() {
        let component = default_component();
        let p = component.parameters();

        // Modern-ish emissions
        let regional = component.calculate_forcing(
            p.sox_pi + 50.0, // SOx
            p.bc_pi + 5.0,   // BC
            p.oc_pi + 20.0,  // OC
            p.nox_pi + 30.0, // NOx
        );

        let global = component.calculate_global_forcing(
            p.sox_pi + 50.0,
            p.bc_pi + 5.0,
            p.oc_pi + 20.0,
            p.nox_pi + 30.0,
        );

        // Regional values should sum to global forcing
        // (regional distribution preserves total, not mean)
        let regional_sum: f64 = regional.as_array().iter().sum();

        assert!(
            (regional_sum - global).abs() < 1e-10,
            "Regional sum {} should equal global {}",
            regional_sum,
            global
        );
    }

    #[test]
    fn test_regional_pattern_reflects_species_weights() {
        let component = default_component();
        let p = component.parameters();

        // Pure SOx scenario
        let sox_only =
            component.calculate_species_forcing(p.sox_pi + 50.0, p.bc_pi, p.oc_pi, p.nox_pi);
        let regional_sox = component.distribute_regional(&sox_only);

        // Northern land should have most forcing (highest SOx weight)
        assert!(
            regional_sox[FourBoxRegion::NorthernLand].abs()
                > regional_sox[FourBoxRegion::SouthernOcean].abs(),
            "NH Land should have higher forcing than SH Ocean for SOx"
        );
    }

    #[test]
    fn test_regional_all_regions_have_forcing() {
        let component = default_component();
        let p = component.parameters();

        let regional = component.calculate_forcing(
            p.sox_pi + 50.0,
            p.bc_pi + 5.0,
            p.oc_pi + 20.0,
            p.nox_pi + 30.0,
        );

        // All regions should have non-zero forcing
        for region in [
            FourBoxRegion::NorthernOcean,
            FourBoxRegion::NorthernLand,
            FourBoxRegion::SouthernOcean,
            FourBoxRegion::SouthernLand,
        ] {
            assert!(
                regional[region].abs() > 1e-15,
                "Region {:?} should have non-zero forcing",
                region
            );
        }
    }

    // ===== Realistic Magnitude Tests =====

    #[test]
    fn test_realistic_forcing_magnitude() {
        let component = default_component();

        // Approximate modern emissions (circa 2019)
        // SOx: ~50-60 Tg S/yr anthropogenic
        // BC: ~8-10 Tg BC/yr
        // OC: ~30-40 Tg OC/yr
        // NOx: ~40-50 Tg N/yr
        let global = component.calculate_global_forcing(
            60.0, // SOx
            10.0, // BC
            40.0, // OC
            45.0, // NOx
        );

        // AR6 direct aerosol: -0.22 [-0.47 to +0.04] W/m²
        // Our simplified model should be in roughly the right range
        assert!(
            global > -1.0 && global < 0.5,
            "Modern direct aerosol forcing should be in plausible range, got {} W/m²",
            global
        );
    }

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        assert_eq!(defs.len(), 5); // 4 inputs + 1 output

        let input_names: Vec<_> = defs
            .iter()
            .filter(|d| d.requirement_type == RequirementType::Input)
            .map(|d| d.name.as_str())
            .collect();

        assert!(input_names.contains(&"Emissions|SOx"));
        assert!(input_names.contains(&"Emissions|BC"));
        assert!(input_names.contains(&"Emissions|OC"));
        assert!(input_names.contains(&"Emissions|NOx"));

        let output_names: Vec<_> = defs
            .iter()
            .filter(|d| d.requirement_type == RequirementType::Output)
            .map(|d| d.name.as_str())
            .collect();

        assert!(output_names.contains(&"Effective Radiative Forcing|Aerosol|Direct"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let component = AerosolDirect::from_parameters(AerosolDirectParameters {
            sox_coefficient: -0.004,
            bc_coefficient: 0.01,
            ..AerosolDirectParameters::default()
        });

        let json = serde_json::to_string(&component).unwrap();
        let restored: AerosolDirect = serde_json::from_str(&json).unwrap();

        assert!(
            (component.parameters().sox_coefficient - restored.parameters().sox_coefficient).abs()
                < 1e-10
        );
        assert!(
            (component.parameters().bc_coefficient - restored.parameters().bc_coefficient).abs()
                < 1e-10
        );
    }
}

//! CO2 Budget Component
//!
//! Closes the global carbon cycle by integrating emissions and uptakes to
//! update atmospheric CO2 concentration.
//!
//! # What This Component Does
//!
//! 1. Sums total emissions from fossil fuels and land use change
//! 2. Sums total uptake from terrestrial and ocean carbon sinks
//! 3. Calculates net change in atmospheric CO2 using mass balance
//! 4. Updates atmospheric CO2 concentration for the next timestep
//! 5. Calculates the airborne fraction (net emissions / total emissions)
//!
//! # Inputs
//!
//! - `Emissions|CO2|Fossil` (GtC/yr) - Fossil fuel and industrial emissions
//! - `Emissions|CO2|Land Use` (GtC/yr) - Land use change emissions
//! - `Carbon Flux|Terrestrial` (GtC/yr) - Net land uptake (positive = uptake)
//! - `Carbon Flux|Ocean` (GtC/yr) - Net ocean uptake (positive = uptake)
//!
//! # States (tracked between timesteps)
//!
//! - `Atmospheric Concentration|CO2` (ppm) - Atmospheric CO2 concentration
//!
//! # Outputs
//!
//! - `Emissions|CO2|Net` (GtC/yr) - Net emissions to atmosphere
//! - `Airborne Fraction|CO2` (1) - Fraction of emissions remaining in atmosphere
//!
//! # Differences from MAGICC7 Module 11
//!
//! This is a simplified implementation:
//!
//! - **No sub-annual integration**: MAGICC7 does monthly sub-stepping for ocean
//!   carbon cycle. Here, integration is annual (ocean sub-stepping is internal
//!   to the OceanCarbon component).
//! - **No mode switching**: MAGICC7 can switch between concentration-driven and
//!   emissions-driven modes mid-run. Here, only emissions-driven mode is supported.
//! - **No CH4 oxidation source**: MAGICC7 can add CO2 from methane oxidation.
//!   Not implemented.
//! - **No permafrost emissions**: MAGICC7 can include permafrost CO2 release.
//!   Not implemented.
//! - **No concentration cap**: MAGICC7 can cap CO2 concentration. Not implemented.
//! - **No inverse emissions calculation**: MAGICC7 calculates what emissions would
//!   be needed to match prescribed concentrations. Not implemented.

use crate::parameters::CO2BudgetParameters;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// CO2 budget component that integrates the global carbon cycle.
///
/// Implements the mass balance equation:
///
/// $$\frac{dC}{dt} = \frac{E_{fossil} + E_{landuse} - F_{land} - F_{ocean}}{GtC\_per\_ppm}$$
///
/// where:
/// - $C$ is atmospheric CO2 concentration (ppm)
/// - $E$ are emissions (GtC/yr)
/// - $F$ are uptake fluxes (GtC/yr, positive = uptake from atmosphere)
///
/// # Algorithm
///
/// For each timestep:
///
/// 1. Calculate total emissions: $E_{total} = E_{fossil} + E_{landuse}$
/// 2. Calculate total uptake: $F_{total} = F_{land} + F_{ocean}$
/// 3. Calculate net to atmosphere: $E_{net} = E_{total} - F_{total}$
/// 4. Update concentration: $C_{n+1} = C_n + E_{net} \times \Delta t / GtC\_per\_ppm$
/// 5. Calculate airborne fraction: $AF = E_{net} / E_{total}$
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "budget", "magicc"], category = "Carbon Cycle")]
#[inputs(
    fossil_emissions { name = "Emissions|CO2|Fossil", unit = "GtC/yr" },
    landuse_emissions { name = "Emissions|CO2|Land Use", unit = "GtC/yr" },
    terrestrial_flux { name = "Carbon Flux|Terrestrial", unit = "GtC/yr" },
    ocean_flux { name = "Carbon Flux|Ocean", unit = "GtC/yr" },
)]
#[states(
    co2_concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
)]
#[outputs(
    net_emissions { name = "Emissions|CO2|Net", unit = "GtC/yr" },
    airborne_fraction { name = "Airborne Fraction|CO2", unit = "1" },
)]
pub struct CO2Budget {
    parameters: CO2BudgetParameters,
}

impl CO2Budget {
    /// Create a new CO2 budget component with default parameters.
    pub fn new() -> Self {
        Self::from_parameters(CO2BudgetParameters::default())
    }

    /// Create a new CO2 budget component from parameters.
    pub fn from_parameters(parameters: CO2BudgetParameters) -> Self {
        Self { parameters }
    }

    /// Solve the CO2 budget for one timestep.
    ///
    /// # Arguments
    ///
    /// * `fossil_emissions` - Fossil fuel emissions (GtC/yr)
    /// * `landuse_emissions` - Land use change emissions (GtC/yr)
    /// * `terrestrial_flux` - Net land uptake (GtC/yr, positive = uptake)
    /// * `ocean_flux` - Net ocean uptake (GtC/yr, positive = uptake)
    /// * `co2_current` - Current CO2 concentration (ppm)
    /// * `dt` - Timestep (years)
    ///
    /// # Returns
    ///
    /// (new_co2, net_emissions, airborne_fraction)
    pub fn solve_budget(
        &self,
        fossil_emissions: FloatValue,
        landuse_emissions: FloatValue,
        terrestrial_flux: FloatValue,
        ocean_flux: FloatValue,
        co2_current: FloatValue,
        dt: FloatValue,
    ) -> (FloatValue, FloatValue, FloatValue) {
        // Total emissions to atmosphere
        let total_emissions = fossil_emissions + landuse_emissions;

        // Total uptake from atmosphere (positive fluxes are uptake)
        let total_uptake = terrestrial_flux + ocean_flux;

        // Net change to atmosphere
        let net_to_atm = total_emissions - total_uptake;

        // Update concentration
        let delta_co2 = (net_to_atm * dt) / self.parameters.gtc_per_ppm;
        let co2_next = co2_current + delta_co2;

        // Calculate airborne fraction
        // AF = net emissions / total emissions
        // When total emissions are zero or negative, AF is undefined
        let airborne_fraction = if total_emissions > 0.0 {
            net_to_atm / total_emissions
        } else {
            0.0
        };

        (co2_next, net_to_atm, airborne_fraction)
    }

    /// Get the parameters.
    pub fn parameters(&self) -> &CO2BudgetParameters {
        &self.parameters
    }
}

impl Default for CO2Budget {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for CO2Budget {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = CO2BudgetInputs::from_input_state(input_state);

        // Get current inputs
        let fossil = inputs.fossil_emissions.at_start();
        let landuse = inputs.landuse_emissions.at_start();
        let terrestrial = inputs.terrestrial_flux.at_start();
        let ocean = inputs.ocean_flux.at_start();
        let co2 = inputs.co2_concentration.at_start();

        let dt = t_next - t_current;

        // Solve budget
        let (co2_next, net_emissions, airborne_fraction) =
            self.solve_budget(fossil, landuse, terrestrial, ocean, co2, dt);

        let outputs = CO2BudgetOutputs {
            co2_concentration: co2_next,
            net_emissions,
            airborne_fraction,
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_component() -> CO2Budget {
        CO2Budget::from_parameters(CO2BudgetParameters::default())
    }

    // ===== Mass Conservation Tests =====

    #[test]
    fn test_mass_conservation_basic() {
        let component = default_component();
        let gtc_per_ppm = component.parameters.gtc_per_ppm;
        let co2_initial = 400.0;

        // Given: emissions = 10 GtC/yr, uptakes = 4 GtC/yr
        // Net to atmosphere = 6 GtC/yr
        // Delta CO2 = 6 / 2.123 ≈ 2.83 ppm/yr
        let (co2_next, net_emissions, _) = component.solve_budget(
            10.0, // fossil
            0.0,  // landuse
            2.0,  // terrestrial uptake
            2.0,  // ocean uptake
            co2_initial,
            1.0, // 1 year
        );

        assert!(
            (net_emissions - 6.0).abs() < 1e-10,
            "Net emissions should be 6 GtC/yr, got {:.6}",
            net_emissions
        );

        let expected_delta = 6.0 / gtc_per_ppm;
        let actual_delta = co2_next - co2_initial;
        assert!(
            (actual_delta - expected_delta).abs() < 1e-10,
            "CO2 change should be {:.4} ppm, got {:.4} ppm",
            expected_delta,
            actual_delta
        );
    }

    #[test]
    fn test_mass_conservation_with_landuse() {
        let component = default_component();
        let gtc_per_ppm = component.parameters.gtc_per_ppm;
        let co2_initial = 350.0;

        // Emissions: 8 GtC/yr fossil + 2 GtC/yr landuse = 10 GtC/yr total
        // Uptakes: 3 GtC/yr land + 2 GtC/yr ocean = 5 GtC/yr total
        // Net = 10 - 5 = 5 GtC/yr
        let (co2_next, net_emissions, _) = component.solve_budget(
            8.0, // fossil
            2.0, // landuse
            3.0, // terrestrial uptake
            2.0, // ocean uptake
            co2_initial,
            1.0,
        );

        assert!(
            (net_emissions - 5.0).abs() < 1e-10,
            "Net emissions should be 5 GtC/yr, got {:.6}",
            net_emissions
        );

        let expected_delta = 5.0 / gtc_per_ppm;
        let actual_delta = co2_next - co2_initial;
        assert!(
            (actual_delta - expected_delta).abs() < 1e-10,
            "CO2 change should be {:.4} ppm, got {:.4} ppm",
            expected_delta,
            actual_delta
        );
    }

    // ===== Steady State Tests =====

    #[test]
    fn test_steady_state_when_balanced() {
        let component = default_component();
        let co2_initial = 400.0;

        // When emissions = uptakes, concentration should be stable
        let (co2_next, net_emissions, _) = component.solve_budget(
            10.0, // fossil
            0.0,  // landuse
            6.0,  // terrestrial uptake
            4.0,  // ocean uptake
            co2_initial,
            1.0,
        );

        assert!(
            net_emissions.abs() < 1e-10,
            "Net emissions should be zero at steady state, got {:.6}",
            net_emissions
        );
        assert!(
            (co2_next - co2_initial).abs() < 1e-10,
            "CO2 should remain stable at steady state: {:.2} vs {:.2}",
            co2_next,
            co2_initial
        );
    }

    // ===== Declining Concentration Tests =====

    #[test]
    fn test_declining_with_zero_emissions() {
        let component = default_component();
        let co2_initial = 400.0;

        // Zero emissions with positive uptake → declining concentration
        let (co2_next, net_emissions, _) = component.solve_budget(
            0.0, // fossil
            0.0, // landuse
            2.0, // terrestrial uptake
            1.0, // ocean uptake
            co2_initial,
            1.0,
        );

        assert!(
            net_emissions < 0.0,
            "Net emissions should be negative (net uptake), got {:.6}",
            net_emissions
        );
        assert!(
            co2_next < co2_initial,
            "CO2 should decline with zero emissions and positive uptake: {:.2} vs {:.2}",
            co2_next,
            co2_initial
        );
    }

    #[test]
    fn test_declining_with_large_uptake() {
        let component = default_component();
        let co2_initial = 500.0;

        // Small emissions with large uptake
        let (co2_next, net_emissions, _) = component.solve_budget(
            5.0,  // fossil
            0.0,  // landuse
            10.0, // terrestrial uptake (large)
            5.0,  // ocean uptake
            co2_initial,
            1.0,
        );

        assert!(
            net_emissions < 0.0,
            "Net should be negative when uptake exceeds emissions: {:.6}",
            net_emissions
        );
        assert!(
            co2_next < co2_initial,
            "CO2 should decline: {:.2} vs {:.2}",
            co2_next,
            co2_initial
        );
    }

    // ===== Airborne Fraction Tests =====

    #[test]
    fn test_airborne_fraction_typical() {
        let component = default_component();

        // Typical scenario: ~45% airborne fraction
        // Emissions = 10 GtC/yr, uptakes = 5.5 GtC/yr
        // Net = 4.5 GtC/yr, AF = 4.5/10 = 0.45
        let (_, _, airborne_fraction) = component.solve_budget(
            10.0, // fossil
            0.0,  // landuse
            3.0,  // terrestrial uptake
            2.5,  // ocean uptake
            400.0, 1.0,
        );

        assert!(
            (airborne_fraction - 0.45).abs() < 1e-10,
            "Airborne fraction should be 0.45, got {:.4}",
            airborne_fraction
        );
    }

    #[test]
    fn test_airborne_fraction_zero_emissions() {
        let component = default_component();

        // With zero emissions, airborne fraction is undefined (set to 0)
        let (_, _, airborne_fraction) = component.solve_budget(
            0.0, // fossil
            0.0, // landuse
            2.0, // terrestrial uptake
            1.0, // ocean uptake
            400.0, 1.0,
        );

        assert!(
            airborne_fraction.abs() < 1e-10,
            "Airborne fraction should be 0 with zero emissions, got {:.4}",
            airborne_fraction
        );
    }

    #[test]
    fn test_airborne_fraction_high() {
        let component = default_component();

        // High emissions, low uptake → high airborne fraction
        // Emissions = 10, uptakes = 2, net = 8, AF = 0.8
        let (_, _, airborne_fraction) = component.solve_budget(
            10.0, // fossil
            0.0,  // landuse
            1.0,  // terrestrial uptake
            1.0,  // ocean uptake
            400.0, 1.0,
        );

        assert!(
            (airborne_fraction - 0.8).abs() < 1e-10,
            "Airborne fraction should be 0.8, got {:.4}",
            airborne_fraction
        );
    }

    #[test]
    fn test_airborne_fraction_negative() {
        let component = default_component();

        // Net negative (uptake > emissions) → negative airborne fraction
        // Emissions = 5, uptakes = 8, net = -3, AF = -0.6
        let (_, _, airborne_fraction) = component.solve_budget(
            5.0, // fossil
            0.0, // landuse
            5.0, // terrestrial uptake
            3.0, // ocean uptake
            400.0, 1.0,
        );

        assert!(
            (airborne_fraction - (-0.6)).abs() < 1e-10,
            "Airborne fraction should be -0.6, got {:.4}",
            airborne_fraction
        );
    }

    // ===== Multi-Year Tests =====

    #[test]
    fn test_multi_year_accumulation() {
        let component = default_component();
        let gtc_per_ppm = component.parameters.gtc_per_ppm;

        let mut co2 = 280.0;
        let emissions = 10.0;
        let uptake = 5.0;
        let net = emissions - uptake;

        // Run for 10 years
        for _ in 0..10 {
            let (new_co2, _, _) = component.solve_budget(
                emissions,
                0.0, // emissions
                uptake / 2.0,
                uptake / 2.0, // split uptake
                co2,
                1.0,
            );
            co2 = new_co2;
        }

        // Total net = 5 GtC/yr × 10 years = 50 GtC
        // Delta CO2 = 50 / 2.123 ≈ 23.6 ppm
        let expected_delta = 10.0 * net / gtc_per_ppm;
        let actual_delta = co2 - 280.0;

        assert!(
            (actual_delta - expected_delta).abs() < 0.01,
            "Cumulative CO2 change should be {:.2} ppm, got {:.2} ppm",
            expected_delta,
            actual_delta
        );
    }

    // ===== Timestep Tests =====

    #[test]
    fn test_half_year_timestep() {
        let component = default_component();
        let gtc_per_ppm = component.parameters.gtc_per_ppm;
        let co2_initial = 400.0;

        // Half year should give half the change
        let (co2_half, _, _) = component.solve_budget(
            10.0,
            0.0, // emissions
            3.0,
            2.0, // uptakes
            co2_initial,
            0.5, // half year
        );

        let expected_delta = (5.0 * 0.5) / gtc_per_ppm;
        let actual_delta = co2_half - co2_initial;

        assert!(
            (actual_delta - expected_delta).abs() < 1e-10,
            "Half-year change should be {:.4} ppm, got {:.4} ppm",
            expected_delta,
            actual_delta
        );
    }

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        // Should have: 4 inputs + 1 state + 2 outputs = 7 definitions
        assert!(
            defs.len() >= 7,
            "Should have at least 7 definitions, got {}",
            defs.len()
        );

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Emissions|CO2|Fossil"));
        assert!(names.contains(&"Emissions|CO2|Land Use"));
        assert!(names.contains(&"Carbon Flux|Terrestrial"));
        assert!(names.contains(&"Carbon Flux|Ocean"));
        assert!(names.contains(&"Atmospheric Concentration|CO2"));
        assert!(names.contains(&"Emissions|CO2|Net"));
        assert!(names.contains(&"Airborne Fraction|CO2"));
    }

    #[test]
    fn test_serialization() {
        let component = default_component();
        let json = serde_json::to_string(&component).expect("Serialization failed");
        let parsed: CO2Budget = serde_json::from_str(&json).expect("Deserialization failed");

        assert!(
            (component.parameters.gtc_per_ppm - parsed.parameters.gtc_per_ppm).abs() < 1e-10,
            "Parameters should survive round-trip serialization"
        );
    }

    // ===== Edge Cases =====

    #[test]
    fn test_negative_uptakes_are_emissions() {
        let component = default_component();
        let co2_initial = 400.0;

        // Negative uptake = source to atmosphere
        // Emissions = 10, uptakes = -2 (actually emissions), net = 12
        let (co2_next, net_emissions, _) = component.solve_budget(
            10.0,
            0.0, // emissions
            -1.0,
            -1.0, // negative uptakes (sources)
            co2_initial,
            1.0,
        );

        assert!(
            (net_emissions - 12.0).abs() < 1e-10,
            "Net should include negative uptakes as emissions: got {:.4}",
            net_emissions
        );
        assert!(
            co2_next > co2_initial,
            "CO2 should increase more with negative uptakes"
        );
    }

    #[test]
    fn test_very_high_emissions() {
        let component = default_component();

        // Extreme emissions scenario
        let (co2_next, net_emissions, airborne_fraction) = component.solve_budget(
            100.0, // extreme emissions
            0.0, 5.0, 5.0, 400.0, 1.0,
        );

        assert!(
            co2_next.is_finite(),
            "CO2 should remain finite under extreme emissions"
        );
        assert!(
            net_emissions.is_finite() && net_emissions > 0.0,
            "Net emissions should be finite and positive"
        );
        assert!(
            airborne_fraction.is_finite() && airborne_fraction > 0.0 && airborne_fraction <= 1.0,
            "Airborne fraction should be in (0,1]: got {:.4}",
            airborne_fraction
        );
    }

    #[test]
    fn test_gtc_ppm_conversion_factor() {
        // Verify the standard conversion factor
        let params = CO2BudgetParameters::default();
        assert!(
            (params.gtc_per_ppm - 2.123).abs() < 0.001,
            "Default GtC/ppm should be 2.123, got {}",
            params.gtc_per_ppm
        );
    }
}

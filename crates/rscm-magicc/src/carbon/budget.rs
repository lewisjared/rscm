//! CO2 Budget Component
//!
//! Closes the global carbon cycle by integrating emissions and uptakes to
//! update atmospheric CO2 concentration.
//!
//! # What This Component Does
//!
//! 1. Sums fossil and industrial emissions with CO2 from fossil CH4 oxidation
//! 2. Subtracts the net land and ocean uptake
//! 3. Updates atmospheric CO2 concentration for the next timestep
//! 4. Calculates the airborne fraction of the fossil sources
//!
//! # Inputs
//!
//! - `Emissions|CO2|Fossil` (GtC/yr) - Fossil fuel and industrial emissions
//! - `Emissions|CO2|CH4 Oxidation` (GtC/yr) - CO2 from oxidised fossil CH4
//! - `Carbon Flux|Terrestrial` (GtC/yr) - Net atmosphere-to-land flux (positive =
//!   uptake). Land-use emissions are part of this flux, as in MAGICC7, because
//!   deforestation removes carbon from the land pools.
//! - `Carbon Flux|Ocean` (GtC/yr) - Net ocean uptake (positive = uptake)
//!
//! # States (tracked between timesteps)
//!
//! - `Atmospheric Concentration|CO2` (ppm) - Atmospheric CO2 concentration
//!
//! # Outputs
//!
//! - `Emissions|CO2|Net` (GtC/yr) - Net emissions to atmosphere
//! - `Airborne Fraction|CO2` (1) - Fraction of fossil sources remaining in the atmosphere
//!
//! # Differences from MAGICC7 Module 11
//!
//! - **Annual integration**: MAGICC7 updates atmospheric CO2 monthly inside the
//!   ocean loop. `OceanCarbon` reproduces that monthly trajectory, so the annual
//!   sum here gives the same end-of-year concentration.
//! - **No mode switching**: MAGICC7 can switch between concentration-driven and
//!   emissions-driven modes mid-run. Here, only emissions-driven mode is supported.
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

/// CO2 budget integrator component
///
/// Closes the global carbon cycle mass balance by integrating the fossil
/// sources and the net land and ocean exchange:
///
/// $$\frac{dC}{dt} = \frac{E_{fossil} + E_{CH_4\,ox} - F_{land} - F_{ocean}}{\text{GtC\_per\_ppm}}$$
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "budget", "magicc"], category = "Carbon Cycle")]
#[inputs(
    fossil_emissions { name = "Emissions|CO2|Fossil", unit = "GtC/yr" },
    ch4_oxidation { name = "Emissions|CO2|CH4 Oxidation", unit = "GtC/yr" },
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
    /// Returns (new CO2 concentration, net emissions, airborne fraction).
    pub fn solve_budget(
        &self,
        fossil_emissions: FloatValue,
        ch4_oxidation: FloatValue,
        terrestrial_flux: FloatValue,
        ocean_flux: FloatValue,
        co2_current: FloatValue,
        dt: FloatValue,
    ) -> (FloatValue, FloatValue, FloatValue) {
        let sources = fossil_emissions + ch4_oxidation;
        let net_to_atm = sources - terrestrial_flux - ocean_flux;
        let co2_next = co2_current + net_to_atm * dt / self.parameters.gtc_per_ppm;

        // Undefined without positive sources
        let airborne_fraction = if sources > 0.0 {
            net_to_atm / sources
        } else {
            0.0
        };

        (co2_next, net_to_atm, airborne_fraction)
    }

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

        let (co2_next, net_emissions, airborne_fraction) = self.solve_budget(
            inputs.fossil_emissions.get(),
            inputs.ch4_oxidation.get(),
            inputs.terrestrial_flux.get(),
            inputs.ocean_flux.get(),
            inputs.co2_concentration.at_start(),
            t_next - t_current,
        );

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

    #[test]
    fn test_mass_conservation() {
        let component = default_component();
        let gtc_per_ppm = component.parameters.gtc_per_ppm;
        // 10 fossil + 0.5 oxidation - 3 land - 2 ocean = 5.5 GtC/yr
        let (co2_next, net, _) = component.solve_budget(10.0, 0.5, 3.0, 2.0, 400.0, 1.0);
        assert!((net - 5.5).abs() < 1e-12);
        assert!((co2_next - 400.0 - 5.5 / gtc_per_ppm).abs() < 1e-12);
    }

    #[test]
    fn test_land_use_enters_through_terrestrial_flux() {
        // A land source (negative uptake) raises CO2 without a separate input
        let component = default_component();
        let (co2_next, net, _) = component.solve_budget(0.0, 0.0, -1.0, 0.0, 280.0, 1.0);
        assert!((net - 1.0).abs() < 1e-12);
        assert!(co2_next > 280.0);
    }

    #[test]
    fn test_steady_state_when_balanced() {
        let component = default_component();
        let (co2_next, net, _) = component.solve_budget(5.0, 0.0, 3.0, 2.0, 350.0, 1.0);
        assert!(net.abs() < 1e-12);
        assert!((co2_next - 350.0).abs() < 1e-12);
    }

    #[test]
    fn test_declining_with_zero_emissions() {
        let component = default_component();
        let (co2_next, net, _) = component.solve_budget(0.0, 0.0, 1.0, 2.0, 400.0, 1.0);
        assert!(net < 0.0);
        assert!(co2_next < 400.0);
    }

    #[test]
    fn test_airborne_fraction_typical() {
        let component = default_component();
        let (_, _, af) = component.solve_budget(10.0, 0.0, 3.0, 2.5, 400.0, 1.0);
        assert!((af - 0.45).abs() < 1e-12);
    }

    #[test]
    fn test_airborne_fraction_zero_emissions() {
        let component = default_component();
        let (_, _, af) = component.solve_budget(0.0, 0.0, 1.0, 1.0, 400.0, 1.0);
        assert_eq!(af, 0.0);
    }

    #[test]
    fn test_multi_year_accumulation() {
        let component = default_component();
        let mut co2 = 280.0;
        for _ in 0..10 {
            co2 = component.solve_budget(5.0, 0.0, 1.0, 1.0, co2, 1.0).0;
        }
        let expected = 280.0 + 10.0 * 3.0 / component.parameters.gtc_per_ppm;
        assert!((co2 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_half_year_timestep() {
        let component = default_component();
        let (co2_half, _, _) = component.solve_budget(10.0, 0.0, 2.0, 2.0, 400.0, 0.5);
        let (co2_full, _, _) = component.solve_budget(10.0, 0.0, 2.0, 2.0, 400.0, 1.0);
        assert!(((co2_half - 400.0) * 2.0 - (co2_full - 400.0)).abs() < 1e-12);
    }

    #[test]
    fn test_definitions() {
        let component = default_component();
        let names: Vec<_> = component
            .definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert!(names.contains(&"Emissions|CO2|CH4 Oxidation".to_string()));
        assert!(!names.contains(&"Emissions|CO2|Land Use".to_string()));
    }
}

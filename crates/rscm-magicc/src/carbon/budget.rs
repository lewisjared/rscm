//! CO2 Budget Component
//!
//! Closes the global carbon cycle by integrating emissions and uptakes to
//! update the atmospheric CO2 concentration.
//! The concentration changes as
//!
//! $$\frac{dC}{dt} = \frac{E_{fossil} + E_{CH_4\,ox} - F_{land} - F_{ocean}}{\text{GtC\_per\_ppm}}$$
//!
//! # What This Component Does
//!
//! 1. Sums fossil emissions with CO2 from fossil CH4 oxidation
//! 2. Subtracts the net land and ocean uptake
//! 3. Updates atmospheric CO2 concentration for the next step
//! 4. Calculates the airborne fraction of the fossil sources
//!
//! Biosphere CO2 emissions are part of the net atmosphere to land flux, as in MAGICC7,
//! because deforestation removes carbon from the land pools.
//!
//! # Inputs
//!
//! - `Ocean Carbon Atmospheric Concentrations|CO2` (ppm), only with the `ocean_end_of_year` parameter:
//!   concentration after the ocean's last substep
//! - `Emissions|CO2|Permafrost` (GtC/yr), only with the `include_permafrost` parameter
//! - `Prescribed Atmospheric Concentrations|CO2` (ppm), only with the `concentration_switch_year` parameter
//!
//! # Differences from MAGICC7
//!
//! - **Monthly integration in the ocean**: in MAGICC7, atmospheric CO2 evolves over
//!   the ocean's substeps, and the concentration it ends the year on becomes next year's.
//!   `ocean_end_of_year` takes that value from `OceanCarbon`. Without it, the annual sum
//!   in [`CO2Budget::solve_budget`] differs from it only by rounding.
//! - **No concentration cap**: MAGICC7 can cap CO2 concentration. Not implemented.
//! - **No inverse fossil emissions**: before `concentration_switch_year` the
//!   concentration follows `Prescribed Atmospheric Concentrations|CO2` and
//!   `Net Emissions|CO2` is the implied net source, but fossil emissions are not inverted.

use crate::parameters::CO2BudgetParameters;
use crate::reference::{permafrost_emissions, prescribed_concentration, prescribed_definition};
use rscm_core::component::{
    Component, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use rscm_core::errors::RSCMResult;
use rscm_core::names::{
    ATMOSPHERIC_CONCENTRATIONS_CO2, EMISSIONS_CO2_PERMAFROST,
    OCEAN_CARBON_ATMOSPHERIC_CONCENTRATIONS_CO2,
};
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// CO2 budget integrator component.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "budget", "magicc"], category = "Carbon Cycle")]
#[inputs(
    fossil_emissions { name = "Emissions|CO2|Fossil", unit = "GtC/yr", description = "Fossil CO2 emissions over the step" },
    ch4_oxidation { name = "Emissions|CO2|CH4 Oxidation", unit = "GtC/yr", description = "CO2 from oxidised fossil CH4 over the step" },
    terrestrial_flux { name = "Net Atmosphere to Land Flux|CO2", unit = "GtC/yr", description = "Net atmosphere to land flux over the step, positive for uptake and including Biosphere CO2 emissions" },
    ocean_flux { name = "Net Atmosphere to Ocean Flux|CO2", unit = "GtC/yr", description = "Net atmosphere to ocean flux over the step, positive for uptake" },
)]
#[states(
    co2_concentration { name = "Atmospheric Concentrations|CO2", unit = "ppm", description = "Atmospheric CO2 concentration at the end of the step" },
)]
#[outputs(
    net_emissions { name = "Net Emissions|CO2", unit = "GtC/yr", description = "Net CO2 emissions to the atmosphere over the step" },
    airborne_fraction { name = "Airborne Fraction|CO2", unit = "1", description = "Fraction of the step's fossil sources that stays in the atmosphere" },
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

    /// Solve the CO2 budget for one step.
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

#[typetag::serde(name = "magicc.CO2Budget")]
impl Component for CO2Budget {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions = Self::generated_definitions();
        if self.parameters.include_permafrost {
            definitions.push(VariableDefinition::scalar_input(
                EMISSIONS_CO2_PERMAFROST,
                "GtC/yr",
            ));
        }
        if self.parameters.ocean_end_of_year {
            definitions.push(VariableDefinition::scalar_input(
                OCEAN_CARBON_ATMOSPHERIC_CONCENTRATIONS_CO2,
                "ppm",
            ));
        }
        definitions.extend(prescribed_definition(
            ATMOSPHERIC_CONCENTRATIONS_CO2,
            "ppm",
            self.parameters.concentration_switch_year,
        ));
        definitions
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = CO2BudgetInputs::from_input_state(input_state);

        let co2_start = inputs.co2_concentration.get();
        let fossil = inputs.fossil_emissions.get()
            + permafrost_emissions(
                self.parameters.include_permafrost,
                input_state,
                EMISSIONS_CO2_PERMAFROST,
            );
        let sources = fossil + inputs.ch4_oxidation.get();
        let (mut co2_next, mut net_emissions, mut airborne_fraction) = self.solve_budget(
            fossil,
            inputs.ch4_oxidation.get(),
            inputs.terrestrial_flux.get(),
            inputs.ocean_flux.get(),
            co2_start,
            t_next - t_current,
        );
        if self.parameters.ocean_end_of_year {
            co2_next = input_state
                .get_scalar_window(OCEAN_CARBON_ATMOSPHERIC_CONCENTRATIONS_CO2)
                .get();
        }
        if let Some(prescribed) = prescribed_concentration(
            input_state,
            ATMOSPHERIC_CONCENTRATIONS_CO2,
            self.parameters.concentration_switch_year,
            t_next,
        ) {
            co2_next = prescribed;
            net_emissions =
                (prescribed - co2_start) * self.parameters.gtc_per_ppm / (t_next - t_current);
            airborne_fraction = if sources > 0.0 {
                net_emissions / sources
            } else {
                0.0
            };
        }

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
    fn test_half_year_step() {
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
        assert!(!names.contains(&"Emissions|CO2|Biosphere".to_string()));
    }
}

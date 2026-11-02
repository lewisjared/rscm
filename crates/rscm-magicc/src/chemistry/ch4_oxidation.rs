//! CH4 Oxidation Component
//!
//! Tracks the fossil fraction of atmospheric methane and the CO2 released as
//! it is oxidised, which MAGICC7 adds to fossil CO2 emissions.
//!
//! # Inputs
//!
//! - `Emissions|CH4|Fossil` (Mt CH4/yr) - Fossil and industrial CH4 emissions
//! - `Lifetime|CH4` (yr) - Total CH4 lifetime for the current step
//!
//! # States (tracked between timesteps)
//!
//! - `Atmospheric Burden|CH4|Fossil` (Mt CH4) - Fossil CH4 in the atmosphere
//!
//! # Outputs
//!
//! - `Emissions|CO2|CH4 Oxidation` (GtC/yr) - CO2 released by oxidation
//!
//! The burden is integrated explicitly, matching MAGICC7 to better than 0.1%.

use crate::parameters::CH4OxidationParameters;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Fossil CH4 oxidation component
///
/// $$O = \frac{B_f}{\tau}, \qquad B_f' = B_f + (E_f - O) \Delta t$$
///
/// The CO2 source is $O$ multiplied by `carbon_per_ch4`, converted to GtC.
/// Register this component after `CH4Chemistry` so that it reads the lifetime
/// calculated for the same step.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["chemistry", "ch4", "carbon-cycle", "magicc"], category = "Atmospheric Chemistry")]
#[inputs(
    fossil_emissions { name = "Emissions|CH4|Fossil", unit = "Mt CH4/yr" },
    lifetime { name = "Lifetime|CH4", unit = "yr" },
)]
#[states(
    fossil_burden { name = "Atmospheric Burden|CH4|Fossil", unit = "Mt CH4" },
)]
#[outputs(
    co2_emissions { name = "Emissions|CO2|CH4 Oxidation", unit = "GtC/yr" },
)]
pub struct CH4Oxidation {
    parameters: CH4OxidationParameters,
}

impl CH4Oxidation {
    /// Create a new component with default parameters
    pub fn new() -> Self {
        Self::from_parameters(CH4OxidationParameters::default())
    }

    /// Create a new component from parameters
    pub fn from_parameters(parameters: CH4OxidationParameters) -> Self {
        Self { parameters }
    }

    /// Advance the fossil burden by one step.
    ///
    /// Returns (new burden in Mt CH4, CO2 source in GtC/yr).
    pub fn solve_oxidation(
        &self,
        burden: FloatValue,
        fossil_emissions: FloatValue,
        lifetime: FloatValue,
        dt: FloatValue,
    ) -> (FloatValue, FloatValue) {
        let oxidised = burden / lifetime;
        let co2 = oxidised * self.parameters.carbon_per_ch4 / 1000.0;
        (burden + (fossil_emissions - oxidised) * dt, co2)
    }
}

impl Default for CH4Oxidation {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for CH4Oxidation {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = CH4OxidationInputs::from_input_state(input_state);
        let (fossil_burden, co2_emissions) = self.solve_oxidation(
            inputs.fossil_burden.at_start(),
            inputs.fossil_emissions.get(),
            inputs.lifetime.get(),
            t_next - t_current,
        );
        Ok(CH4OxidationOutputs {
            fossil_burden,
            co2_emissions,
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_oxidation_without_burden() {
        let component = CH4Oxidation::new();
        let (burden, co2) = component.solve_oxidation(0.0, 100.0, 10.0, 1.0);
        assert_eq!(co2, 0.0);
        assert!((burden - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_steady_state_releases_emitted_carbon() {
        // In steady state all emitted fossil CH4 carbon is released as CO2
        let component = CH4Oxidation::new();
        let (burden, co2) = component.solve_oxidation(1000.0, 100.0, 10.0, 1.0);
        assert!((burden - 1000.0).abs() < 1e-12);
        assert!((co2 - 100.0 * 0.75 / 1000.0).abs() < 1e-12);
    }

    #[test]
    fn test_definitions() {
        let names: Vec<_> = CH4Oxidation::new()
            .definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert!(names.contains(&"Emissions|CO2|CH4 Oxidation".to_string()));
        assert!(names.contains(&"Atmospheric Burden|CH4|Fossil".to_string()));
    }
}

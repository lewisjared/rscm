//! CO2 Effective Radiative Forcing component
//!
//! This component calculates the effective radiative forcing (ERF) from CO2 concentrations
//! using the standard logarithmic relationship.

use rscm_core::component::{Component, InputState, OutputState, RequirementDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::standard_variables::{VAR_CO2_CONCENTRATION, VAR_CO2_ERF};
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export variable names for use in this module
const VAR_CONC_CO2: &str = VAR_CO2_CONCENTRATION.name;
const VAR_ERF_CO2: &str = VAR_CO2_ERF.name;

/// Parameters for the CO2 ERF component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CO2ERFParameters {
    /// ERF due to a doubling of atmospheric CO2 concentrations
    /// unit: W / m^2
    pub erf_2xco2: FloatValue,
    /// Pre-industrial atmospheric CO2 concentration
    /// unit: ppm
    pub conc_pi: FloatValue,
}

/// CO2 effective radiative forcing (ERF) calculations
///
/// Computes ERF using the standard logarithmic relationship:
/// $$ ERF = \frac{ERF_{2xCO2}}{\log(2)} \cdot \log\left(1 + \frac{C - C_0}{C_0}\right) $$
///
/// Where:
/// - $ERF_{2xCO2}$ is the ERF for a doubling of CO2
/// - $C$ is the current CO2 concentration
/// - $C_0$ is the pre-industrial CO2 concentration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CO2ERF {
    parameters: CO2ERFParameters,
}

impl CO2ERF {
    /// Create a new CO2ERF component from parameters
    pub fn from_parameters(parameters: CO2ERFParameters) -> Self {
        Self { parameters }
    }

    /// Calculate ERF from concentration
    ///
    /// This is the core physics calculation, extracted for testability.
    pub fn calculate_erf(&self, concentration: FloatValue) -> FloatValue {
        self.parameters.erf_2xco2 / 2.0_f64.ln()
            * (1.0 + (concentration - self.parameters.conc_pi) / self.parameters.conc_pi).ln()
    }
}

#[typetag::serde]
impl Component for CO2ERF {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::scalar_input(VAR_CONC_CO2, "ppm"),
            RequirementDefinition::scalar_output(VAR_ERF_CO2, "W / m^2"),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let concentration = input_state.get_latest(VAR_CONC_CO2);
        let erf = self.calculate_erf(concentration);

        Ok(HashMap::from([(VAR_ERF_CO2.to_string(), erf)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_parameters() -> CO2ERFParameters {
        CO2ERFParameters {
            erf_2xco2: 3.7,
            conc_pi: 278.0,
        }
    }

    #[test]
    fn test_calculate_erf_at_preindustrial() {
        let component = CO2ERF::from_parameters(default_parameters());
        let erf = component.calculate_erf(278.0);
        assert!(
            (erf - 0.0).abs() < 1e-10,
            "ERF at pre-industrial should be ~0"
        );
    }

    #[test]
    fn test_calculate_erf_at_2x_co2() {
        let params = default_parameters();
        let component = CO2ERF::from_parameters(params.clone());
        let erf = component.calculate_erf(params.conc_pi * 2.0);
        assert!(
            (erf - params.erf_2xco2).abs() < 1e-10,
            "ERF at 2xCO2 should equal erf_2xco2"
        );
    }

    #[test]
    fn test_definitions() {
        let component = CO2ERF::from_parameters(default_parameters());
        let defs = component.definitions();

        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].variable_name, VAR_CONC_CO2);
        assert_eq!(defs[1].variable_name, VAR_ERF_CO2);
    }
}

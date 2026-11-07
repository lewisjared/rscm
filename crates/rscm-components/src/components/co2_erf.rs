//! CO2 Effective Radiative Forcing component
//!
//! This component calculates the effective radiative forcing (ERF) from CO2 concentrations
//! using the standard logarithmic relationship.

use rscm_core::component::{
    Component, GridType, InputState, OutputState, ScalarWindow, VariableDefinition, VariableRole,
};
use rscm_core::errors::RscmResult;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Parameters for the CO2 ERF component
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CO2ErfParameters {
    /// ERF due to a doubling of atmospheric CO2 concentrations
    /// unit: W/m^2
    pub erf_2xco2: FloatValue,
    /// Pre-industrial atmospheric CO2 concentration
    /// unit: ppm
    pub co2_pi: FloatValue,
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
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["radiative-forcing", "co2", "simple", "stable"], category = "Radiative Forcing")]
#[inputs(
    concentration { name = "Atmospheric Concentrations|CO2", unit = "ppm" },
)]
#[outputs(
    erf { name = "Effective Radiative Forcing|CO2", unit = "W/m^2" },
)]
pub struct CO2Erf {
    parameters: CO2ErfParameters,
}

impl CO2Erf {
    /// Create a new CO2Erf component from parameters
    pub fn from_parameters(parameters: CO2ErfParameters) -> Self {
        Self { parameters }
    }

    /// Calculate ERF from concentration
    ///
    /// This is the core physics calculation, extracted for testability.
    pub fn calculate_erf(&self, concentration: FloatValue) -> FloatValue {
        self.parameters.erf_2xco2 / 2.0_f64.ln()
            * (1.0 + (concentration - self.parameters.co2_pi) / self.parameters.co2_pi).ln()
    }
}

#[typetag::serde]
impl Component for CO2Erf {
    fn definitions(&self) -> Vec<VariableDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        let inputs = CO2ErfInputs::from_input_state(input_state);
        let concentration = inputs.concentration.get();
        let erf = self.calculate_erf(concentration);

        Ok(CO2ErfOutputs { erf }.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_parameters() -> CO2ErfParameters {
        CO2ErfParameters {
            erf_2xco2: 3.7,
            co2_pi: 278.0,
        }
    }

    #[test]
    fn test_calculate_erf_at_preindustrial() {
        let component = CO2Erf::from_parameters(default_parameters());
        let erf = component.calculate_erf(278.0);
        assert!(
            (erf - 0.0).abs() < 1e-10,
            "ERF at pre-industrial should be ~0"
        );
    }

    #[test]
    fn test_calculate_erf_at_2x_co2() {
        let params = default_parameters();
        let component = CO2Erf::from_parameters(params.clone());
        let erf = component.calculate_erf(params.co2_pi * 2.0);
        assert!(
            (erf - params.erf_2xco2).abs() < 1e-10,
            "ERF at 2xCO2 should equal erf_2xco2"
        );
    }

    #[test]
    fn test_definitions() {
        let component = CO2Erf::from_parameters(default_parameters());
        let defs = component.definitions();

        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "Atmospheric Concentrations|CO2");
        assert_eq!(defs[1].name, "Effective Radiative Forcing|CO2");
    }
}

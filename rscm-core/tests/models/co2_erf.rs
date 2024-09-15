use rscm_core::component::{
    Component, InputState, OutputState, RequirementDefinition, RequirementType, State,
};
use rscm_core::errors::RSCMResult;
use rscm_core::timeseries::{FloatValue, Time};

#[derive(Debug, Clone)]
pub struct CO2ERFParameters {
    /// ERF due to a doubling of atmospheric CO_2 concentrations
    /// unit: W / m^2
    pub erf_2xco2: FloatValue,
    /// Pre-industrial atmospheric CO_2 concentration
    /// unit: ppm
    pub conc_pi: FloatValue,
}

#[derive(Debug, Clone)]
/// CO2 effective radiative forcing (ERF) calculations
pub struct CO2ERF {
    parameters: CO2ERFParameters,
}

impl CO2ERF {
    pub fn from_parameters(parameters: CO2ERFParameters) -> Self {
        Self { parameters }
    }
}

impl Component for CO2ERF {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::new(
                "Atmospheric Concentration|CO2",
                "ppm",
                RequirementType::Input,
            ),
            RequirementDefinition::new(
                "Effective Radiative Forcing|CO2",
                "W / m^2",
                RequirementType::Output,
            ),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let erf = self.parameters.erf_2xco2 / 2.0_f64.log10()
            * (1.0
                + (input_state.get("Atmospheric Concentration|CO2") - self.parameters.conc_pi)
                    / self.parameters.conc_pi)
                .log10();

        Ok(OutputState::from_vectors(vec![erf], self.output_names()))
    }
}

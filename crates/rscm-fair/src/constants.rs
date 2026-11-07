//! Physical constants from FaIR's `constants.py` and `earth_params.py`.

use rscm_core::component::{InputState, VariableDefinition};
use rscm_core::names::{EFFICACY_ADJUSTED_ERF, ERF, SURFACE_AIR_TEMPERATURE_CHANGE};
use rscm_core::timeseries::FloatValue;

/// Radius of the Earth (m).
pub const EARTH_RADIUS: FloatValue = 6_371_000.0;

/// Length of a tropical year (s).
pub const SECONDS_PER_YEAR: FloatValue = 60.0 * 60.0 * 24.0 * 365.24219;

pub(crate) fn surface_temperature() -> String {
    SURFACE_AIR_TEMPERATURE_CHANGE.to_string()
}

pub(crate) fn temperature_unit() -> String {
    "K".to_string()
}

pub(crate) fn forcing_unit() -> String {
    "W/m^2".to_string()
}

pub(crate) fn total_forcing() -> String {
    ERF.to_string()
}

pub(crate) fn efficacy_adjusted_forcing() -> String {
    EFFICACY_ADJUSTED_ERF.to_string()
}

pub(crate) fn one() -> FloatValue {
    1.0
}

/// Surface temperature input, one input per driver, then the forcing output.
pub(crate) fn driver_definitions<'a>(
    temperature: &str,
    drivers: impl Iterator<Item = (&'a str, &'a str)>,
    forcing: &str,
) -> Vec<VariableDefinition> {
    let mut definitions = vec![VariableDefinition::scalar_input(
        temperature,
        &temperature_unit(),
    )];
    definitions.extend(drivers.map(|(v, u)| VariableDefinition::scalar_input(v, u)));
    definitions.push(VariableDefinition::scalar_output(forcing, &forcing_unit()));
    definitions
}

/// Value the model hands this component for the step.
pub(crate) fn get(input_state: &InputState, name: &str) -> FloatValue {
    input_state.get_scalar_window(name).get()
}

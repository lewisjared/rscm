//! Fixed calendar-year references used by MAGICC7 feedbacks

use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::names::prescribed;
use rscm_core::state::ScalarWindow;
use rscm_core::timeseries::{FloatValue, Time};

/// Reject non-annual steps, because references are resolved as whole years back along the axis.
pub(crate) fn require_annual(component: &str, t_current: Time, t_next: Time) -> RSCMResult<()> {
    let dt = t_next - t_current;
    if !dt.is_finite() || (dt - 1.0).abs() >= 1e-6 {
        return Err(RSCMError::Error(format!(
            "{component} only supports annual steps (dt=1.0), got dt={dt}"
        )));
    }
    Ok(())
}

/// Value of an annual series at `year`, clamped to the stored time axis.
///
/// Feedback references in MAGICC7 are fixed calendar years, which may lie
/// before the run starts or after the current step.
pub(crate) fn value_at_year(window: &ScalarWindow, t_current: Time, year: Time) -> FloatValue {
    window
        .interpolate(t_current + (year - t_current).round())
        .expect("whole years from an annual step are nodes of the axis")
}

/// Value at the first step of the run.
pub(crate) fn first_value(window: &ScalarWindow) -> FloatValue {
    window
        .history(window.history_len() - 1)
        .expect("history_len values are stored")
}

/// Reference for a feedback starting at `start_year`.
///
/// MAGICC7 follows the current value until then and fixes the reference at the year before.
pub(crate) fn reference_value(
    window: &ScalarWindow,
    t_current: Time,
    start_year: Time,
) -> FloatValue {
    if t_current < start_year {
        window.get()
    } else {
        value_at_year(window, t_current, start_year - 1.0)
    }
}

/// Warming since `start_year`, zero before it and never negative.
///
/// This is MAGICC7's temperature feedback driver for the carbon cycle.
pub(crate) fn feedback_warming(
    temperature: &ScalarWindow,
    t_current: Time,
    start_year: Time,
) -> FloatValue {
    if t_current < start_year {
        return 0.0;
    }
    (temperature.get() - value_at_year(temperature, t_current, start_year)).max(0.0)
}

/// Permafrost emissions named `name` when `enabled`, otherwise zero.
pub(crate) fn permafrost_emissions(
    enabled: bool,
    input_state: &rscm_core::component::InputState,
    name: &str,
) -> FloatValue {
    if enabled {
        input_state.get_scalar_window(name).get()
    } else {
        0.0
    }
}

/// Input of prescribed `concentration` when a switch year is set.
pub(crate) fn prescribed_definition(
    concentration: &str,
    unit: &str,
    switch_year: Option<Time>,
) -> Option<rscm_core::component::VariableDefinition> {
    switch_year.map(|_| {
        rscm_core::component::VariableDefinition::scalar_input(&prescribed(concentration), unit)
    })
}

/// Prescribed `concentration` at the end of a step that ends by `switch_year`.
///
/// This is MAGICC7's `*_SWITCHFROMCONC2EMIS_YEAR`: concentrations follow the input
/// up to and including that year and are calculated from emissions afterwards.
pub(crate) fn prescribed_concentration(
    input_state: &rscm_core::component::InputState,
    concentration: &str,
    switch_year: Option<Time>,
    t_next: Time,
) -> Option<FloatValue> {
    if t_next > switch_year? {
        return None;
    }
    prescribed_input(input_state, &prescribed(concentration), switch_year, t_next)
}

/// [`prescribed_concentration`] given the name of its `prescribed` input.
pub(crate) fn prescribed_input(
    input_state: &rscm_core::component::InputState,
    prescribed: &str,
    switch_year: Option<Time>,
    t_next: Time,
) -> Option<FloatValue> {
    if t_next > switch_year? {
        return None;
    }
    Some(input_state.get_scalar_window(prescribed).get())
}

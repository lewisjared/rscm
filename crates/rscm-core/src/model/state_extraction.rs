//! State extraction functions for model components.

use crate::component::InputState;
use crate::state::TransformContext;
use crate::timeseries::Time;
use crate::timeseries_collection::TimeseriesCollection;

/// Extract the input state for the current time step.
///
/// By default, for endogenous variables which are calculated as part of the model
/// the most recent value is used, whereas, for exogenous variables the values are linearly
/// interpolated.
/// This ensures that state calculated from previous components within the same timestep
/// is used.
///
/// The result should contain values for the current time step for all input variables.
pub fn extract_state(
    collection: &TimeseriesCollection,
    input_names: Vec<String>,
    t_current: Time,
) -> InputState<'_> {
    let mut state = Vec::new();

    input_names.into_iter().for_each(|name| {
        let ts = collection
            .get_by_name(name.as_str())
            .unwrap_or_else(|| panic!("No timeseries with variable='{}'", name));
        state.push(ts);
    });

    InputState::build(state, t_current)
}

/// Extract the input state with transform context for grid aggregation.
///
/// Like `extract_state`, but includes transformation context for automatic
/// grid aggregation when reading variables at coarser resolutions.
pub fn extract_state_with_transforms(
    collection: &TimeseriesCollection,
    input_names: Vec<String>,
    t_current: Time,
    transform_context: TransformContext,
) -> InputState<'_> {
    let mut state = Vec::new();

    input_names.into_iter().for_each(|name| {
        let ts = collection
            .get_by_name(name.as_str())
            .unwrap_or_else(|| panic!("No timeseries with variable='{}'", name));
        state.push(ts);
    });

    InputState::build_with_transforms(state, t_current, transform_context)
}

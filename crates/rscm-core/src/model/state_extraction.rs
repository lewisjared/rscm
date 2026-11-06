//! State extraction functions for model components.

use std::collections::HashMap;

use crate::component::{InputState, RequirementDefinition, RequirementType};
use crate::state::{ReadTransformInfo, TransformContext, VariableSource};
use crate::timeseries::Time;
use crate::timeseries_collection::TimeseriesCollection;

/// Extract the input state for solving one component on its own at `t_current`.
///
/// `inputs` are the component's input definitions, as `Component::inputs` returns them.
/// A state input reads as the component's own state and every other input as exogenous,
/// each with the point or average type its definition resolves to.
pub fn extract_state(
    collection: &TimeseriesCollection,
    inputs: Vec<RequirementDefinition>,
    t_current: Time,
) -> InputState<'_> {
    let mut state = Vec::new();
    let mut read_transforms = HashMap::new();

    for input in inputs {
        let ts = collection
            .get_by_name(input.name.as_str())
            .unwrap_or_else(|| panic!("No timeseries with variable='{}'", input.name));
        state.push(ts);
        let source = if input.requirement_type == RequirementType::State {
            VariableSource::OwnState
        } else {
            VariableSource::Exogenous
        };
        let transform = ReadTransformInfo::new(input.grid_type, None)
            .with_source(source)
            .with_timeseries_type(input.resolved_timeseries_type());
        read_transforms.insert(input.name, transform);
    }

    InputState::build_with_transforms(state, t_current, TransformContext { read_transforms })
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

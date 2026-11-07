//! State extraction functions for model components.

use std::collections::HashMap;

use crate::component::{InputState, VariableDefinition, VariableRole};
use crate::state::{ReadTransformInfo, TransformContext, VariableSource};
use crate::timeseries::Time;
use crate::timeseries_collection::TimeseriesCollection;

/// Extract the input state for solving one component on its own at `t_current`.
///
/// `inputs` are the component's input definitions, as `Component::inputs` returns them.
/// A state input reads as the component's own state and every other input as prescribed,
/// each with the point or average type its definition resolves to.
pub fn extract_state(
    collection: &TimeseriesCollection,
    inputs: Vec<VariableDefinition>,
    t_current: Time,
) -> InputState<'_> {
    let mut state = Vec::new();
    let mut read_transforms = HashMap::new();

    for input in inputs {
        let ts = collection
            .get_by_name(input.name.as_str())
            .unwrap_or_else(|| panic!("No timeseries with variable='{}'", input.name));
        state.push(ts);
        let source = if input.role == VariableRole::State {
            VariableSource::OwnState
        } else {
            VariableSource::Prescribed
        };
        let transform = ReadTransformInfo::new(ts.data.grid(), None)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
    use crate::spatial::FourBoxGrid;
    use crate::timeseries::{GridTimeseries, TimeAxis};
    use numpy::array;
    use numpy::ndarray::Array2;
    use std::sync::Arc;

    #[test]
    fn scalar_input_aggregates_a_four_box_series() {
        let values =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let mut collection = TimeseriesCollection::new();
        collection.add_four_box_timeseries(
            "Forcing".to_string(),
            GridTimeseries::new(
                values,
                Arc::new(TimeAxis::from_values(array![2000.0, 2001.0])),
                FourBoxGrid::magicc_standard(),
                "W/m^2".to_string(),
                InterpolationStrategy::from(PreviousStrategy::new(true)),
            ),
        );

        let inputs = vec![VariableDefinition::scalar_input("Forcing", "W/m^2")];
        let state = extract_state(&collection, inputs, 2000.0);

        // A prescribed point reads the step's end, averaged with equal weights.
        assert_eq!(state.get_scalar_window("Forcing").get(), 6.5);
    }
}

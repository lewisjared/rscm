//! State extraction functions for model components.

use crate::component::{InputState, VariableDefinition};
use crate::errors::RSCMResult;
use crate::timeseries::Time;
use crate::timeseries_collection::TimeseriesCollection;

use super::plan::resolve_standalone;

/// Extract the input state for solving one component on its own at `t_current`.
///
/// `definitions` are the component's, and only its inputs and states are read.
/// A state input reads as the component's own state and every other input as prescribed,
/// converted to its declared unit and grid.
pub fn extract_state<'a>(
    collection: &'a TimeseriesCollection,
    definitions: &[VariableDefinition],
    t_current: Time,
) -> RSCMResult<InputState<'a>> {
    let step = resolve_standalone("Component", collection, definitions)?;
    Ok(InputState::from_step(collection, &step, t_current))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::VariableRole;
    use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
    use crate::spatial::{FourBoxGrid, ScalarGrid};
    use crate::timeseries::{GridTimeseries, TimeAxis, Timeseries};
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
        let state = extract_state(&collection, &inputs, 2000.0).unwrap();

        // A prescribed point reads the step's end, averaged with equal weights.
        assert_eq!(state.get_scalar_window("Forcing").get(), 6.5);
    }

    fn collection(name: &str, unit: &str) -> TimeseriesCollection {
        let mut collection = TimeseriesCollection::new();
        collection.add_timeseries(
            name.to_string(),
            Timeseries::new(
                array![[1.0], [2.0]],
                Arc::new(TimeAxis::from_values(array![2000.0, 2001.0])),
                ScalarGrid,
                unit.to_string(),
                InterpolationStrategy::from(PreviousStrategy::new(true)),
            ),
        );
        collection
    }

    #[test]
    fn an_input_reads_in_its_declared_unit() {
        let collection = collection("Forcing", "W/m^2");
        let inputs = vec![VariableDefinition::scalar_input("Forcing", "mW/m^2")];
        let state = extract_state(&collection, &inputs, 2000.0).unwrap();

        assert_eq!(state.get_scalar_window("Forcing").get(), 2000.0);
    }

    #[test]
    fn a_state_reads_its_own_previous_value() {
        let collection = collection("Heat", "J");
        let inputs = vec![VariableDefinition::new("Heat", "J", VariableRole::State)];
        let state = extract_state(&collection, &inputs, 2000.0).unwrap();

        assert_eq!(state.get_scalar_window("Heat").get(), 1.0);
    }

    #[test]
    fn an_incompatible_unit_or_a_finer_grid_fails() {
        let collection = collection("Forcing", "W/m^2");
        for input in [
            VariableDefinition::scalar_input("Forcing", "K"),
            VariableDefinition::four_box_input("Forcing", "W/m^2"),
        ] {
            assert!(extract_state(&collection, &[input], 2000.0).is_err());
        }
    }
}

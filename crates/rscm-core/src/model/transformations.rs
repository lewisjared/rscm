//! Grid transformation functions for model execution.

use crate::component::GridType;
use crate::errors::{RSCMError, RSCMResult};
use crate::spatial::{FourBoxGrid, FourBoxRegion, HemisphericGrid};
use crate::state::{HemisphericSlice, StateValue};

/// Aggregate a StateValue from a finer grid to a coarser grid.
///
/// This function performs write-side grid transformation, aggregating component output
/// at a finer resolution to the schema's declared coarser resolution before storage.
///
/// # Arguments
///
/// * `value` - The StateValue to aggregate
/// * `source_grid` - The grid type the component produced (finer resolution)
/// * `target_grid` - The grid type the schema expects (coarser resolution)
/// * `weights` - Optional custom weights for aggregation; uses default grid weights if None
///
/// # Supported transformations
///
/// * FourBox -> Scalar: weighted average of all 4 regions
/// * FourBox -> Hemispheric: weighted average within each hemisphere
/// * Hemispheric -> Scalar: weighted average of 2 hemispheres
///
/// # Errors
///
/// Returns an error if:
/// * The source grid is coarser than the target (disaggregation not supported)
/// * The StateValue variant doesn't match the declared source_grid
pub(crate) fn aggregate_state_value(
    value: &StateValue,
    source_grid: GridType,
    target_grid: GridType,
    weights: Option<&Vec<f64>>,
) -> RSCMResult<StateValue> {
    match (source_grid, target_grid) {
        // FourBox -> Scalar: aggregate all 4 regions to global
        (GridType::FourBox, GridType::Scalar) => {
            let slice = match value {
                StateValue::FourBox(s) => s,
                _ => panic!(
                    "StateValue type mismatch: expected FourBox but got {:?}",
                    value
                ),
            };
            let grid = match weights {
                Some(w) => {
                    let arr: [f64; 4] = w.as_slice().try_into().unwrap_or_else(|_| {
                        panic!("FourBox weights must have 4 elements, got {}", w.len())
                    });
                    FourBoxGrid::with_weights(arr)
                }
                None => FourBoxGrid::magicc_standard(),
            };
            Ok(StateValue::Scalar(slice.aggregate_global(&grid)))
        }

        // FourBox -> Hemispheric: aggregate by hemisphere
        (GridType::FourBox, GridType::Hemispheric) => {
            let slice = match value {
                StateValue::FourBox(s) => s,
                _ => panic!(
                    "StateValue type mismatch: expected FourBox but got {:?}",
                    value
                ),
            };
            let grid_weights = match weights {
                Some(w) => {
                    let arr: [f64; 4] = w.as_slice().try_into().unwrap_or_else(|_| {
                        panic!("FourBox weights must have 4 elements, got {}", w.len())
                    });
                    arr
                }
                None => [0.25, 0.25, 0.25, 0.25],
            };

            // Aggregate by hemisphere using weighted averages
            let values = slice.as_array();
            let no = FourBoxRegion::NorthernOcean as usize;
            let nl = FourBoxRegion::NorthernLand as usize;
            let so = FourBoxRegion::SouthernOcean as usize;
            let sl = FourBoxRegion::SouthernLand as usize;

            let northern = (values[no] * grid_weights[no] + values[nl] * grid_weights[nl])
                / (grid_weights[no] + grid_weights[nl]);
            let southern = (values[so] * grid_weights[so] + values[sl] * grid_weights[sl])
                / (grid_weights[so] + grid_weights[sl]);

            Ok(StateValue::Hemispheric(HemisphericSlice::from([
                northern, southern,
            ])))
        }

        // Hemispheric -> Scalar: aggregate both hemispheres to global
        (GridType::Hemispheric, GridType::Scalar) => {
            let slice = match value {
                StateValue::Hemispheric(s) => s,
                _ => panic!(
                    "StateValue type mismatch: expected Hemispheric but got {:?}",
                    value
                ),
            };
            let grid = match weights {
                Some(w) => {
                    let arr: [f64; 2] = w.as_slice().try_into().unwrap_or_else(|_| {
                        panic!("Hemispheric weights must have 2 elements, got {}", w.len())
                    });
                    HemisphericGrid::with_weights(arr)
                }
                None => HemisphericGrid::default(),
            };
            Ok(StateValue::Scalar(slice.aggregate_global(&grid)))
        }

        // Same grid type: no transformation needed (identity)
        (s, t) if s == t => Ok(value.clone()),

        // Any other combination is not supported (disaggregation)
        _ => Err(RSCMError::GridTransformationNotSupported {
            source_grid: format!("{:?}", source_grid),
            target_grid: format!("{:?}", target_grid),
            variable: "unknown".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::FourBoxSlice;
    use is_close::is_close;

    #[test]
    fn test_fourbox_to_scalar_default_weights() {
        // With equal weights (default MAGICC standard), result is mean
        let fourbox = StateValue::FourBox(FourBoxSlice::from([10.0, 20.0, 30.0, 40.0]));

        let result =
            aggregate_state_value(&fourbox, GridType::FourBox, GridType::Scalar, None).unwrap();

        match result {
            StateValue::Scalar(v) => {
                // Default weights are equal [0.25, 0.25, 0.25, 0.25]
                // 10*0.25 + 20*0.25 + 30*0.25 + 40*0.25 = 25.0
                assert!(is_close!(v, 25.0), "Expected 25.0, got {}", v);
            }
            _ => panic!("Expected Scalar, got {:?}", result),
        }
    }

    #[test]
    fn test_fourbox_to_scalar_custom_weights() {
        let fourbox = StateValue::FourBox(FourBoxSlice::from([10.0, 20.0, 30.0, 40.0]));
        let weights = vec![0.36, 0.14, 0.36, 0.14]; // Ocean-biased weights

        let result = aggregate_state_value(
            &fourbox,
            GridType::FourBox,
            GridType::Scalar,
            Some(&weights),
        )
        .unwrap();

        match result {
            StateValue::Scalar(v) => {
                // 10*0.36 + 20*0.14 + 30*0.36 + 40*0.14 = 3.6 + 2.8 + 10.8 + 5.6 = 22.8
                assert!(is_close!(v, 22.8), "Expected 22.8, got {}", v);
            }
            _ => panic!("Expected Scalar, got {:?}", result),
        }
    }

    #[test]
    fn test_fourbox_to_hemispheric_default_weights() {
        let fourbox = StateValue::FourBox(FourBoxSlice::from([10.0, 20.0, 30.0, 40.0]));

        let result =
            aggregate_state_value(&fourbox, GridType::FourBox, GridType::Hemispheric, None)
                .unwrap();

        match result {
            StateValue::Hemispheric(slice) => {
                // With equal weights [0.25, 0.25, 0.25, 0.25]:
                // Northern = (10*0.25 + 20*0.25) / (0.25 + 0.25) = 7.5 / 0.5 = 15.0
                // Southern = (30*0.25 + 40*0.25) / (0.25 + 0.25) = 17.5 / 0.5 = 35.0
                assert!(
                    is_close!(slice.as_array()[0], 15.0),
                    "Expected Northern=15.0, got {}",
                    slice.as_array()[0]
                );
                assert!(
                    is_close!(slice.as_array()[1], 35.0),
                    "Expected Southern=35.0, got {}",
                    slice.as_array()[1]
                );
            }
            _ => panic!("Expected Hemispheric, got {:?}", result),
        }
    }

    #[test]
    fn test_fourbox_to_hemispheric_custom_weights() {
        let fourbox = StateValue::FourBox(FourBoxSlice::from([10.0, 20.0, 30.0, 40.0]));
        let weights = vec![0.36, 0.14, 0.36, 0.14]; // Ocean-biased

        let result = aggregate_state_value(
            &fourbox,
            GridType::FourBox,
            GridType::Hemispheric,
            Some(&weights),
        )
        .unwrap();

        match result {
            StateValue::Hemispheric(slice) => {
                // Northern = (10*0.36 + 20*0.14) / (0.36 + 0.14) = (3.6 + 2.8) / 0.5 = 12.8
                // Southern = (30*0.36 + 40*0.14) / (0.36 + 0.14) = (10.8 + 5.6) / 0.5 = 32.8
                assert!(
                    is_close!(slice.as_array()[0], 12.8),
                    "Expected Northern=12.8, got {}",
                    slice.as_array()[0]
                );
                assert!(
                    is_close!(slice.as_array()[1], 32.8),
                    "Expected Southern=32.8, got {}",
                    slice.as_array()[1]
                );
            }
            _ => panic!("Expected Hemispheric, got {:?}", result),
        }
    }

    #[test]
    fn test_hemispheric_to_scalar_default_weights() {
        let hemispheric = StateValue::Hemispheric(HemisphericSlice::from([15.0, 35.0]));

        let result =
            aggregate_state_value(&hemispheric, GridType::Hemispheric, GridType::Scalar, None)
                .unwrap();

        match result {
            StateValue::Scalar(v) => {
                // Default weights [0.5, 0.5] -> mean
                // 15*0.5 + 35*0.5 = 25.0
                assert!(is_close!(v, 25.0), "Expected 25.0, got {}", v);
            }
            _ => panic!("Expected Scalar, got {:?}", result),
        }
    }

    #[test]
    fn test_hemispheric_to_scalar_custom_weights() {
        let hemispheric = StateValue::Hemispheric(HemisphericSlice::from([10.0, 30.0]));
        let weights = vec![0.4, 0.6]; // Southern-biased

        let result = aggregate_state_value(
            &hemispheric,
            GridType::Hemispheric,
            GridType::Scalar,
            Some(&weights),
        )
        .unwrap();

        match result {
            StateValue::Scalar(v) => {
                // 10*0.4 + 30*0.6 = 4 + 18 = 22.0
                assert!(is_close!(v, 22.0), "Expected 22.0, got {}", v);
            }
            _ => panic!("Expected Scalar, got {:?}", result),
        }
    }

    #[test]
    fn test_identity_transformation_scalar() {
        let scalar = StateValue::Scalar(42.0);

        let result =
            aggregate_state_value(&scalar, GridType::Scalar, GridType::Scalar, None).unwrap();

        match result {
            StateValue::Scalar(v) => assert_eq!(v, 42.0),
            _ => panic!("Expected Scalar, got {:?}", result),
        }
    }

    #[test]
    fn test_identity_transformation_fourbox() {
        let fourbox = StateValue::FourBox(FourBoxSlice::from([1.0, 2.0, 3.0, 4.0]));

        let result =
            aggregate_state_value(&fourbox, GridType::FourBox, GridType::FourBox, None).unwrap();

        match result {
            StateValue::FourBox(slice) => {
                assert_eq!(*slice.as_array(), [1.0, 2.0, 3.0, 4.0]);
            }
            _ => panic!("Expected FourBox, got {:?}", result),
        }
    }

    #[test]
    fn test_disaggregation_scalar_to_fourbox_rejected() {
        let scalar = StateValue::Scalar(25.0);

        let result = aggregate_state_value(&scalar, GridType::Scalar, GridType::FourBox, None);

        assert!(
            result.is_err(),
            "Disaggregation Scalar->FourBox should be rejected"
        );
    }

    #[test]
    fn test_disaggregation_scalar_to_hemispheric_rejected() {
        let scalar = StateValue::Scalar(25.0);

        let result = aggregate_state_value(&scalar, GridType::Scalar, GridType::Hemispheric, None);

        assert!(
            result.is_err(),
            "Disaggregation Scalar->Hemispheric should be rejected"
        );
    }

    #[test]
    fn test_disaggregation_hemispheric_to_fourbox_rejected() {
        let hemispheric = StateValue::Hemispheric(HemisphericSlice::from([15.0, 35.0]));

        let result =
            aggregate_state_value(&hemispheric, GridType::Hemispheric, GridType::FourBox, None);

        assert!(
            result.is_err(),
            "Disaggregation Hemispheric->FourBox should be rejected"
        );
    }
}

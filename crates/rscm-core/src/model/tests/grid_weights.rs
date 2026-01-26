//! Grid weight configuration tests.

use crate::component::GridType;
use crate::model::ModelBuilder;
use crate::schema::VariableSchema;
use crate::timeseries::TimeAxis;
use numpy::ndarray::Array;

#[test]
fn test_with_grid_weights_fourbox_valid() {
    let mut builder = ModelBuilder::new();
    builder.with_grid_weights(GridType::FourBox, vec![0.36, 0.14, 0.36, 0.14]);

    // Build and check the model has the weights
    let model = builder
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .build()
        .unwrap();

    assert_eq!(
        model.get_grid_weights(GridType::FourBox),
        Some(&vec![0.36, 0.14, 0.36, 0.14])
    );
}

#[test]
fn test_with_grid_weights_hemispheric_valid() {
    let mut builder = ModelBuilder::new();
    builder.with_grid_weights(GridType::Hemispheric, vec![0.6, 0.4]);

    let model = builder
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .build()
        .unwrap();

    assert_eq!(
        model.get_grid_weights(GridType::Hemispheric),
        Some(&vec![0.6, 0.4])
    );
}

#[test]
#[should_panic(expected = "Cannot set weights for Scalar")]
fn test_with_grid_weights_scalar_panics() {
    let mut builder = ModelBuilder::new();
    builder.with_grid_weights(GridType::Scalar, vec![1.0]);
}

#[test]
#[should_panic(expected = "Weights length")]
fn test_with_grid_weights_wrong_length_panics() {
    let mut builder = ModelBuilder::new();
    builder.with_grid_weights(GridType::FourBox, vec![0.5, 0.5]); // Wrong: 2 instead of 4
}

#[test]
#[should_panic(expected = "Weights must sum to 1.0")]
fn test_with_grid_weights_wrong_sum_panics() {
    let mut builder = ModelBuilder::new();
    builder.with_grid_weights(GridType::FourBox, vec![0.3, 0.3, 0.3, 0.3]);
    // Sum = 1.2
}

#[test]
fn test_custom_weights_applied_to_fourbox_timeseries() {
    // Custom weights: different from default [0.25, 0.25, 0.25, 0.25]
    let custom_weights = vec![0.36, 0.14, 0.36, 0.14];

    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::FourBox);

    let model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_grid_weights(GridType::FourBox, custom_weights.clone())
        .build()
        .unwrap();

    // Verify custom weights are stored in the Model
    let model_weights = model.get_grid_weights(GridType::FourBox);
    assert_eq!(model_weights, Some(&custom_weights));
}

#[test]
fn test_model_get_grid_weights_returns_none_for_unset() {
    let model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .build()
        .unwrap();

    assert!(model.get_grid_weights(GridType::FourBox).is_none());
    assert!(model.get_grid_weights(GridType::Hemispheric).is_none());
    assert!(model.get_grid_weights(GridType::Scalar).is_none());
}

#[test]
fn test_grid_weights_serialisation_roundtrip() {
    use crate::model::Model;

    let custom_weights = vec![0.36, 0.14, 0.36, 0.14];
    let schema = VariableSchema::new().variable_with_grid("Temperature", "K", GridType::FourBox);

    let model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_grid_weights(GridType::FourBox, custom_weights.clone())
        .build()
        .unwrap();

    // Serialise and deserialise
    let serialised = toml::to_string(&model).unwrap();
    let deserialised: Model = toml::from_str(&serialised).unwrap();

    // Verify weights are preserved
    assert_eq!(
        deserialised.get_grid_weights(GridType::FourBox),
        Some(&custom_weights)
    );
}

#[test]
fn test_empty_grid_weights_not_serialised() {
    let model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .build()
        .unwrap();

    let serialised = toml::to_string(&model).unwrap();

    // The grid_weights section should not appear
    assert!(
        !serialised.contains("grid_weights"),
        "Empty grid_weights should not be serialised: {}",
        serialised
    );
}

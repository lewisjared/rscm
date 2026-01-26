//! Model runner trait for calibration.
//!
//! This module provides the interface between calibration algorithms and climate models.
//! The ModelRunner trait defines how to run models with parameter vectors and extract
//! outputs for comparison with observations.

use crate::likelihood::{ModelOutput, VariableOutput};
use crate::{Error, Result};
use rayon::prelude::*;
use rscm_core::model::Model;
use rscm_core::timeseries_collection::TimeseriesData;

/// Trait for running models with parameter vectors.
///
/// This trait provides the interface between calibration algorithms and climate models.
/// Implementations handle:
/// - Mapping indexed parameter vectors to model parameters
/// - Running the model
/// - Extracting relevant outputs
/// - Handling model failures
///
/// # Design
///
/// Parameters are indexed (`Vec<f64>`) rather than named (`HashMap<String, f64>`) for performance.
/// The parameter order is defined by `param_names()` and must remain consistent across all
/// method calls.
///
/// # Example
///
/// ```ignore
/// let runner = MyModelRunner::new(/* ... */);
/// let param_names = runner.param_names();
/// // param_names = ["climate_sensitivity", "ocean_diffusivity"]
///
/// let params = vec![3.0, 1200.0];  // Must match order of param_names
/// let output = runner.run(&params)?;
/// ```
pub trait ModelRunner {
    /// Get the names of parameters in the order they appear in parameter vectors.
    ///
    /// This defines the indexed parameter order used by `run()` and `run_batch()`.
    /// The order must remain consistent for the lifetime of the ModelRunner.
    fn param_names(&self) -> &[String];

    /// Run the model with the given parameter vector.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter values indexed by `param_names()` order
    ///
    /// # Returns
    ///
    /// Model output for the requested variables, or an error if the model fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Parameter vector length doesn't match `param_names().len()`
    /// - Model construction fails (invalid parameters)
    /// - Model execution fails (solver divergence, NaN, etc.)
    /// - Output extraction fails (missing variables)
    fn run(&self, params: &[f64]) -> Result<ModelOutput>;

    /// Run the model for multiple parameter vectors.
    ///
    /// Default implementation calls `run()` sequentially. Implementations can override
    /// this to use parallel execution via rayon.
    ///
    /// # Arguments
    ///
    /// * `param_sets` - Multiple parameter vectors to evaluate
    ///
    /// # Returns
    ///
    /// Vector of results in the same order as `param_sets`. Failed runs are returned
    /// as `Err` values (not filtered out).
    ///
    /// # Errors
    ///
    /// Individual model failures are returned as `Err` items in the result vector.
    /// The method itself only fails if there's a fundamental problem (e.g., thread pool error).
    fn run_batch(&self, param_sets: &[Vec<f64>]) -> Vec<Result<ModelOutput>> {
        param_sets.iter().map(|params| self.run(params)).collect()
    }
}

/// Default model runner using a factory closure.
///
/// This implementation creates a fresh model for each parameter set using a factory closure,
/// runs the model, and extracts specified output variables.
///
/// # Type Parameters
///
/// * `F` - Factory function type: `Fn(&[f64]) -> Result<Model>`
///
/// # Example
///
/// ```ignore
/// use rscm_calibrate::DefaultModelRunner;
/// use rscm_core::model::ModelBuilder;
///
/// let runner = DefaultModelRunner::new(
///     vec!["climate_sensitivity".to_string(), "ocean_diffusivity".to_string()],
///     vec!["Temperature|Global".to_string(), "OHC".to_string()],
///     |params| {
///         let climate_sensitivity = params[0];
///         let ocean_diffusivity = params[1];
///
///         let model = ModelBuilder::new()
///             // ... configure model with parameters ...
///             .build()?;
///         Ok(model)
///     }
/// );
/// ```
pub struct DefaultModelRunner<F>
where
    F: Fn(&[f64]) -> Result<Model>,
{
    /// Parameter names in indexed order
    param_names: Vec<String>,
    /// Output variable names to extract
    output_variables: Vec<String>,
    /// Factory function to create models from parameter vectors
    factory: F,
}

impl<F> DefaultModelRunner<F>
where
    F: Fn(&[f64]) -> Result<Model>,
{
    /// Create a new default model runner.
    ///
    /// # Arguments
    ///
    /// * `param_names` - Names of parameters in the order they appear in parameter vectors
    /// * `output_variables` - Names of variables to extract from model output
    /// * `factory` - Function to create a model from a parameter vector
    ///
    /// # Example
    ///
    /// ```ignore
    /// let runner = DefaultModelRunner::new(
    ///     vec!["param1".to_string(), "param2".to_string()],
    ///     vec!["Temperature|Global".to_string()],
    ///     |params| {
    ///         // Create model with params[0] and params[1]
    ///         Ok(model)
    ///     }
    /// );
    /// ```
    pub fn new(param_names: Vec<String>, output_variables: Vec<String>, factory: F) -> Self {
        Self {
            param_names,
            output_variables,
            factory,
        }
    }

    /// Extract model outputs for the specified variables.
    fn extract_outputs(&self, model: &Model) -> Result<ModelOutput> {
        let mut output = ModelOutput::new();
        let collection = model.timeseries();

        for var_name in &self.output_variables {
            let ts_data = collection.get_data(var_name).ok_or_else(|| {
                Error::ModelError(format!("Model output missing variable: {}", var_name))
            })?;

            let mut var_output = VariableOutput::new(var_name);

            // Extract scalar values at each time
            // We only extract scalar values for now (grid support can be added later)
            let timeseries = match ts_data {
                TimeseriesData::Scalar(ts) => ts,
                TimeseriesData::FourBox(_) => {
                    return Err(Error::ModelError(format!(
                        "Grid variables not yet supported: {}",
                        var_name
                    )))
                }
                TimeseriesData::Hemispheric(_) => {
                    return Err(Error::ModelError(format!(
                        "Grid variables not yet supported: {}",
                        var_name
                    )))
                }
            };

            // Extract all time points
            // The timeseries may contain NaN values for uncomputed timesteps
            // We only extract non-NaN values
            let time_axis = timeseries.time_axis();
            for i in 0..timeseries.len() {
                let time = time_axis
                    .at(i)
                    .ok_or_else(|| Error::ModelError(format!("Time index {} out of bounds", i)))?;
                let value = timeseries
                    .at_index(i, 0)
                    .ok_or_else(|| Error::ModelError(format!("Value index {} out of bounds", i)))?;

                // Skip NaN values (uncomputed timesteps)
                if !value.is_nan() {
                    var_output.add(time, value);
                }
            }

            output.add_variable(var_output);
        }

        Ok(output)
    }
}

impl<F> ModelRunner for DefaultModelRunner<F>
where
    F: Fn(&[f64]) -> Result<Model> + Sync,
{
    fn param_names(&self) -> &[String] {
        &self.param_names
    }

    fn run(&self, params: &[f64]) -> Result<ModelOutput> {
        // Validate parameter count
        if params.len() != self.param_names.len() {
            return Err(Error::ModelError(format!(
                "Expected {} parameters, got {}",
                self.param_names.len(),
                params.len()
            )));
        }

        // Create model using factory
        let mut model = (self.factory)(params)
            .map_err(|e| Error::ModelError(format!("Model construction failed: {}", e)))?;

        // Run model
        model.run();

        // Check if model finished successfully
        if !model.finished() {
            return Err(Error::ModelError(
                "Model did not complete all timesteps".to_string(),
            ));
        }

        // Extract outputs
        self.extract_outputs(&model)
    }

    /// Run models in parallel using rayon.
    ///
    /// This implementation leverages rayon's parallel iterators to execute
    /// multiple model runs concurrently, which can significantly speed up
    /// calibration workflows.
    ///
    /// # Thread Safety
    ///
    /// The factory function must be `Sync` to allow safe parallel execution.
    /// Each thread constructs its own model instance, so there's no shared state.
    fn run_batch(&self, param_sets: &[Vec<f64>]) -> Vec<Result<ModelOutput>> {
        param_sets
            .par_iter()
            .map(|params| self.run(params))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::likelihood::VariableOutput;

    /// Simple test implementation that squares parameters
    struct TestRunner {
        param_names: Vec<String>,
        output_variables: Vec<String>,
    }

    impl TestRunner {
        fn new() -> Self {
            Self {
                param_names: vec!["param1".to_string(), "param2".to_string()],
                output_variables: vec!["output1".to_string()],
            }
        }
    }

    impl ModelRunner for TestRunner {
        fn param_names(&self) -> &[String] {
            &self.param_names
        }

        fn run(&self, params: &[f64]) -> Result<ModelOutput> {
            if params.len() != self.param_names.len() {
                return Err(Error::ModelError(format!(
                    "Expected {} parameters, got {}",
                    self.param_names.len(),
                    params.len()
                )));
            }

            // Simple test: output is sum of squared parameters
            let value = params.iter().map(|&p| p * p).sum::<f64>();

            let mut output = ModelOutput::new();
            let mut var = VariableOutput::new(&self.output_variables[0]);
            var.add(2020.0, value);
            output.add_variable(var);

            Ok(output)
        }
    }

    #[test]
    fn test_param_names() {
        let runner = TestRunner::new();
        assert_eq!(runner.param_names(), &["param1", "param2"]);
    }

    #[test]
    fn test_run_success() {
        let runner = TestRunner::new();
        let params = vec![3.0, 4.0]; // 3^2 + 4^2 = 9 + 16 = 25
        let output = runner.run(&params).unwrap();

        let var = output.get_variable("output1").unwrap();
        assert_eq!(var.get(2020.0), Some(25.0));
    }

    #[test]
    fn test_run_wrong_param_count() {
        let runner = TestRunner::new();
        let params = vec![3.0]; // Wrong length
        let result = runner.run(&params);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ModelError(_)));
    }

    #[test]
    fn test_run_batch_default_implementation() {
        let runner = TestRunner::new();
        let param_sets = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let results = runner.run_batch(&param_sets);

        assert_eq!(results.len(), 3);

        // Check first result: 1^2 + 2^2 = 5
        let output1 = results[0].as_ref().unwrap();
        assert_eq!(
            output1.get_variable("output1").unwrap().get(2020.0),
            Some(5.0)
        );

        // Check second result: 3^2 + 4^2 = 25
        let output2 = results[1].as_ref().unwrap();
        assert_eq!(
            output2.get_variable("output1").unwrap().get(2020.0),
            Some(25.0)
        );

        // Check third result: 5^2 + 6^2 = 61
        let output3 = results[2].as_ref().unwrap();
        assert_eq!(
            output3.get_variable("output1").unwrap().get(2020.0),
            Some(61.0)
        );
    }

    #[test]
    fn test_run_batch_with_failures() {
        let runner = TestRunner::new();
        let param_sets = vec![
            vec![1.0, 2.0], // Valid
            vec![3.0],      // Invalid - wrong length
            vec![4.0, 5.0], // Valid
        ];

        let results = runner.run_batch(&param_sets);

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert!(results[2].is_ok());
    }

    // Tests for DefaultModelRunner with simple mock Model
    use rscm_core::component::{Component, InputState, OutputState, RequirementDefinition};
    use rscm_core::errors::RSCMResult;
    use rscm_core::model::ModelBuilder;
    use rscm_core::timeseries::{Time, TimeAxis};
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    /// Simple test component that outputs a constant value based on a parameter
    ///
    /// This component has no inputs and produces a constant output value at each timestep.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ConstantComponent {
        value: f64,
    }

    #[typetag::serde]
    impl Component for ConstantComponent {
        fn definitions(&self) -> Vec<RequirementDefinition> {
            vec![RequirementDefinition::scalar_output("TestOutput", "units")]
        }

        fn solve(
            &self,
            _t_current: Time,
            _t_next: Time,
            _input_state: &InputState,
        ) -> RSCMResult<OutputState> {
            let mut output = OutputState::new();
            output.insert("TestOutput".to_string(), self.value.into());
            Ok(output)
        }
    }

    #[test]
    fn test_default_runner_param_names() {
        let runner = DefaultModelRunner::new(
            vec!["param1".to_string(), "param2".to_string()],
            vec!["TestOutput".to_string()],
            |_params| {
                Err(Error::ModelError(
                    "Should not be called in this test".to_string(),
                ))
            },
        );

        assert_eq!(runner.param_names(), &["param1", "param2"]);
    }

    #[test]
    fn test_default_runner_simple_model() {
        let runner = DefaultModelRunner::new(
            vec!["value".to_string()],
            vec!["TestOutput".to_string()],
            |params| {
                let value = params[0];
                let component = ConstantComponent { value };

                let time_axis = TimeAxis::from_values(ndarray::Array::range(2000.0, 2010.0, 1.0));

                let model = ModelBuilder::new()
                    .with_time_axis(time_axis)
                    .with_component(Arc::new(component))
                    .build()
                    .map_err(|e| Error::ModelError(format!("Build failed: {}", e)))?;

                Ok(model)
            },
        );

        // Run with parameter value = 42.0
        let output = runner.run(&[42.0]).unwrap();

        // Check output
        let var = output.get_variable("TestOutput").unwrap();
        assert_eq!(var.name, "TestOutput");

        // Model writes to time_index + 1, so time[0] (2000.0) will be NaN
        // time[1] onwards should have value 42.0
        assert_eq!(var.get(2000.0), None); // Index 0 is NaN, so skipped
        assert_eq!(var.get(2001.0), Some(42.0)); // First computed value
        assert_eq!(var.get(2005.0), Some(42.0));
        assert_eq!(var.get(2009.0), Some(42.0)); // Last computed value
    }

    #[test]
    fn test_default_runner_wrong_param_count() {
        let runner = DefaultModelRunner::new(
            vec!["param1".to_string(), "param2".to_string()],
            vec!["TestOutput".to_string()],
            |_params| Err(Error::ModelError("Should not be called".to_string())),
        );

        let result = runner.run(&[1.0]); // Wrong length
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ModelError(_)));
    }

    #[test]
    fn test_default_runner_model_construction_failure() {
        let runner = DefaultModelRunner::new(
            vec!["param1".to_string()],
            vec!["TestOutput".to_string()],
            |params| {
                if params[0] < 0.0 {
                    Err(Error::ModelError("Negative values not allowed".to_string()))
                } else {
                    // Build a simple model
                    let component = ConstantComponent { value: params[0] };
                    let time_axis =
                        TimeAxis::from_values(ndarray::Array::range(2000.0, 2005.0, 1.0));
                    let model = ModelBuilder::new()
                        .with_time_axis(time_axis)
                        .with_component(Arc::new(component))
                        .build()
                        .map_err(|e| Error::ModelError(format!("Build failed: {}", e)))?;
                    Ok(model)
                }
            },
        );

        // Valid parameter
        let result1 = runner.run(&[5.0]);
        assert!(result1.is_ok());

        // Invalid parameter (should trigger factory error)
        let result2 = runner.run(&[-5.0]);
        assert!(result2.is_err());
    }

    #[test]
    fn test_default_runner_missing_output_variable() {
        let runner = DefaultModelRunner::new(
            vec!["value".to_string()],
            vec!["NonExistentOutput".to_string()],
            |params| {
                let component = ConstantComponent { value: params[0] };
                let time_axis = TimeAxis::from_values(ndarray::Array::range(2000.0, 2005.0, 1.0));
                let model = ModelBuilder::new()
                    .with_time_axis(time_axis)
                    .with_component(Arc::new(component))
                    .build()
                    .map_err(|e| Error::ModelError(format!("Build failed: {}", e)))?;
                Ok(model)
            },
        );

        let result = runner.run(&[42.0]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::ModelError(_)));
        assert!(err.to_string().contains("missing variable"));
    }

    #[test]
    fn test_default_runner_multiple_output_variables() {
        /// Component with two outputs
        #[derive(Debug, Serialize, Deserialize)]
        struct DualOutputComponent {
            value1: f64,
            value2: f64,
        }

        #[typetag::serde]
        impl Component for DualOutputComponent {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![
                    RequirementDefinition::scalar_output("Output1", "units"),
                    RequirementDefinition::scalar_output("Output2", "units"),
                ]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                _input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let mut output = OutputState::new();
                output.insert("Output1".to_string(), self.value1.into());
                output.insert("Output2".to_string(), self.value2.into());
                Ok(output)
            }
        }

        let runner = DefaultModelRunner::new(
            vec!["param1".to_string(), "param2".to_string()],
            vec!["Output1".to_string(), "Output2".to_string()],
            |params| {
                let component = DualOutputComponent {
                    value1: params[0],
                    value2: params[1],
                };
                let time_axis = TimeAxis::from_values(ndarray::Array::range(2000.0, 2005.0, 1.0));
                let model = ModelBuilder::new()
                    .with_time_axis(time_axis)
                    .with_component(Arc::new(component))
                    .build()
                    .map_err(|e| Error::ModelError(format!("Build failed: {}", e)))?;
                Ok(model)
            },
        );

        let output = runner.run(&[10.0, 20.0]).unwrap();

        assert_eq!(output.variables.len(), 2);
        // time[0] (2000.0) will be NaN, first value is at time[1] (2001.0)
        assert_eq!(
            output.get_variable("Output1").unwrap().get(2001.0),
            Some(10.0)
        );
        assert_eq!(
            output.get_variable("Output2").unwrap().get(2001.0),
            Some(20.0)
        );
    }

    #[test]
    fn test_default_runner_parallel_batch() {
        let runner = DefaultModelRunner::new(
            vec!["value".to_string()],
            vec!["TestOutput".to_string()],
            |params| {
                let component = ConstantComponent { value: params[0] };
                let time_axis = TimeAxis::from_values(ndarray::Array::range(2000.0, 2005.0, 1.0));
                let model = ModelBuilder::new()
                    .with_time_axis(time_axis)
                    .with_component(Arc::new(component))
                    .build()
                    .map_err(|e| Error::ModelError(format!("Build failed: {}", e)))?;
                Ok(model)
            },
        );

        // Run multiple parameter sets in parallel
        let param_sets = vec![vec![10.0], vec![20.0], vec![30.0], vec![40.0], vec![50.0]];

        let results = runner.run_batch(&param_sets);

        // All should succeed
        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.is_ok());
        }

        // Check values
        assert_eq!(
            results[0]
                .as_ref()
                .unwrap()
                .get_variable("TestOutput")
                .unwrap()
                .get(2001.0),
            Some(10.0)
        );
        assert_eq!(
            results[1]
                .as_ref()
                .unwrap()
                .get_variable("TestOutput")
                .unwrap()
                .get(2001.0),
            Some(20.0)
        );
        assert_eq!(
            results[4]
                .as_ref()
                .unwrap()
                .get_variable("TestOutput")
                .unwrap()
                .get(2001.0),
            Some(50.0)
        );
    }
}

use crate::likelihood::{ModelOutput, VariableOutput};
use crate::model_runner::ModelRunner;
use crate::{Error, Result};

/// Fails every run with "solver diverged", or only above `threshold` when one is set.
pub(crate) struct FailingModel {
    parameter_names: Vec<String>,
    threshold: Option<f64>,
}

impl FailingModel {
    pub(crate) fn new(threshold: Option<f64>) -> Self {
        Self {
            parameter_names: vec!["x".to_string()],
            threshold,
        }
    }
}

impl ModelRunner for FailingModel {
    fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    fn run(&self, parameters: &[f64]) -> Result<ModelOutput> {
        if self.threshold.is_none_or(|t| parameters[0] > t) {
            return Err(Error::ModelError("solver diverged".to_string()));
        }
        let mut output = ModelOutput::new();
        let mut var = VariableOutput::new("value");
        var.add(0.0, parameters[0]);
        output.add_variable(var);
        Ok(output)
    }
}

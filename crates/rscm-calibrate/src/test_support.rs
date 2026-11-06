use crate::likelihood::{ModelOutput, VariableOutput};
use crate::model_runner::ModelRunner;
use crate::{Error, Result};

/// Fails every run with "solver diverged", or only above `threshold` when one is set.
pub(crate) struct FailingModel {
    param_names: Vec<String>,
    threshold: Option<f64>,
}

impl FailingModel {
    pub(crate) fn new(threshold: Option<f64>) -> Self {
        Self {
            param_names: vec!["x".to_string()],
            threshold,
        }
    }
}

impl ModelRunner for FailingModel {
    fn param_names(&self) -> &[String] {
        &self.param_names
    }

    fn run(&self, params: &[f64]) -> Result<ModelOutput> {
        if self.threshold.is_none_or(|t| params[0] > t) {
            return Err(Error::ModelError("solver diverged".to_string()));
        }
        let mut output = ModelOutput::new();
        let mut var = VariableOutput::new("value");
        var.add(0.0, params[0]);
        output.add_variable(var);
        Ok(output)
    }
}

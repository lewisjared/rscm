//! A component with no variables, for tests.

use crate::component::{Component, InputState, OutputState, VariableDefinition};
use crate::errors::RSCMResult;
use crate::timeseries::Time;
use serde::{Deserialize, Serialize};

/// A component that declares and does nothing.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct NullComponent {}

#[typetag::serde]
impl Component for NullComponent {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        Ok(OutputState::new())
    }
}

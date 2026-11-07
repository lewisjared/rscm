//! Null component for graph initialisation.

use crate::component::{Component, InputState, OutputState, VariableDefinition};
use crate::errors::RscmResult;
use crate::timeseries::Time;
use serde::{Deserialize, Serialize};

/// A null component that does nothing.
///
/// Used as an initial component to ensure that the model graph is connected.
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
    ) -> RscmResult<OutputState> {
        Ok(OutputState::new())
    }
}

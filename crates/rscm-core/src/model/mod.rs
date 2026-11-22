//! A model consists of a series of coupled components which are solved together.
//!
//! The model orchestrates the passing of state between different components.
//! Each component is solved for a given step in an order determined by their
//! dependencies.
//! Once all components and state is solved for, the model will move to the next step.
//! The state from previous steps is preserved as it is useful as output or in the case where
//! a component needs previous values.
//!
//! The model also holds all of the prescribed variables required by the model.
//! The required variables are identified when building the model.
//! If a required prescribed variable isn't provided, then the build step will fail.

mod builder;
mod coupling;
#[cfg(test)]
mod null_component;
mod parallel;
mod plan;
mod record;
mod runtime;
mod state_extraction;
mod storage;
mod transformations;
mod types;
mod validation;

#[cfg(test)]
mod tests;

// Public re-exports
pub(crate) use builder::RegionValues;
pub use builder::{check_grid_weights, ModelBuilder, PrescribedInput};
pub use coupling::{ComponentCoupling, Coupling, CouplingEdge};
pub use parallel::{build_and_run, build_models, run_each, run_models, Pool};
pub use plan::{
    resolve_standalone, Conversion, GridConversion, Read, Step, StepPlan, UnitConversion, Write,
};
pub use record::{Branch, Record, RECORD_FORMAT};
pub use runtime::Model;
pub use state_extraction::extract_state;
pub use types::TransformDirection;

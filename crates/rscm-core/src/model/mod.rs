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
mod null_component;
mod parallel;
mod runtime;
mod state_extraction;
mod transformations;
mod types;
mod validation;

#[cfg(test)]
mod tests;

// Public re-exports
pub use builder::{ModelBuilder, PrescribedInput};
pub use coupling::{ComponentCoupling, ConversionInfo, Coupling, CouplingEdge, TransformInfo};
pub use parallel::{build_models, run_models};
pub use runtime::Model;
pub use state_extraction::{extract_state, extract_state_with_transforms};
pub use types::{GraphEdge, RequiredTransformation, TransformDirection, UnitConversionInfo};

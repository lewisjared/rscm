//! Python bindings for the FaIR components.

use crate::{
    AerosolCloud, AerosolCloudParameters, CH4Lifetime, CH4LifetimeParameters, Eesc, EescParameters,
    EnergyBalance, EnergyBalanceParameters, ForcingAggregation, ForcingAggregationParameters,
    GasCycle, GasCycleParameters, GhgForcing, GhgForcingParameters, LinearForcing,
    LinearForcingParameters,
};
use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};
use rscm_core::python::PyRustComponent;
use rscm_core::{create_component_builder, create_fallible_component_builder};

create_fallible_component_builder!(EnergyBalanceBuilder, EnergyBalance, EnergyBalanceParameters);
create_fallible_component_builder!(GasCycleBuilder, GasCycle, GasCycleParameters);
create_fallible_component_builder!(AerosolCloudBuilder, AerosolCloud, AerosolCloudParameters);
create_component_builder!(CH4LifetimeBuilder, CH4Lifetime, CH4LifetimeParameters);
create_component_builder!(EescBuilder, Eesc, EescParameters);
create_component_builder!(GhgForcingBuilder, GhgForcing, GhgForcingParameters);
create_component_builder!(LinearForcingBuilder, LinearForcing, LinearForcingParameters);
create_component_builder!(
    ForcingAggregationBuilder,
    ForcingAggregation,
    ForcingAggregationParameters
);

#[pymodule]
pub fn fair(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CH4LifetimeBuilder>()?;
    m.add_class::<GasCycleBuilder>()?;
    m.add_class::<EescBuilder>()?;
    m.add_class::<GhgForcingBuilder>()?;
    m.add_class::<LinearForcingBuilder>()?;
    m.add_class::<AerosolCloudBuilder>()?;
    m.add_class::<ForcingAggregationBuilder>()?;
    m.add_class::<EnergyBalanceBuilder>()?;
    Ok(())
}

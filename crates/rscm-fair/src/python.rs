//! Python bindings for the FaIR components.

use crate::{
    AerosolCloud, AerosolCloudParameters, CH4Lifetime, CH4LifetimeParameters, EESCParameters,
    EnergyBalance, EnergyBalanceParameters, ForcingAggregation, ForcingAggregationParameters,
    GHGForcing, GHGForcingParameters, GasCycle, GasCycleParameters, LinearForcing,
    LinearForcingParameters, EESC,
};
use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};
use rscm_core::python::PyRustComponent;
use rscm_core::{create_component_builder, create_fallible_component_builder};

create_fallible_component_builder!(EnergyBalanceBuilder, EnergyBalance, EnergyBalanceParameters);
create_fallible_component_builder!(GasCycleBuilder, GasCycle, GasCycleParameters);
create_fallible_component_builder!(AerosolCloudBuilder, AerosolCloud, AerosolCloudParameters);
create_component_builder!(CH4LifetimeBuilder, CH4Lifetime, CH4LifetimeParameters);
create_component_builder!(EESCBuilder, EESC, EESCParameters);
create_component_builder!(GHGForcingBuilder, GHGForcing, GHGForcingParameters);
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
    m.add_class::<EESCBuilder>()?;
    m.add_class::<GHGForcingBuilder>()?;
    m.add_class::<LinearForcingBuilder>()?;
    m.add_class::<AerosolCloudBuilder>()?;
    m.add_class::<ForcingAggregationBuilder>()?;
    m.add_class::<EnergyBalanceBuilder>()?;
    Ok(())
}

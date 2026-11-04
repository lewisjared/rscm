//! Python bindings for the FaIR components.

use crate::{
    FairAerosolCloud, FairAerosolCloudParameters, FairCH4Lifetime, FairCH4LifetimeParameters,
    FairEesc, FairEescParameters, FairEnergyBalance, FairEnergyBalanceParameters,
    FairForcingAggregation, FairForcingAggregationParameters, FairGasCycle, FairGasCycleParameters,
    FairGhgForcing, FairGhgForcingParameters, FairLinearForcing, FairLinearForcingParameters,
};
use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};
use rscm_core::python::PyRustComponent;
use rscm_core::{create_component_builder, create_fallible_component_builder};

create_fallible_component_builder!(
    FairEnergyBalanceBuilder,
    FairEnergyBalance,
    FairEnergyBalanceParameters
);
create_fallible_component_builder!(FairGasCycleBuilder, FairGasCycle, FairGasCycleParameters);
create_component_builder!(
    FairAerosolCloudBuilder,
    FairAerosolCloud,
    FairAerosolCloudParameters
);
create_component_builder!(
    FairCH4LifetimeBuilder,
    FairCH4Lifetime,
    FairCH4LifetimeParameters
);
create_component_builder!(FairEescBuilder, FairEesc, FairEescParameters);
create_component_builder!(
    FairGhgForcingBuilder,
    FairGhgForcing,
    FairGhgForcingParameters
);
create_component_builder!(
    FairLinearForcingBuilder,
    FairLinearForcing,
    FairLinearForcingParameters
);
create_component_builder!(
    FairForcingAggregationBuilder,
    FairForcingAggregation,
    FairForcingAggregationParameters
);

#[pymodule]
pub fn fair(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FairCH4LifetimeBuilder>()?;
    m.add_class::<FairGasCycleBuilder>()?;
    m.add_class::<FairEescBuilder>()?;
    m.add_class::<FairGhgForcingBuilder>()?;
    m.add_class::<FairLinearForcingBuilder>()?;
    m.add_class::<FairAerosolCloudBuilder>()?;
    m.add_class::<FairForcingAggregationBuilder>()?;
    m.add_class::<FairEnergyBalanceBuilder>()?;
    Ok(())
}

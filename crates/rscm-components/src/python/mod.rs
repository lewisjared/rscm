use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};

use rscm_core::create_component_builder;
use rscm_core::python::PyRustComponent;

use crate::components::*;

create_component_builder!(CO2ErfBuilder, CO2Erf, CO2ErfParameters);
create_component_builder!(CarbonCycleBuilder, CarbonCycle, CarbonCycleParameters);
create_component_builder!(
    FourBoxOceanHeatUptakeBuilder,
    FourBoxOceanHeatUptake,
    FourBoxOceanHeatUptakeParameters
);

#[pymodule]
pub fn components(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CO2ErfBuilder>()?;
    m.add_class::<CarbonCycleBuilder>()?;
    m.add_class::<FourBoxOceanHeatUptakeBuilder>()?;
    Ok(())
}

use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};

use rscm_core::create_component_builder;
use rscm_core::python::PyRustComponent;

use crate::carbon::{CO2Budget, OceanCarbon, TerrestrialCarbon};
use crate::chemistry::{CH4Chemistry, HalocarbonChemistry, N2OChemistry};
use crate::climate::ClimateUDEB;
use crate::forcing::{AerosolDirect, AerosolIndirect, GhgForcing, OzoneForcing};
use crate::parameters::{
    AerosolDirectParameters, AerosolIndirectParameters, CH4ChemistryParameters,
    CO2BudgetParameters, ClimateUDEBParameters, GhgForcingParameters, HalocarbonParameters,
    N2OChemistryParameters, OceanCarbonParameters, OzoneForcingParameters,
    TerrestrialCarbonParameters,
};

// Climate components
// ClimateUDEB uses a manual builder because from_parameters returns Result
#[pyclass]
pub struct ClimateUDEBBuilder {
    parameters: ClimateUDEBParameters,
}

#[pymethods]
impl ClimateUDEBBuilder {
    #[staticmethod]
    pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
        use pyo3::exceptions::PyValueError;

        let parameters = pythonize::depythonize::<ClimateUDEBParameters>(&parameters);
        match parameters {
            Ok(parameters) => Ok(Self { parameters }),
            Err(e) => Err(PyValueError::new_err(format!("{}", e))),
        }
    }
    pub fn build(&self) -> PyResult<PyRustComponent> {
        use pyo3::exceptions::PyValueError;

        let component = ClimateUDEB::from_parameters(self.parameters.clone())
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(PyRustComponent(std::sync::Arc::new(component)))
    }
}

// Chemistry components
create_component_builder!(CH4ChemistryBuilder, CH4Chemistry, CH4ChemistryParameters);
create_component_builder!(N2OChemistryBuilder, N2OChemistry, N2OChemistryParameters);
create_component_builder!(
    HalocarbonChemistryBuilder,
    HalocarbonChemistry,
    HalocarbonParameters
);

// Carbon cycle components
create_component_builder!(
    TerrestrialCarbonBuilder,
    TerrestrialCarbon,
    TerrestrialCarbonParameters
);
create_component_builder!(OceanCarbonBuilder, OceanCarbon, OceanCarbonParameters);
create_component_builder!(CO2BudgetBuilder, CO2Budget, CO2BudgetParameters);

// Forcing components
create_component_builder!(GhgForcingBuilder, GhgForcing, GhgForcingParameters);
create_component_builder!(OzoneForcingBuilder, OzoneForcing, OzoneForcingParameters);
create_component_builder!(AerosolDirectBuilder, AerosolDirect, AerosolDirectParameters);
create_component_builder!(
    AerosolIndirectBuilder,
    AerosolIndirect,
    AerosolIndirectParameters
);

#[pymodule]
pub fn magicc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Climate
    m.add_class::<ClimateUDEBBuilder>()?;
    // Chemistry
    m.add_class::<CH4ChemistryBuilder>()?;
    m.add_class::<N2OChemistryBuilder>()?;
    m.add_class::<HalocarbonChemistryBuilder>()?;
    // Carbon
    m.add_class::<TerrestrialCarbonBuilder>()?;
    m.add_class::<OceanCarbonBuilder>()?;
    m.add_class::<CO2BudgetBuilder>()?;
    // Forcing
    m.add_class::<GhgForcingBuilder>()?;
    m.add_class::<OzoneForcingBuilder>()?;
    m.add_class::<AerosolDirectBuilder>()?;
    m.add_class::<AerosolIndirectBuilder>()?;
    Ok(())
}

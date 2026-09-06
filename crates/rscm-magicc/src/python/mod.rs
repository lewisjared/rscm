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
    /// Initial state and diagnostics for an unperturbed climate at the supplied ERF.
    pub fn initial_values(&self, erf: f64) -> PyResult<std::collections::HashMap<String, f64>> {
        let component = ClimateUDEB::from_parameters(self.parameters.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if !erf.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Initial forcing must be finite",
            ));
        }
        Ok(std::collections::HashMap::from([
            ("Surface Temperature".into(), 0.0),
            ("Sea Surface Temperature".into(), 0.0),
            ("Ocean Heat Content".into(), 0.0),
            ("Heat Uptake".into(), component.initial_heat_uptake(erf)),
        ]))
    }

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
/// Builder with a pointwise evaluator for initializing boundary forcing.
#[pyclass]
pub struct GhgForcingBuilder {
    parameters: GhgForcingParameters,
}

#[pymethods]
impl GhgForcingBuilder {
    #[staticmethod]
    pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
        let parameters = pythonize::depythonize::<GhgForcingParameters>(&parameters)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { parameters })
    }

    pub fn build(&self) -> PyRustComponent {
        PyRustComponent(std::sync::Arc::new(GhgForcing::from_parameters(
            self.parameters.clone(),
        )))
    }

    /// Evaluate CO2/CH4/N2O ERF at a single boundary, using the same physics as solve.
    pub fn calculate_forcings(
        &self,
        co2: f64,
        ch4: f64,
        n2o: f64,
    ) -> PyResult<std::collections::HashMap<String, f64>> {
        if [co2, ch4, n2o].iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Concentrations must be finite and positive",
            ));
        }
        let result =
            GhgForcing::from_parameters(self.parameters.clone()).calculate_forcings(co2, ch4, n2o);
        Ok(std::collections::HashMap::from([
            ("CO2".to_string(), result.co2_erf),
            ("CH4".to_string(), result.ch4_erf),
            ("N2O".to_string(), result.n2o_erf),
        ]))
    }
}
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

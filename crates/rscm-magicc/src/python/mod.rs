use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};
use rscm_core::names::SEA_LEVEL_RISE_THERMAL_EXPANSION;

use rscm_core::create_component_builder;
use rscm_core::python::PyRustComponent;

use crate::carbon::{CO2Budget, OceanCarbon, Permafrost, TerrestrialCarbon};
use crate::chemistry::{CH4Chemistry, CH4Oxidation, HalocarbonChemistry, N2OChemistry};
use crate::climate::{ClimateUdeb, SeaLevelRise};
use crate::forcing::{
    AerosolDirect, AerosolIndirect, ForcingAggregation, GhgForcing, OzoneForcing,
};
use crate::parameters::{
    AerosolDirectParameters, AerosolIndirectParameters, CH4ChemistryParameters,
    CH4OxidationParameters, CO2BudgetParameters, ClimateUdebParameters,
    ForcingAggregationParameters, GhgForcingParameters, HalocarbonChemistryParameters,
    N2OChemistryParameters, OceanCarbonParameters, OzoneForcingParameters, PermafrostParameters,
    SeaLevelRiseParameters, TerrestrialCarbonParameters,
};

// Climate components
// ClimateUdeb uses a manual builder because from_parameters returns Result
#[pyclass]
pub struct ClimateUdebBuilder {
    parameters: ClimateUdebParameters,
}

#[pymethods]
impl ClimateUdebBuilder {
    /// Initial state and diagnostics for an unperturbed climate at the supplied ERF.
    pub fn initial_values(&self, erf: f64) -> PyResult<std::collections::HashMap<String, f64>> {
        let component = ClimateUdeb::from_parameters(self.parameters.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if !erf.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Initial forcing must be finite",
            ));
        }
        let mut values = std::collections::HashMap::from([
            ("Surface Air Temperature Change".into(), 0.0),
            ("Sea Surface Temperature".into(), 0.0),
            ("Ocean Heat Content".into(), 0.0),
            ("Heat Uptake".into(), component.initial_heat_uptake(erf)),
        ]);
        if self.parameters.thermal_expansion.is_some() {
            values.insert(SEA_LEVEL_RISE_THERMAL_EXPANSION.into(), 0.0);
        }
        Ok(values)
    }

    /// Internal efficacy of a forcing agent with the given regional pattern.
    pub fn internal_efficacy(&self, regional_pattern: [f64; 4]) -> PyResult<f64> {
        let component = ClimateUdeb::from_parameters(self.parameters.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(component.internal_efficacy(&regional_pattern))
    }

    #[staticmethod]
    pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
        use pyo3::exceptions::PyValueError;

        let parameters = pythonize::depythonize::<ClimateUdebParameters>(&parameters);
        match parameters {
            Ok(parameters) => Ok(Self { parameters }),
            Err(e) => Err(PyValueError::new_err(format!("{}", e))),
        }
    }
    pub fn build(&self) -> PyResult<PyRustComponent> {
        use pyo3::exceptions::PyValueError;

        let component = ClimateUdeb::from_parameters(self.parameters.clone())
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(PyRustComponent(std::sync::Arc::new(component)))
    }
}

// Chemistry components
create_component_builder!(CH4ChemistryBuilder, CH4Chemistry, CH4ChemistryParameters);
create_component_builder!(CH4OxidationBuilder, CH4Oxidation, CH4OxidationParameters);
create_component_builder!(N2OChemistryBuilder, N2OChemistry, N2OChemistryParameters);
create_component_builder!(
    HalocarbonChemistryBuilder,
    HalocarbonChemistry,
    HalocarbonChemistryParameters
);

// Carbon cycle components
create_component_builder!(
    TerrestrialCarbonBuilder,
    TerrestrialCarbon,
    TerrestrialCarbonParameters
);
create_component_builder!(OceanCarbonBuilder, OceanCarbon, OceanCarbonParameters);
create_component_builder!(CO2BudgetBuilder, CO2Budget, CO2BudgetParameters);
create_component_builder!(PermafrostBuilder, Permafrost, PermafrostParameters);
create_component_builder!(SeaLevelRiseBuilder, SeaLevelRise, SeaLevelRiseParameters);

// Forcing components
/// Builder with a pointwise evaluator for initialising boundary forcing.
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

    /// ERF from doubling CO2 above its preindustrial reference, with CH4 and N2O
    /// at theirs. MAGICC7 derives its climate feedback from this value.
    pub fn erf_2xco2(&self) -> f64 {
        let p = &self.parameters;
        GhgForcing::from_parameters(p.clone())
            .calculate_forcings(2.0 * p.co2_pi, p.ch4_pi, p.n2o_pi)
            .co2_erf
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

/// Builder for `ForcingAggregation`, whose parameters are validated on build.
#[pyclass]
pub struct ForcingAggregationBuilder {
    parameters: ForcingAggregationParameters,
}

#[pymethods]
impl ForcingAggregationBuilder {
    #[staticmethod]
    pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
        pythonize::depythonize::<ForcingAggregationParameters>(&parameters)
            .map(|parameters| Self { parameters })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Total ERF and four-box efficacy-adjusted forcing for one value per agent.
    pub fn aggregate(&self, values: Vec<Vec<f64>>) -> PyResult<(f64, [f64; 4])> {
        let component = ForcingAggregation::from_parameters(self.parameters.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let agents = &component.parameters().agents;
        let valid = values.len() == agents.len()
            && agents
                .iter()
                .zip(&values)
                .all(|(agent, v)| v.len() == if agent.regional { 4 } else { 1 });
        if !valid {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Supply one value per global agent and four per regional agent",
            ));
        }
        Ok(component.aggregate(&values))
    }

    /// Internal efficacy of each agent at climate sensitivity `ecs` and CO2 doubling forcing `erf_2xco2`.
    pub fn internal_efficacies(&self, ecs: f64, erf_2xco2: f64) -> PyResult<Vec<f64>> {
        ForcingAggregation::from_parameters(self.parameters.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .internal_efficacies(ecs, erf_2xco2)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("LAMCALC did not converge"))
    }

    pub fn build(&self) -> PyResult<PyRustComponent> {
        ForcingAggregation::from_parameters(self.parameters.clone())
            .map(|component| PyRustComponent(std::sync::Arc::new(component)))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[pymodule]
pub fn magicc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Climate
    m.add_class::<ClimateUdebBuilder>()?;
    m.add_class::<SeaLevelRiseBuilder>()?;
    // Chemistry
    m.add_class::<CH4ChemistryBuilder>()?;
    m.add_class::<CH4OxidationBuilder>()?;
    m.add_class::<N2OChemistryBuilder>()?;
    m.add_class::<HalocarbonChemistryBuilder>()?;
    // Carbon
    m.add_class::<TerrestrialCarbonBuilder>()?;
    m.add_class::<OceanCarbonBuilder>()?;
    m.add_class::<CO2BudgetBuilder>()?;
    m.add_class::<PermafrostBuilder>()?;
    // Forcing
    m.add_class::<GhgForcingBuilder>()?;
    m.add_class::<OzoneForcingBuilder>()?;
    m.add_class::<AerosolDirectBuilder>()?;
    m.add_class::<AerosolIndirectBuilder>()?;
    m.add_class::<ForcingAggregationBuilder>()?;
    Ok(())
}

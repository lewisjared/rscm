use numpy::{PyReadonlyArray2, PyUntypedArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{pymodule, Bound, PyResult};
use rscm_core::names::{
    ATMOSPHERIC_BURDEN_CH4_BIOSPHERE, ATMOSPHERIC_BURDEN_CH4_FOSSIL,
    ATMOSPHERIC_BURDEN_CH4_NATURAL, ATMOSPHERIC_BURDEN_CH4_PERMAFROST, HEAT_UPTAKE,
    OCEAN_HEAT_CONTENT, SEA_LEVEL_RISE_THERMAL_EXPANSION, SEA_SURFACE_TEMPERATURE,
    SURFACE_AIR_TEMPERATURE_CHANGE,
};

use rscm_core::create_component_builder;
use rscm_core::python::PyRustComponent;
use rscm_core::timeseries::FloatValue;

use crate::carbon::{CO2Budget, OceanCarbon, Permafrost, TerrestrialCarbon};
use crate::chemistry::{CH4Chemistry, CH4Oxidation, HalocarbonChemistry, N2OChemistry};
use crate::climate::{ClimateUDEB, SeaLevelRise};
use crate::forcing::{Drivers, ForcingAggregation, GHGForcing, NitrateCloudForcing, OzoneForcing};
use crate::parameters::{
    CH4ChemistryParameters, CH4OxidationParameters, CO2BudgetParameters, ClimateUDEBParameters,
    ForcingAggregationParameters, GHGForcingParameters, HalocarbonChemistryParameters,
    N2OChemistryParameters, NitrateCloudForcingParameters, OceanCarbonParameters,
    OzoneForcingParameters, PermafrostParameters, SeaLevelRiseParameters,
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
    /// Initial values and diagnostics for an unperturbed climate at the supplied ERF.
    pub fn initial_values(&self, erf: f64) -> PyResult<std::collections::HashMap<String, f64>> {
        let component = ClimateUDEB::from_parameters(self.parameters.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if !erf.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Initial forcing must be finite",
            ));
        }
        let mut values = std::collections::HashMap::from([
            (SURFACE_AIR_TEMPERATURE_CHANGE.into(), 0.0),
            (SEA_SURFACE_TEMPERATURE.into(), 0.0),
            (OCEAN_HEAT_CONTENT.into(), 0.0),
            (HEAT_UPTAKE.into(), component.initial_heat_uptake(erf)),
        ]);
        if self.parameters.thermal_expansion.is_some() {
            values.insert(SEA_LEVEL_RISE_THERMAL_EXPANSION.into(), 0.0);
        }
        Ok(values)
    }

    /// Internal efficacy of a forcing agent with the given regional pattern.
    pub fn internal_efficacy(&self, regional_pattern: [f64; 4]) -> PyResult<f64> {
        let component = ClimateUDEB::from_parameters(self.parameters.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(component.internal_efficacy(&regional_pattern))
    }

    /// Regional pattern and first-years internal efficacy of the subannual forcing,
    /// with any unset ones derived, or `None` without subannual forcing.
    pub fn subannual_efficacy(&self) -> PyResult<Option<(Option<[f64; 4]>, f64)>> {
        let component = ClimateUDEB::from_parameters(self.parameters.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(component
            .parameters()
            .subannual_forcing
            .as_ref()
            .map(|f| (f.regional_pattern, f.internal_efficacy.unwrap_or(1.0))))
    }

    #[staticmethod]
    pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
        let (parameters, values) = take_subannual_values(&parameters)?;
        let mut parameters =
            rscm_core::python::parameters::depythonize::<ClimateUDEBParameters>(&parameters)?;
        if let (Some(forcing), Some(values)) = (parameters.subannual_forcing.as_mut(), values) {
            forcing.values = values;
        }
        Ok(Self { parameters })
    }
    pub fn build(&self) -> PyResult<PyRustComponent> {
        use pyo3::exceptions::PyValueError;

        let component = ClimateUDEB::from_parameters(self.parameters.clone())
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(PyRustComponent(std::sync::Arc::new(component)))
    }
}

/// Subannual forcing values, one row of four boxes per substep.
type SubannualValues = Vec<[FloatValue; 4]>;

/// The parameters without a numpy array of subannual forcing values, and those values.
///
/// pythonize reads lists and tuples but not numpy arrays.
fn take_subannual_values<'py>(
    parameters: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Option<SubannualValues>)> {
    let unchanged = || Ok((parameters.clone(), None));
    let Ok(dict) = parameters.cast::<PyDict>() else {
        return unchanged();
    };
    let forcing = dict.get_item("subannual_forcing")?;
    let Some(forcing) = forcing.as_ref().and_then(|f| f.cast::<PyDict>().ok()) else {
        return unchanged();
    };
    let values = forcing.get_item("values")?;
    if !values
        .as_ref()
        .is_some_and(|v| v.is_instance_of::<PyUntypedArray>())
    {
        return unchanged();
    }
    let array = values
        .and_then(|v| v.extract::<PyReadonlyArray2<FloatValue>>().ok())
        .ok_or_else(|| PyValueError::new_err("subannual_forcing values must be 2-D float64"))?;
    let array = array.as_array();
    if array.ncols() != 4 {
        return Err(PyValueError::new_err(format!(
            "subannual_forcing values need 4 boxes per substep, got {}",
            array.ncols()
        )));
    }
    let rows = array
        .outer_iter()
        .map(|row| [row[0], row[1], row[2], row[3]])
        .collect();
    let forcing = forcing.copy()?;
    forcing.set_item("values", PyList::empty(parameters.py()))?;
    let dict = dict.copy()?;
    dict.set_item("subannual_forcing", forcing)?;
    Ok((dict.into_any(), Some(rows)))
}

// Chemistry components
create_component_builder!(CH4ChemistryBuilder, CH4Chemistry, CH4ChemistryParameters);

#[pymethods]
impl CH4ChemistryBuilder {
    /// CH4 lifetime (yr) against soil, stratospheric and chlorine loss.
    pub fn non_oh_lifetime(&self) -> f64 {
        self.parameters.non_oh_lifetime()
    }

    /// CH4 lifetime (yr) against OH at the start.
    pub fn initial_oh_lifetime(&self) -> f64 {
        self.parameters.initial_oh_lifetime()
    }
}
create_component_builder!(CH4OxidationBuilder, CH4Oxidation, CH4OxidationParameters);

#[pymethods]
impl CH4OxidationBuilder {
    /// Initial CH4 burdens by source at `ch4_concentration` ppb, all of it natural.
    pub fn initial_values(&self, ch4_concentration: f64) -> std::collections::HashMap<String, f64> {
        std::collections::HashMap::from([
            (ATMOSPHERIC_BURDEN_CH4_FOSSIL.into(), 0.0),
            (ATMOSPHERIC_BURDEN_CH4_BIOSPHERE.into(), 0.0),
            (
                ATMOSPHERIC_BURDEN_CH4_NATURAL.into(),
                self.parameters.initial_natural_burden(ch4_concentration),
            ),
            (ATMOSPHERIC_BURDEN_CH4_PERMAFROST.into(), 0.0),
        ])
    }
}
create_component_builder!(N2OChemistryBuilder, N2OChemistry, N2OChemistryParameters);

#[pymethods]
impl N2OChemistryBuilder {
    /// Natural emissions (Mt N2ON/yr) that hold the starting concentration (ppb)
    /// over the first step, given the anthropogenic emissions (Mt N2ON/yr) read in that step.
    pub fn natural_emissions_in_balance(
        &self,
        concentration: f64,
        anthropogenic_emissions: f64,
    ) -> f64 {
        N2OChemistry::from_parameters(self.parameters.clone())
            .natural_emissions_in_balance(concentration, anthropogenic_emissions)
    }
}
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
pub struct GHGForcingBuilder {
    parameters: GHGForcingParameters,
}

#[pymethods]
impl GHGForcingBuilder {
    #[staticmethod]
    pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
        let parameters =
            rscm_core::python::parameters::depythonize::<GHGForcingParameters>(&parameters)?;
        Ok(Self { parameters })
    }

    pub fn build(&self) -> PyRustComponent {
        PyRustComponent(std::sync::Arc::new(GHGForcing::from_parameters(
            self.parameters.clone(),
        )))
    }

    /// ERF from doubling CO2 above its preindustrial reference, with CH4 and N2O
    /// at theirs. MAGICC7 derives its climate feedback from this value.
    pub fn erf_2xco2(&self) -> f64 {
        let p = &self.parameters;
        GHGForcing::from_parameters(p.clone())
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
            GHGForcing::from_parameters(self.parameters.clone()).calculate_forcings(co2, ch4, n2o);
        Ok(std::collections::HashMap::from([
            ("CO2".to_string(), result.co2_erf),
            ("CH4".to_string(), result.ch4_erf),
            ("N2O".to_string(), result.n2o_erf),
        ]))
    }
}
create_component_builder!(OzoneForcingBuilder, OzoneForcing, OzoneForcingParameters);

/// Builder for `ForcingAggregation`, whose parameters are validated on build.
#[pyclass]
pub struct ForcingAggregationBuilder {
    parameters: ForcingAggregationParameters,
}

#[pymethods]
impl ForcingAggregationBuilder {
    #[staticmethod]
    pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
        rscm_core::python::parameters::depythonize::<ForcingAggregationParameters>(&parameters)
            .map(|parameters| Self { parameters })
    }

    /// Total effective radiative forcing and four-box efficacy-adjusted forcing
    /// at the first of `time`.
    ///
    /// Takes one value per agent variable, each `sum_of` variable in turn.
    /// `inputs` holds the prescribed series of agent variables on `time`,
    /// which harmonised and held agents read.
    #[pyo3(signature = (values, inputs=None, time=None))]
    pub fn aggregate(
        &self,
        values: Vec<Vec<f64>>,
        inputs: Option<std::collections::HashMap<String, numpy::PyReadonlyArrayDyn<f64>>>,
        time: Option<Vec<f64>>,
    ) -> PyResult<(f64, [f64; 4])> {
        let component = ForcingAggregation::from_parameters(self.parameters.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let lengths: Vec<usize> = component
            .parameters()
            .agents
            .iter()
            .flat_map(|agent| {
                let n = agent.sum_of.len().max(1);
                std::iter::repeat_n(if agent.regional { 4 } else { 1 }, n)
            })
            .collect();
        let valid = values.len() == lengths.len()
            && lengths.iter().zip(&values).all(|(n, v)| v.len() == *n);
        if !valid {
            return Err(PyValueError::new_err(
                "Supply one value per global agent variable and four per regional one",
            ));
        }
        let series = SeriesLookup::new(inputs, time)?;
        component
            .aggregate(&values, &|name, _, t| series.at(name, t), series.first())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Scaled four-box ERF before efficacies of each agent variable in the prescribed `inputs`
    /// on `time`, at each of `at`, which defaults to `time`.
    #[pyo3(signature = (inputs, time, at=None))]
    pub fn variable_box_forcing(
        &self,
        py: Python<'_>,
        inputs: std::collections::HashMap<String, numpy::PyReadonlyArrayDyn<f64>>,
        time: Vec<f64>,
        at: Option<Vec<f64>>,
    ) -> PyResult<std::collections::HashMap<String, Py<PyAny>>> {
        let component = ForcingAggregation::from_parameters(self.parameters.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let at = at.unwrap_or_else(|| time.clone());
        let series = SeriesLookup::new(Some(inputs), Some(time))?;
        let forcing = component
            .variable_box_forcing(
                &|name, _, t| series.at(name, t),
                &|name| series.series.contains_key(name),
                &at,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        forcing
            .into_iter()
            .map(|(name, boxes)| {
                let array = numpy::PyArray2::from_vec2(
                    py,
                    &boxes.iter().map(|b| b.to_vec()).collect::<Vec<_>>(),
                )?;
                Ok((name, array.into_any().unbind()))
            })
            .collect()
    }

    /// Scale of each agent variable, keyed by variable, from the prescribed `inputs` on `time`.
    #[pyo3(signature = (inputs=None, time=None))]
    pub fn variable_scales(
        &self,
        inputs: Option<std::collections::HashMap<String, numpy::PyReadonlyArrayDyn<f64>>>,
        time: Option<Vec<f64>>,
    ) -> PyResult<std::collections::HashMap<String, f64>> {
        let component = ForcingAggregation::from_parameters(self.parameters.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let series = SeriesLookup::new(inputs, time)?;
        let scales = component
            .variable_scales(&|name, _, t| series.at(name, t))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let names = component.parameters().agents.iter().flat_map(|agent| {
            if agent.sum_of.is_empty() {
                std::slice::from_ref(&agent.variable)
            } else {
                agent.sum_of.as_slice()
            }
        });
        Ok(names.cloned().zip(scales).collect())
    }

    /// Internal efficacy of each agent at equilibrium climate sensitivity `ecs`
    /// and 2xCO2 effective radiative forcing `erf_2xco2`.
    ///
    /// Agents with `pattern_terms` take their pattern from the prescribed `inputs` on `time`.
    #[pyo3(signature = (ecs, erf_2xco2, inputs=None, time=None))]
    pub fn internal_efficacies(
        &self,
        ecs: f64,
        erf_2xco2: f64,
        inputs: Option<std::collections::HashMap<String, numpy::PyReadonlyArrayDyn<f64>>>,
        time: Option<Vec<f64>>,
    ) -> PyResult<Vec<f64>> {
        let component = ForcingAggregation::from_parameters(self.parameters.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let series = SeriesLookup::new(inputs, time)?;
        component
            .internal_efficacies_from(ecs, erf_2xco2, &|name, _, t| series.at(name, t))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .ok_or_else(|| PyValueError::new_err("LAMCALC did not converge"))
    }

    /// Pattern of each agent, from the prescribed `inputs` on `time` for agents with `pattern_terms`.
    #[pyo3(signature = (inputs=None, time=None))]
    pub fn patterns(
        &self,
        inputs: Option<std::collections::HashMap<String, numpy::PyReadonlyArrayDyn<f64>>>,
        time: Option<Vec<f64>>,
    ) -> PyResult<Vec<[f64; 4]>> {
        let component = ForcingAggregation::from_parameters(self.parameters.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let series = SeriesLookup::new(inputs, time)?;
        component
            .patterns(&|name, _, t| series.at(name, t))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn build(&self) -> PyResult<PyRustComponent> {
        ForcingAggregation::from_parameters(self.parameters.clone())
            .map(|component| PyRustComponent(std::sync::Arc::new(component)))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// Builder for `NitrateCloudForcing`, whose parameters are validated on build.
#[pyclass]
pub struct NitrateCloudForcingBuilder {
    parameters: NitrateCloudForcingParameters,
}

impl NitrateCloudForcingBuilder {
    fn with_drivers<T>(
        &self,
        inputs: std::collections::HashMap<String, numpy::PyReadonlyArrayDyn<f64>>,
        time: Vec<f64>,
        f: impl FnOnce(&NitrateCloudForcing, &Drivers) -> rscm_core::errors::RSCMResult<T>,
    ) -> PyResult<T> {
        let component = NitrateCloudForcing::from_parameters(self.parameters.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (Some(&first), Some(&last)) = (time.first(), time.last()) else {
            return Err(PyValueError::new_err("time needs at least one value"));
        };
        let series = SeriesLookup::new(Some(inputs), Some(time))?;
        let emissions = |name: &str, t: f64| Ok(series.at(name, t)?[0]);
        let optical_thickness = |name: &str, t: f64| {
            let row = series.at(name, t)?;
            <[f64; 4]>::try_from(row).map_err(|_| {
                rscm_core::errors::RSCMError::Error(format!("{name} needs four boxes"))
            })
        };
        let drivers = Drivers {
            emissions: &emissions,
            optical_thickness: &optical_thickness,
            first,
            last,
        };
        f(&component, &drivers).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl NitrateCloudForcingBuilder {
    #[staticmethod]
    pub fn from_parameters(parameters: Bound<PyAny>) -> PyResult<Self> {
        rscm_core::python::parameters::depythonize::<NitrateCloudForcingParameters>(&parameters)
            .map(|parameters| Self { parameters })
    }

    /// Four-box forcing of each output in each year of `time`, from the prescribed `inputs` on it.
    pub fn forcing(
        &self,
        py: Python<'_>,
        inputs: std::collections::HashMap<String, numpy::PyReadonlyArrayDyn<f64>>,
        time: Vec<f64>,
    ) -> PyResult<std::collections::HashMap<String, Py<PyAny>>> {
        let years = time.clone();
        let rows = self.with_drivers(inputs, time, |component, drivers| {
            years
                .iter()
                .map(|&year| component.forcing(drivers, year))
                .collect::<rscm_core::errors::RSCMResult<Vec<_>>>()
        })?;
        let column = |f: fn(&crate::forcing::NitrateCloud) -> [f64; 4]| {
            numpy::PyArray2::from_vec2(py, &rows.iter().map(|r| f(r).to_vec()).collect::<Vec<_>>())
                .map(|a| a.into_any().unbind())
        };
        Ok(std::collections::HashMap::from([
            (
                crate::forcing::NITRATE_SECTORS[0].to_string(),
                column(|r| r.nitrate_fossil)?,
            ),
            (
                crate::forcing::NITRATE_SECTORS[1].to_string(),
                column(|r| r.nitrate_biosphere)?,
            ),
            (
                rscm_core::names::ERF_AEROSOLS_INDIRECT_EFFECT.to_string(),
                column(|r| r.cloud_albedo)?,
            ),
        ]))
    }

    /// Box nitrate forcing per unit global nitrate forcing in `year`,
    /// from the prescribed `inputs` on `time`.
    pub fn nitrate_pattern(
        &self,
        inputs: std::collections::HashMap<String, numpy::PyReadonlyArrayDyn<f64>>,
        time: Vec<f64>,
        year: f64,
    ) -> PyResult<[f64; 4]> {
        self.with_drivers(inputs, time, |component, drivers| {
            component.nitrate_pattern(drivers, year)
        })
    }

    pub fn build(&self) -> PyResult<PyRustComponent> {
        NitrateCloudForcing::from_parameters(self.parameters.clone())
            .map(|component| PyRustComponent(std::sync::Arc::new(component)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Prescribed series by name on a time axis, read at any time as a model window reads them.
struct SeriesLookup {
    series: std::collections::HashMap<String, Vec<Vec<f64>>>,
    time: Vec<f64>,
}

impl SeriesLookup {
    fn new(
        inputs: Option<std::collections::HashMap<String, numpy::PyReadonlyArrayDyn<f64>>>,
        time: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let time = time.unwrap_or_default();
        let mut series = std::collections::HashMap::new();
        for (name, array) in inputs.unwrap_or_default() {
            let array = array.as_array();
            if array.shape().first() != Some(&time.len()) || array.ndim() > 2 {
                return Err(PyValueError::new_err(format!(
                    "{name} needs one row per time, got shape {:?}",
                    array.shape()
                )));
            }
            let rows = array
                .outer_iter()
                .map(|row| row.iter().copied().collect())
                .collect();
            series.insert(name, rows);
        }
        Ok(Self { series, time })
    }

    fn first(&self) -> f64 {
        self.time.first().copied().unwrap_or(f64::NAN)
    }

    /// Value at a node, held beyond the ends.
    ///
    /// Errors between nodes, because the model's interpolation is not repeated here.
    fn at(&self, name: &str, t: f64) -> rscm_core::errors::RSCMResult<Vec<f64>> {
        let rows = self.series.get(name).ok_or_else(|| {
            rscm_core::errors::RSCMError::Error(format!("no prescribed series for {name}"))
        })?;
        let time = &self.time;
        let Some(last) = time.len().checked_sub(1) else {
            return Err(rscm_core::errors::RSCMError::Error(format!(
                "{name} needs the time of its prescribed series"
            )));
        };
        if t <= time[0] {
            return Ok(rows[0].clone());
        }
        if t >= time[last] {
            return Ok(rows[last].clone());
        }
        let i = time.partition_point(|&x| x <= t) - 1;
        if time[i] == t {
            return Ok(rows[i].clone());
        }
        Err(rscm_core::errors::RSCMError::Error(format!(
            "{name} is read at {t}, between the times of its prescribed series"
        )))
    }
}

#[pymodule]
pub fn magicc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Climate
    m.add_class::<ClimateUDEBBuilder>()?;
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
    m.add_class::<GHGForcingBuilder>()?;
    m.add_class::<OzoneForcingBuilder>()?;
    m.add_class::<NitrateCloudForcingBuilder>()?;
    m.add_class::<ForcingAggregationBuilder>()?;
    Ok(())
}

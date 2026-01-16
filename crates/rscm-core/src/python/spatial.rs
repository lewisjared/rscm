use crate::spatial::{
    FourBoxGrid, FourBoxRegion, HemisphericGrid, HemisphericRegion, ScalarGrid, ScalarRegion,
    SpatialGrid,
};
use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};

#[pyclass]
#[pyo3(name = "ScalarRegion")]
#[derive(Clone, Copy)]
pub struct PyScalarRegion(pub ScalarRegion);

#[pymethods]
impl PyScalarRegion {
    #[classattr]
    const GLOBAL: usize = ScalarRegion::Global as usize;

    fn __repr__(&self) -> String {
        "ScalarRegion.GLOBAL".to_string()
    }
}

#[pyclass]
#[pyo3(name = "FourBoxRegion")]
#[derive(Clone, Copy)]
pub struct PyFourBoxRegion(pub FourBoxRegion);

#[pymethods]
impl PyFourBoxRegion {
    #[classattr]
    const NORTHERN_OCEAN: usize = FourBoxRegion::NorthernOcean as usize;

    #[classattr]
    const NORTHERN_LAND: usize = FourBoxRegion::NorthernLand as usize;

    #[classattr]
    const SOUTHERN_OCEAN: usize = FourBoxRegion::SouthernOcean as usize;

    #[classattr]
    const SOUTHERN_LAND: usize = FourBoxRegion::SouthernLand as usize;

    fn __repr__(&self) -> String {
        match self.0 {
            FourBoxRegion::NorthernOcean => "FourBoxRegion.NORTHERN_OCEAN".to_string(),
            FourBoxRegion::NorthernLand => "FourBoxRegion.NORTHERN_LAND".to_string(),
            FourBoxRegion::SouthernOcean => "FourBoxRegion.SOUTHERN_OCEAN".to_string(),
            FourBoxRegion::SouthernLand => "FourBoxRegion.SOUTHERN_LAND".to_string(),
        }
    }
}

#[pyclass]
#[pyo3(name = "HemisphericRegion")]
#[derive(Clone, Copy)]
pub struct PyHemisphericRegion(pub HemisphericRegion);

#[pymethods]
impl PyHemisphericRegion {
    #[classattr]
    const NORTHERN: usize = HemisphericRegion::Northern as usize;

    #[classattr]
    const SOUTHERN: usize = HemisphericRegion::Southern as usize;

    fn __repr__(&self) -> String {
        match self.0 {
            HemisphericRegion::Northern => "HemisphericRegion.NORTHERN".to_string(),
            HemisphericRegion::Southern => "HemisphericRegion.SOUTHERN".to_string(),
        }
    }
}

#[pyclass]
#[pyo3(name = "ScalarGrid")]
#[derive(Clone)]
pub struct PyScalarGrid(pub ScalarGrid);

#[pymethods]
impl PyScalarGrid {
    #[new]
    fn new() -> Self {
        Self(ScalarGrid)
    }

    fn __repr__(&self) -> String {
        format!("<ScalarGrid size={}>", self.0.size())
    }

    fn grid_name(&self) -> &'static str {
        self.0.grid_name()
    }

    fn size(&self) -> usize {
        self.0.size()
    }

    fn region_names(&self) -> Vec<String> {
        self.0.region_names().to_vec()
    }

    fn aggregate_global(&self, values: Vec<f64>) -> f64 {
        self.0.aggregate_global(&values)
    }
}

#[pyclass]
#[pyo3(name = "FourBoxGrid")]
#[derive(Clone)]
pub struct PyFourBoxGrid(pub FourBoxGrid);

#[pymethods]
impl PyFourBoxGrid {
    #[new]
    #[pyo3(signature = (weights=None))]
    fn new(weights: Option<[f64; 4]>) -> Self {
        match weights {
            Some(w) => Self(FourBoxGrid::with_weights(w)),
            None => Self(FourBoxGrid::magicc_standard()),
        }
    }

    #[staticmethod]
    fn magicc_standard() -> Self {
        Self(FourBoxGrid::magicc_standard())
    }

    #[staticmethod]
    fn with_weights(weights: [f64; 4]) -> Self {
        Self(FourBoxGrid::with_weights(weights))
    }

    fn __repr__(&self) -> String {
        format!("<FourBoxGrid size={}>", self.0.size())
    }

    fn grid_name(&self) -> &'static str {
        self.0.grid_name()
    }

    fn size(&self) -> usize {
        self.0.size()
    }

    fn region_names(&self) -> Vec<String> {
        self.0.region_names().to_vec()
    }

    fn weights(&self) -> [f64; 4] {
        *self.0.weights()
    }

    fn aggregate_global(&self, values: Vec<f64>) -> f64 {
        self.0.aggregate_global(&values)
    }
}

#[pyclass]
#[pyo3(name = "HemisphericGrid")]
#[derive(Clone)]
pub struct PyHemisphericGrid(pub HemisphericGrid);

#[pymethods]
impl PyHemisphericGrid {
    #[new]
    #[pyo3(signature = (weights=None))]
    fn new(weights: Option<[f64; 2]>) -> Self {
        match weights {
            Some(w) => Self(HemisphericGrid::with_weights(w)),
            None => Self(HemisphericGrid::equal_weights()),
        }
    }

    #[staticmethod]
    fn equal_weights() -> Self {
        Self(HemisphericGrid::equal_weights())
    }

    #[staticmethod]
    fn with_weights(weights: [f64; 2]) -> Self {
        Self(HemisphericGrid::with_weights(weights))
    }

    fn __repr__(&self) -> String {
        format!("<HemisphericGrid size={}>", self.0.size())
    }

    fn grid_name(&self) -> &'static str {
        self.0.grid_name()
    }

    fn size(&self) -> usize {
        self.0.size()
    }

    fn region_names(&self) -> Vec<String> {
        self.0.region_names().to_vec()
    }

    fn weights(&self) -> [f64; 2] {
        *self.0.weights()
    }

    fn aggregate_global(&self, values: Vec<f64>) -> f64 {
        self.0.aggregate_global(&values)
    }
}

#[pymodule]
pub fn spatial(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScalarRegion>()?;
    m.add_class::<PyScalarGrid>()?;
    m.add_class::<PyFourBoxRegion>()?;
    m.add_class::<PyFourBoxGrid>()?;
    m.add_class::<PyHemisphericRegion>()?;
    m.add_class::<PyHemisphericGrid>()?;
    Ok(())
}

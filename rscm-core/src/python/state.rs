//! Python bindings for typed state access
//!
//! This module provides Python wrappers for TimeseriesWindow, GridTimeseriesWindow,
//! and typed output slices (FourBoxSlice, HemisphericSlice).

use crate::spatial::{FourBoxRegion, HemisphericRegion};
use crate::state::{FourBoxSlice, HemisphericSlice};
use crate::timeseries::FloatValue;
use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// =============================================================================
// Python Typed Output Slices
// =============================================================================

/// Python wrapper for FourBoxSlice
///
/// Provides type-safe access to regional values with named keyword arguments.
///
/// Example:
///     slice = FourBoxSlice(
///         northern_ocean=15.0,
///         northern_land=14.0,
///         southern_ocean=10.0,
///         southern_land=9.0
///     )
#[pyclass]
#[pyo3(name = "FourBoxSlice")]
#[derive(Debug, Clone)]
pub struct PyFourBoxSlice(pub FourBoxSlice);

#[pymethods]
impl PyFourBoxSlice {
    #[new]
    #[pyo3(signature = (northern_ocean=f64::NAN, northern_land=f64::NAN, southern_ocean=f64::NAN, southern_land=f64::NAN))]
    fn new(
        northern_ocean: FloatValue,
        northern_land: FloatValue,
        southern_ocean: FloatValue,
        southern_land: FloatValue,
    ) -> Self {
        Self(FourBoxSlice::from_array([
            northern_ocean,
            northern_land,
            southern_ocean,
            southern_land,
        ]))
    }

    /// Create a slice with all regions set to the same value
    #[staticmethod]
    fn uniform(value: FloatValue) -> Self {
        Self(FourBoxSlice::uniform(value))
    }

    /// Create a slice from an array [northern_ocean, northern_land, southern_ocean, southern_land]
    #[staticmethod]
    fn from_array(values: [FloatValue; 4]) -> Self {
        Self(FourBoxSlice::from_array(values))
    }

    /// Get the northern ocean value
    #[getter]
    fn northern_ocean(&self) -> FloatValue {
        self.0.get(FourBoxRegion::NorthernOcean)
    }

    /// Set the northern ocean value
    #[setter]
    fn set_northern_ocean(&mut self, value: FloatValue) {
        self.0.set(FourBoxRegion::NorthernOcean, value);
    }

    /// Get the northern land value
    #[getter]
    fn northern_land(&self) -> FloatValue {
        self.0.get(FourBoxRegion::NorthernLand)
    }

    /// Set the northern land value
    #[setter]
    fn set_northern_land(&mut self, value: FloatValue) {
        self.0.set(FourBoxRegion::NorthernLand, value);
    }

    /// Get the southern ocean value
    #[getter]
    fn southern_ocean(&self) -> FloatValue {
        self.0.get(FourBoxRegion::SouthernOcean)
    }

    /// Set the southern ocean value
    #[setter]
    fn set_southern_ocean(&mut self, value: FloatValue) {
        self.0.set(FourBoxRegion::SouthernOcean, value);
    }

    /// Get the southern land value
    #[getter]
    fn southern_land(&self) -> FloatValue {
        self.0.get(FourBoxRegion::SouthernLand)
    }

    /// Set the southern land value
    #[setter]
    fn set_southern_land(&mut self, value: FloatValue) {
        self.0.set(FourBoxRegion::SouthernLand, value);
    }

    /// Get value by region index
    fn get(&self, region: usize) -> PyResult<FloatValue> {
        match region {
            0 => Ok(self.0.get(FourBoxRegion::NorthernOcean)),
            1 => Ok(self.0.get(FourBoxRegion::NorthernLand)),
            2 => Ok(self.0.get(FourBoxRegion::SouthernOcean)),
            3 => Ok(self.0.get(FourBoxRegion::SouthernLand)),
            _ => Err(PyValueError::new_err(format!(
                "Invalid region index: {}. Must be 0-3.",
                region
            ))),
        }
    }

    /// Set value by region index
    fn set(&mut self, region: usize, value: FloatValue) -> PyResult<()> {
        match region {
            0 => self.0.set(FourBoxRegion::NorthernOcean, value),
            1 => self.0.set(FourBoxRegion::NorthernLand, value),
            2 => self.0.set(FourBoxRegion::SouthernOcean, value),
            3 => self.0.set(FourBoxRegion::SouthernLand, value),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid region index: {}. Must be 0-3.",
                    region
                )))
            }
        }
        Ok(())
    }

    /// Convert to numpy array
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<FloatValue>> {
        self.0.as_array().to_pyarray(py)
    }

    /// Convert to list
    fn to_list(&self) -> Vec<FloatValue> {
        self.0.to_vec()
    }

    /// Convert to dict with region names as keys
    fn to_dict(&self) -> std::collections::HashMap<String, FloatValue> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "northern_ocean".to_string(),
            self.0.get(FourBoxRegion::NorthernOcean),
        );
        map.insert(
            "northern_land".to_string(),
            self.0.get(FourBoxRegion::NorthernLand),
        );
        map.insert(
            "southern_ocean".to_string(),
            self.0.get(FourBoxRegion::SouthernOcean),
        );
        map.insert(
            "southern_land".to_string(),
            self.0.get(FourBoxRegion::SouthernLand),
        );
        map
    }

    fn __repr__(&self) -> String {
        format!(
            "FourBoxSlice(northern_ocean={}, northern_land={}, southern_ocean={}, southern_land={})",
            self.0.get(FourBoxRegion::NorthernOcean),
            self.0.get(FourBoxRegion::NorthernLand),
            self.0.get(FourBoxRegion::SouthernOcean),
            self.0.get(FourBoxRegion::SouthernLand)
        )
    }

    fn __getitem__(&self, index: usize) -> PyResult<FloatValue> {
        self.get(index)
    }

    fn __setitem__(&mut self, index: usize, value: FloatValue) -> PyResult<()> {
        self.set(index, value)
    }

    fn __len__(&self) -> usize {
        4
    }
}

/// Python wrapper for HemisphericSlice
///
/// Provides type-safe access to hemispheric values with named keyword arguments.
///
/// Example:
///     slice = HemisphericSlice(northern=15.0, southern=10.0)
#[pyclass]
#[pyo3(name = "HemisphericSlice")]
#[derive(Debug, Clone)]
pub struct PyHemisphericSlice(pub HemisphericSlice);

#[pymethods]
impl PyHemisphericSlice {
    #[new]
    #[pyo3(signature = (northern=f64::NAN, southern=f64::NAN))]
    fn new(northern: FloatValue, southern: FloatValue) -> Self {
        Self(HemisphericSlice::from_array([northern, southern]))
    }

    /// Create a slice with both hemispheres set to the same value
    #[staticmethod]
    fn uniform(value: FloatValue) -> Self {
        Self(HemisphericSlice::uniform(value))
    }

    /// Create a slice from an array [northern, southern]
    #[staticmethod]
    fn from_array(values: [FloatValue; 2]) -> Self {
        Self(HemisphericSlice::from_array(values))
    }

    /// Get the northern hemisphere value
    #[getter]
    fn northern(&self) -> FloatValue {
        self.0.get(HemisphericRegion::Northern)
    }

    /// Set the northern hemisphere value
    #[setter]
    fn set_northern(&mut self, value: FloatValue) {
        self.0.set(HemisphericRegion::Northern, value);
    }

    /// Get the southern hemisphere value
    #[getter]
    fn southern(&self) -> FloatValue {
        self.0.get(HemisphericRegion::Southern)
    }

    /// Set the southern hemisphere value
    #[setter]
    fn set_southern(&mut self, value: FloatValue) {
        self.0.set(HemisphericRegion::Southern, value);
    }

    /// Get value by region index
    fn get(&self, region: usize) -> PyResult<FloatValue> {
        match region {
            0 => Ok(self.0.get(HemisphericRegion::Northern)),
            1 => Ok(self.0.get(HemisphericRegion::Southern)),
            _ => Err(PyValueError::new_err(format!(
                "Invalid region index: {}. Must be 0-1.",
                region
            ))),
        }
    }

    /// Set value by region index
    fn set(&mut self, region: usize, value: FloatValue) -> PyResult<()> {
        match region {
            0 => self.0.set(HemisphericRegion::Northern, value),
            1 => self.0.set(HemisphericRegion::Southern, value),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid region index: {}. Must be 0-1.",
                    region
                )))
            }
        }
        Ok(())
    }

    /// Convert to numpy array
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<FloatValue>> {
        self.0.as_array().to_pyarray(py)
    }

    /// Convert to list
    fn to_list(&self) -> Vec<FloatValue> {
        self.0.to_vec()
    }

    /// Convert to dict with region names as keys
    fn to_dict(&self) -> std::collections::HashMap<String, FloatValue> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "northern".to_string(),
            self.0.get(HemisphericRegion::Northern),
        );
        map.insert(
            "southern".to_string(),
            self.0.get(HemisphericRegion::Southern),
        );
        map
    }

    fn __repr__(&self) -> String {
        format!(
            "HemisphericSlice(northern={}, southern={})",
            self.0.get(HemisphericRegion::Northern),
            self.0.get(HemisphericRegion::Southern)
        )
    }

    fn __getitem__(&self, index: usize) -> PyResult<FloatValue> {
        self.get(index)
    }

    fn __setitem__(&mut self, index: usize, value: FloatValue) -> PyResult<()> {
        self.set(index, value)
    }

    fn __len__(&self) -> usize {
        2
    }
}

// =============================================================================
// Python TimeseriesWindow
// =============================================================================

/// Python wrapper for TimeseriesWindow
///
/// Provides zero-cost view-like access to scalar timeseries data.
/// Since Python can't handle Rust lifetimes, this stores a copy of the
/// values and current position.
///
/// Example:
///     window = TimeseriesWindow(values=[1.0, 2.0, 3.0], current_index=2)
///     print(window.current)  # 3.0
///     print(window.previous)  # 2.0
#[pyclass]
#[pyo3(name = "TimeseriesWindow")]
#[derive(Debug, Clone)]
pub struct PyTimeseriesWindow {
    values: Vec<FloatValue>,
    current_index: usize,
}

#[pymethods]
impl PyTimeseriesWindow {
    #[new]
    #[pyo3(signature = (values, current_index))]
    pub fn new(values: Vec<FloatValue>, current_index: usize) -> PyResult<Self> {
        if current_index >= values.len() && !values.is_empty() {
            return Err(PyValueError::new_err(format!(
                "current_index {} out of bounds for values of length {}",
                current_index,
                values.len()
            )));
        }
        Ok(Self {
            values,
            current_index,
        })
    }

    /// Get the current value (at current_index)
    #[getter]
    fn current(&self) -> PyResult<FloatValue> {
        self.values
            .get(self.current_index)
            .copied()
            .ok_or_else(|| PyValueError::new_err("No current value available"))
    }

    /// Get the previous value (at current_index - 1)
    #[getter]
    fn previous(&self) -> PyResult<FloatValue> {
        if self.current_index == 0 {
            return Err(PyValueError::new_err("No previous value available"));
        }
        self.values
            .get(self.current_index - 1)
            .copied()
            .ok_or_else(|| PyValueError::new_err("No previous value available"))
    }

    /// Get value at offset from current (negative = past, positive = future)
    fn at_offset(&self, offset: isize) -> PyResult<FloatValue> {
        let index = self.current_index as isize + offset;
        if index < 0 || index >= self.values.len() as isize {
            return Err(PyValueError::new_err(format!(
                "Offset {} results in index {} which is out of bounds",
                offset, index
            )));
        }
        Ok(self.values[index as usize])
    }

    /// Get last n values as numpy array (including current)
    fn last_n<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<FloatValue>>> {
        if n == 0 {
            return Ok(Vec::<FloatValue>::new().to_pyarray(py));
        }
        let start = if n > self.current_index + 1 {
            0
        } else {
            self.current_index + 1 - n
        };
        let end = self.current_index + 1;
        let slice: Vec<FloatValue> = self.values[start..end].to_vec();
        Ok(slice.to_pyarray(py))
    }

    /// Get all values as numpy array
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<FloatValue>> {
        self.values.clone().to_pyarray(py)
    }

    /// Get the current index
    #[getter]
    fn current_index(&self) -> usize {
        self.current_index
    }

    /// Get the number of values
    fn __len__(&self) -> usize {
        self.values.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "TimeseriesWindow(len={}, current_index={}, current={})",
            self.values.len(),
            self.current_index,
            self.values
                .get(self.current_index)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "N/A".to_string())
        )
    }
}

/// Python wrapper for GridTimeseriesWindow (FourBox)
///
/// Provides view-like access to grid timeseries data with regional access.
///
/// Example:
///     window = FourBoxTimeseriesWindow(values=[[1,2,3,4], [5,6,7,8]], current_index=1)
///     print(window.current)  # [5, 6, 7, 8]
///     print(window.region(0))  # TimeseriesWindow for northern ocean
#[pyclass]
#[pyo3(name = "FourBoxTimeseriesWindow")]
#[derive(Debug, Clone)]
pub struct PyFourBoxTimeseriesWindow {
    /// Values stored as [timestep][region]
    values: Vec<[FloatValue; 4]>,
    current_index: usize,
}

#[pymethods]
impl PyFourBoxTimeseriesWindow {
    #[new]
    #[pyo3(signature = (values, current_index))]
    pub fn new(values: Vec<[FloatValue; 4]>, current_index: usize) -> PyResult<Self> {
        if current_index >= values.len() && !values.is_empty() {
            return Err(PyValueError::new_err(format!(
                "current_index {} out of bounds for values of length {}",
                current_index,
                values.len()
            )));
        }
        Ok(Self {
            values,
            current_index,
        })
    }

    /// Get the current slice as FourBoxSlice
    #[getter]
    fn current(&self) -> PyResult<PyFourBoxSlice> {
        self.values
            .get(self.current_index)
            .map(|v| PyFourBoxSlice(FourBoxSlice::from_array(*v)))
            .ok_or_else(|| PyValueError::new_err("No current value available"))
    }

    /// Get the previous slice as FourBoxSlice
    #[getter]
    fn previous(&self) -> PyResult<PyFourBoxSlice> {
        if self.current_index == 0 {
            return Err(PyValueError::new_err("No previous value available"));
        }
        self.values
            .get(self.current_index - 1)
            .map(|v| PyFourBoxSlice(FourBoxSlice::from_array(*v)))
            .ok_or_else(|| PyValueError::new_err("No previous value available"))
    }

    /// Get a single region's timeseries as a TimeseriesWindow
    fn region(&self, region: usize) -> PyResult<PyTimeseriesWindow> {
        if region >= 4 {
            return Err(PyValueError::new_err(format!(
                "Invalid region index: {}. Must be 0-3.",
                region
            )));
        }
        let values: Vec<FloatValue> = self.values.iter().map(|v| v[region]).collect();
        Ok(PyTimeseriesWindow {
            values,
            current_index: self.current_index,
        })
    }

    /// Get current index
    #[getter]
    fn current_index(&self) -> usize {
        self.current_index
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "FourBoxTimeseriesWindow(len={}, current_index={})",
            self.values.len(),
            self.current_index
        )
    }
}

/// Python wrapper for GridTimeseriesWindow (Hemispheric)
#[pyclass]
#[pyo3(name = "HemisphericTimeseriesWindow")]
#[derive(Debug, Clone)]
pub struct PyHemisphericTimeseriesWindow {
    values: Vec<[FloatValue; 2]>,
    current_index: usize,
}

#[pymethods]
impl PyHemisphericTimeseriesWindow {
    #[new]
    #[pyo3(signature = (values, current_index))]
    pub fn new(values: Vec<[FloatValue; 2]>, current_index: usize) -> PyResult<Self> {
        if current_index >= values.len() && !values.is_empty() {
            return Err(PyValueError::new_err(format!(
                "current_index {} out of bounds for values of length {}",
                current_index,
                values.len()
            )));
        }
        Ok(Self {
            values,
            current_index,
        })
    }

    /// Get the current slice as HemisphericSlice
    #[getter]
    fn current(&self) -> PyResult<PyHemisphericSlice> {
        self.values
            .get(self.current_index)
            .map(|v| PyHemisphericSlice(HemisphericSlice::from_array(*v)))
            .ok_or_else(|| PyValueError::new_err("No current value available"))
    }

    /// Get the previous slice as HemisphericSlice
    #[getter]
    fn previous(&self) -> PyResult<PyHemisphericSlice> {
        if self.current_index == 0 {
            return Err(PyValueError::new_err("No previous value available"));
        }
        self.values
            .get(self.current_index - 1)
            .map(|v| PyHemisphericSlice(HemisphericSlice::from_array(*v)))
            .ok_or_else(|| PyValueError::new_err("No previous value available"))
    }

    /// Get a single region's timeseries as a TimeseriesWindow
    fn region(&self, region: usize) -> PyResult<PyTimeseriesWindow> {
        if region >= 2 {
            return Err(PyValueError::new_err(format!(
                "Invalid region index: {}. Must be 0-1.",
                region
            )));
        }
        let values: Vec<FloatValue> = self.values.iter().map(|v| v[region]).collect();
        Ok(PyTimeseriesWindow {
            values,
            current_index: self.current_index,
        })
    }

    /// Get current index
    #[getter]
    fn current_index(&self) -> usize {
        self.current_index
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "HemisphericTimeseriesWindow(len={}, current_index={})",
            self.values.len(),
            self.current_index
        )
    }
}

// =============================================================================
// GridType Python Enum
// =============================================================================

/// Re-export GridType for Python
pub use crate::component::GridType;

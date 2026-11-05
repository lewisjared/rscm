//! Python bindings for typed state access
//!
//! This module provides Python wrappers for TimeseriesWindow, GridTimeseriesWindow,
//! and typed output slices (FourBoxSlice, HemisphericSlice).

use crate::component::TimeseriesType;
use crate::spatial::{FourBoxRegion, HemisphericRegion};
use crate::state::{FourBoxSlice, HemisphericSlice, ReadIndex, StateValue, VariableSource};
use crate::timeseries::{FloatValue, Time};
use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{pymodule, Bound, PyResult};

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
// Python StateValue
// =============================================================================

/// Python wrapper for StateValue enum
///
/// Represents a value that can be scalar or spatially-resolved (FourBox/Hemispheric).
///
/// Example:
///     scalar = StateValue.scalar(15.0)
///     four_box = StateValue.four_box(FourBoxSlice.uniform(10.0))
///     hemispheric = StateValue.hemispheric(HemisphericSlice(northern=15.0, southern=10.0))
#[pyclass]
#[pyo3(name = "StateValue")]
#[derive(Debug, Clone)]
pub struct PyStateValue(pub StateValue);

#[pymethods]
impl PyStateValue {
    /// Create a scalar StateValue
    #[staticmethod]
    fn scalar(value: FloatValue) -> Self {
        Self(StateValue::Scalar(value))
    }

    /// Create a FourBox StateValue from a FourBoxSlice
    #[staticmethod]
    fn four_box(slice: PyFourBoxSlice) -> Self {
        Self(StateValue::FourBox(slice.0))
    }

    /// Create a Hemispheric StateValue from a HemisphericSlice
    #[staticmethod]
    fn hemispheric(slice: PyHemisphericSlice) -> Self {
        Self(StateValue::Hemispheric(slice.0))
    }

    /// Check if this is a scalar value
    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }

    /// Check if this is a FourBox grid value
    fn is_four_box(&self) -> bool {
        self.0.is_four_box()
    }

    /// Check if this is a Hemispheric grid value
    fn is_hemispheric(&self) -> bool {
        self.0.is_hemispheric()
    }

    /// Get the scalar value if this is a Scalar variant, otherwise None
    fn as_scalar(&self) -> Option<FloatValue> {
        self.0.as_scalar()
    }

    /// Get the FourBoxSlice if this is a FourBox variant, otherwise None
    fn as_four_box(&self) -> Option<PyFourBoxSlice> {
        self.0.as_four_box().map(|s| PyFourBoxSlice(*s))
    }

    /// Get the HemisphericSlice if this is a Hemispheric variant, otherwise None
    fn as_hemispheric(&self) -> Option<PyHemisphericSlice> {
        self.0.as_hemispheric().map(|s| PyHemisphericSlice(*s))
    }

    /// Convert to a scalar value, aggregating grid values if necessary
    ///
    /// For Scalar: returns the value directly
    /// For FourBox: returns the arithmetic mean of all 4 regional values
    ///              (sum of all regions divided by 4, giving equal weight to each region)
    /// For Hemispheric: returns the arithmetic mean of both hemispheres
    ///                  (sum of northern and southern divided by 2, giving equal weight to each)
    fn to_scalar(&self) -> FloatValue {
        self.0.to_scalar()
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            StateValue::Scalar(v) => format!("StateValue.scalar({})", v),
            StateValue::FourBox(slice) => {
                format!(
                    "StateValue.four_box(FourBoxSlice(northern_ocean={}, northern_land={}, southern_ocean={}, southern_land={}))",
                    slice.get(FourBoxRegion::NorthernOcean),
                    slice.get(FourBoxRegion::NorthernLand),
                    slice.get(FourBoxRegion::SouthernOcean),
                    slice.get(FourBoxRegion::SouthernLand)
                )
            }
            StateValue::Hemispheric(slice) => {
                format!(
                    "StateValue.hemispheric(HemisphericSlice(northern={}, southern={}))",
                    slice.get(HemisphericRegion::Northern),
                    slice.get(HemisphericRegion::Southern)
                )
            }
        }
    }
}

// =============================================================================
// Shared read logic of the Python windows
// =============================================================================

/// What a Python window knows beyond its values: their times, source and point or average type.
#[derive(Debug, Clone)]
pub struct WindowMeta {
    times: Option<Vec<Time>>,
    source: VariableSource,
    timeseries_type: TimeseriesType,
}

impl WindowMeta {
    pub fn new(
        times: Option<Vec<Time>>,
        source: VariableSource,
        timeseries_type: TimeseriesType,
        len: usize,
    ) -> PyResult<Self> {
        if let Some(times) = &times {
            if times.len() != len {
                return Err(PyValueError::new_err(format!(
                    "times has length {} but values has length {}",
                    times.len(),
                    len
                )));
            }
        }
        Ok(Self {
            times,
            source,
            timeseries_type,
        })
    }

    fn read_index(&self, current: usize, len: usize) -> ReadIndex {
        ReadIndex {
            current,
            len,
            source: self.source,
            timeseries_type: self.timeseries_type,
        }
    }

    fn times(&self) -> PyResult<&[Time]> {
        self.times
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("This window was built without times"))
    }

    fn time_at(&self, index: usize) -> PyResult<Time> {
        self.times()?
            .get(index)
            .copied()
            .ok_or_else(|| PyValueError::new_err("No value available at current index"))
    }

    fn interval(&self, current: usize, len: usize) -> PyResult<(usize, usize)> {
        self.read_index(current, len)
            .interval()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Value at `t` from `value(index)`, held beyond the ends.
    ///
    /// Between nodes, averages hold the earlier value and points interpolate linearly.
    fn interpolate<T>(
        &self,
        t: Time,
        value: impl Fn(usize) -> T,
        lerp: impl Fn(T, T, f64) -> T,
    ) -> PyResult<T> {
        let times = self.times()?;
        let last = times
            .len()
            .checked_sub(1)
            .ok_or_else(|| PyValueError::new_err("Window has no values"))?;
        let after = times.partition_point(|&node| node <= t);
        if after == 0 {
            return Ok(value(0));
        }
        let before = after - 1;
        if before == last || times[before] == t || self.timeseries_type == TimeseriesType::Average {
            return Ok(value(before));
        }
        let fraction = (t - times[before]) / (times[after] - times[before]);
        Ok(lerp(value(before), value(after), fraction))
    }
}

fn lerp(start: FloatValue, end: FloatValue, fraction: f64) -> FloatValue {
    start + (end - start) * fraction
}

fn lerp_regions<const N: usize>(
    start: [FloatValue; N],
    end: [FloatValue; N],
    fraction: f64,
) -> [FloatValue; N] {
    std::array::from_fn(|i| lerp(start[i], end[i], fraction))
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
///     window = TimeseriesWindow(values=[1.0, 2.0, 3.0], current_index=1)
///     print(window.get())  # 2.0
///     print(window.history(1))  # 1.0
#[pyclass]
#[pyo3(name = "TimeseriesWindow")]
#[derive(Debug, Clone)]
pub struct PyTimeseriesWindow {
    values: Vec<FloatValue>,
    current_index: usize,
    meta: WindowMeta,
}

#[pymethods]
impl PyTimeseriesWindow {
    #[new]
    #[pyo3(signature = (
        values,
        current_index,
        *,
        times=None,
        source=VariableSource::Exogenous,
        timeseries_type=TimeseriesType::Point,
    ))]
    pub fn new(
        values: Vec<FloatValue>,
        current_index: usize,
        times: Option<Vec<Time>>,
        source: VariableSource,
        timeseries_type: TimeseriesType,
    ) -> PyResult<Self> {
        if current_index >= values.len() && !values.is_empty() {
            return Err(PyValueError::new_err(format!(
                "current_index {} out of bounds for values of length {}",
                current_index,
                values.len()
            )));
        }
        let meta = WindowMeta::new(times, source, timeseries_type, values.len())?;
        Ok(Self {
            values,
            current_index,
            meta,
        })
    }

    /// Value at the index the model reads for this input.
    ///
    /// A point reads index N+1, or N for own state and a lagged edge.
    /// An average reads index N, or N-1 for own state.
    fn get(&self) -> PyResult<FloatValue> {
        self.history(0)
            .ok_or_else(|| PyValueError::new_err("No value available at current index"))
    }

    /// Time of the value `get()` returns.
    fn get_time(&self) -> PyResult<Time> {
        self.meta.time_at(self.index().get())
    }

    /// Value `n` steps before the one `get()` returns, or None before the first step.
    fn history(&self, n: usize) -> Option<FloatValue> {
        self.index()
            .history(n)
            .and_then(|i| self.values.get(i).copied())
    }

    /// Number of values `history` can read.
    fn history_len(&self) -> usize {
        self.index().history_len()
    }

    /// Values at the start and end of the step.
    ///
    /// Raises ValueError on an average and on a component's own state.
    fn interval(&self) -> PyResult<(FloatValue, FloatValue)> {
        let (start, end) = self.meta.interval(self.current_index, self.values.len())?;
        Ok((self.values[start], self.values[end]))
    }

    /// Value at time `t`, held at the first or last value beyond the window.
    fn interpolate(&self, t: Time) -> PyResult<FloatValue> {
        self.meta.interpolate(t, |i| self.values[i], lerp)
    }

    /// Source of the variable, which decides the index `get()` reads.
    #[getter]
    fn source(&self) -> VariableSource {
        self.meta.source
    }

    /// Whether the variable is a point or an average.
    #[getter]
    fn timeseries_type(&self) -> TimeseriesType {
        self.meta.timeseries_type
    }

    /// Get all values as numpy array
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<FloatValue>> {
        self.values.clone().to_pyarray(py)
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

impl PyTimeseriesWindow {
    fn index(&self) -> ReadIndex {
        self.meta.read_index(self.current_index, self.values.len())
    }
}

/// Python wrapper for GridTimeseriesWindow (FourBox)
///
/// Provides view-like access to grid timeseries data with regional access.
///
/// Example:
///     window = FourBoxTimeseriesWindow(values=\[\[1,2,3,4\], \[5,6,7,8\]\], current_index=1)
///     print(window.get())  # FourBoxSlice of [5, 6, 7, 8]
///     print(window.region(0))  # TimeseriesWindow for northern ocean
#[pyclass]
#[pyo3(name = "FourBoxTimeseriesWindow")]
#[derive(Debug, Clone)]
pub struct PyFourBoxTimeseriesWindow {
    /// Values stored as [timestep][region]
    values: Vec<[FloatValue; 4]>,
    current_index: usize,
    meta: WindowMeta,
}

#[pymethods]
impl PyFourBoxTimeseriesWindow {
    #[new]
    #[pyo3(signature = (
        values,
        current_index,
        *,
        times=None,
        source=VariableSource::Exogenous,
        timeseries_type=TimeseriesType::Point,
    ))]
    pub fn new(
        values: Vec<[FloatValue; 4]>,
        current_index: usize,
        times: Option<Vec<Time>>,
        source: VariableSource,
        timeseries_type: TimeseriesType,
    ) -> PyResult<Self> {
        if current_index >= values.len() && !values.is_empty() {
            return Err(PyValueError::new_err(format!(
                "current_index {} out of bounds for values of length {}",
                current_index,
                values.len()
            )));
        }
        let meta = WindowMeta::new(times, source, timeseries_type, values.len())?;
        Ok(Self {
            values,
            current_index,
            meta,
        })
    }

    /// All regional values at the index the model reads for this input.
    ///
    /// A point reads index N+1, or N for own state and a lagged edge.
    /// An average reads index N, or N-1 for own state.
    fn get(&self) -> PyResult<PyFourBoxSlice> {
        self.history(0)
            .ok_or_else(|| PyValueError::new_err("No value available at current index"))
    }

    /// Time of the values `get()` returns.
    fn get_time(&self) -> PyResult<Time> {
        self.meta.time_at(self.index().get())
    }

    /// Regional values `n` steps before those `get()` returns, or None before the first step.
    fn history(&self, n: usize) -> Option<PyFourBoxSlice> {
        self.index()
            .history(n)
            .and_then(|i| self.values.get(i))
            .map(|v| PyFourBoxSlice(FourBoxSlice::from_array(*v)))
    }

    /// Number of values `history` can read.
    fn history_len(&self) -> usize {
        self.index().history_len()
    }

    /// Regional values at the start and end of the step.
    ///
    /// Raises ValueError on an average and on a component's own state.
    fn interval(&self) -> PyResult<(PyFourBoxSlice, PyFourBoxSlice)> {
        let (start, end) = self.meta.interval(self.current_index, self.values.len())?;
        Ok((
            PyFourBoxSlice(FourBoxSlice::from_array(self.values[start])),
            PyFourBoxSlice(FourBoxSlice::from_array(self.values[end])),
        ))
    }

    /// Regional values at time `t`, held at the first or last values beyond the window.
    fn interpolate(&self, t: Time) -> PyResult<PyFourBoxSlice> {
        self.meta
            .interpolate(t, |i| self.values[i], lerp_regions)
            .map(|v| PyFourBoxSlice(FourBoxSlice::from_array(v)))
    }

    /// Source of the variable, which decides the index `get()` reads.
    #[getter]
    fn source(&self) -> VariableSource {
        self.meta.source
    }

    /// Whether the variable is a point or an average.
    #[getter]
    fn timeseries_type(&self) -> TimeseriesType {
        self.meta.timeseries_type
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
            meta: self.meta.clone(),
        })
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

impl PyFourBoxTimeseriesWindow {
    fn index(&self) -> ReadIndex {
        self.meta.read_index(self.current_index, self.values.len())
    }
}

/// Python wrapper for GridTimeseriesWindow (Hemispheric)
#[pyclass]
#[pyo3(name = "HemisphericTimeseriesWindow")]
#[derive(Debug, Clone)]
pub struct PyHemisphericTimeseriesWindow {
    values: Vec<[FloatValue; 2]>,
    current_index: usize,
    meta: WindowMeta,
}

#[pymethods]
impl PyHemisphericTimeseriesWindow {
    #[new]
    #[pyo3(signature = (
        values,
        current_index,
        *,
        times=None,
        source=VariableSource::Exogenous,
        timeseries_type=TimeseriesType::Point,
    ))]
    pub fn new(
        values: Vec<[FloatValue; 2]>,
        current_index: usize,
        times: Option<Vec<Time>>,
        source: VariableSource,
        timeseries_type: TimeseriesType,
    ) -> PyResult<Self> {
        if current_index >= values.len() && !values.is_empty() {
            return Err(PyValueError::new_err(format!(
                "current_index {} out of bounds for values of length {}",
                current_index,
                values.len()
            )));
        }
        let meta = WindowMeta::new(times, source, timeseries_type, values.len())?;
        Ok(Self {
            values,
            current_index,
            meta,
        })
    }

    /// All regional values at the index the model reads for this input.
    ///
    /// A point reads index N+1, or N for own state and a lagged edge.
    /// An average reads index N, or N-1 for own state.
    fn get(&self) -> PyResult<PyHemisphericSlice> {
        self.history(0)
            .ok_or_else(|| PyValueError::new_err("No value available at current index"))
    }

    /// Time of the values `get()` returns.
    fn get_time(&self) -> PyResult<Time> {
        self.meta.time_at(self.index().get())
    }

    /// Regional values `n` steps before those `get()` returns, or None before the first step.
    fn history(&self, n: usize) -> Option<PyHemisphericSlice> {
        self.index()
            .history(n)
            .and_then(|i| self.values.get(i))
            .map(|v| PyHemisphericSlice(HemisphericSlice::from_array(*v)))
    }

    /// Number of values `history` can read.
    fn history_len(&self) -> usize {
        self.index().history_len()
    }

    /// Regional values at the start and end of the step.
    ///
    /// Raises ValueError on an average and on a component's own state.
    fn interval(&self) -> PyResult<(PyHemisphericSlice, PyHemisphericSlice)> {
        let (start, end) = self.meta.interval(self.current_index, self.values.len())?;
        Ok((
            PyHemisphericSlice(HemisphericSlice::from_array(self.values[start])),
            PyHemisphericSlice(HemisphericSlice::from_array(self.values[end])),
        ))
    }

    /// Regional values at time `t`, held at the first or last values beyond the window.
    fn interpolate(&self, t: Time) -> PyResult<PyHemisphericSlice> {
        self.meta
            .interpolate(t, |i| self.values[i], lerp_regions)
            .map(|v| PyHemisphericSlice(HemisphericSlice::from_array(v)))
    }

    /// Source of the variable, which decides the index `get()` reads.
    #[getter]
    fn source(&self) -> VariableSource {
        self.meta.source
    }

    /// Whether the variable is a point or an average.
    #[getter]
    fn timeseries_type(&self) -> TimeseriesType {
        self.meta.timeseries_type
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
            meta: self.meta.clone(),
        })
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

impl PyHemisphericTimeseriesWindow {
    fn index(&self) -> ReadIndex {
        self.meta.read_index(self.current_index, self.values.len())
    }
}

// =============================================================================
// GridType Python Enum
// =============================================================================

/// Re-export GridType for Python
pub use crate::component::GridType;

#[pymodule]
pub fn state(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFourBoxSlice>()?;
    m.add_class::<PyHemisphericSlice>()?;
    m.add_class::<PyStateValue>()?;
    m.add_class::<PyTimeseriesWindow>()?;
    m.add_class::<PyFourBoxTimeseriesWindow>()?;
    m.add_class::<PyHemisphericTimeseriesWindow>()?;
    m.add_class::<VariableSource>()?;
    Ok(())
}

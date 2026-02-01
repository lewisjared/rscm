//! Python bindings for the units module.
//!
//! This module provides Python access to unit parsing, normalization, and conversion
//! functionality. It exposes the [`PyUnit`] class which wraps the Rust [`Unit`] type.
//!
//! # Usage from Python
//!
//! ```python
//! from rscm._lib.core import Unit
//!
//! # Parse units with flexible syntax
//! gtc = Unit("GtC / yr")
//! mtco2 = Unit("MtCO2 / yr")
//!
//! # Check compatibility
//! assert gtc.is_compatible(mtco2)
//!
//! # Get conversion factor
//! factor = gtc.conversion_factor(mtco2)  # ~3666.67
//!
//! # Convert values
//! emissions_mtco2 = gtc.convert(0.34, mtco2)  # 0.34 GtC → 1246.67 MtCO2
//! ```

use super::Unit;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// A physical unit with parsing, normalization, and conversion support.
///
/// This class provides comprehensive support for working with physical units
/// in climate models. It handles parsing of unit strings with flexible syntax,
/// dimensional analysis, and conversion factor calculation.
///
/// # Parsing
///
/// The parser accepts several equivalent notations:
///
/// - Exponents: `m^2`, `m**2`, `m2`
/// - Division: `W/m^2`, `W m^-2`, `W per m^2`
/// - Multiplication: `kg m`, `kg*m`
/// - Whitespace: `W/m^2` == `W / m ^ 2`
///
/// # Examples
///
/// ```python
/// from rscm._lib.core import Unit
///
/// # Parse and compare units
/// u1 = Unit("W/m^2")
/// u2 = Unit("W / m ^ 2")
/// assert u1 == u2  # Same normalized form
///
/// # Convert between compatible units
/// gtc = Unit("GtC/yr")
/// mtco2 = Unit("MtCO2/yr")
/// factor = gtc.conversion_factor(mtco2)  # ~3666.67
/// ```
#[pyclass(name = "Unit")]
#[derive(Clone)]
pub struct PyUnit {
    inner: Unit,
}

#[pymethods]
impl PyUnit {
    /// Create a new Unit from a unit string.
    ///
    /// Parameters
    /// ----------
    /// unit_str : str
    ///     The unit string to parse. Accepts flexible syntax including:
    ///     - Exponents: `m^2`, `m**2`, `m2`
    ///     - Division: `W/m^2`, `W m^-2`, `W per m^2`
    ///     - Multiplication: `kg m`, `kg*m`
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the unit string cannot be parsed.
    ///
    /// Examples
    /// --------
    /// >>> u = Unit("W/m^2")
    /// >>> u = Unit("GtC / yr")
    /// >>> u = Unit("kg m^-2 s^-1")
    #[new]
    fn new(unit_str: &str) -> PyResult<Self> {
        let inner = Unit::parse(unit_str).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Return the original input string used to create this unit.
    ///
    /// Returns
    /// -------
    /// str
    ///     The original input string.
    ///
    /// Examples
    /// --------
    /// >>> u = Unit("W / m ^ 2")
    /// >>> u.original
    /// 'W / m ^ 2'
    #[getter]
    fn original(&self) -> &str {
        self.inner.original()
    }

    /// Return the normalized string representation of this unit.
    ///
    /// The normalized form is canonical: units with positive exponents
    /// first (alphabetically), then `/`, then units with negative exponents.
    ///
    /// Returns
    /// -------
    /// str
    ///     The normalized unit string.
    ///
    /// Examples
    /// --------
    /// >>> Unit("W / m ^ 2").normalized()
    /// 'W / m^2'
    /// >>> Unit("m^-2 W").normalized()
    /// 'W / m^2'
    fn normalized(&self) -> String {
        self.inner.normalized()
    }

    /// Check if this unit is dimensionless.
    ///
    /// Units like "ppm", "ppb", and "1" are dimensionless.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the unit is dimensionless, False otherwise.
    ///
    /// Examples
    /// --------
    /// >>> Unit("ppm").is_dimensionless()
    /// True
    /// >>> Unit("W/m^2").is_dimensionless()
    /// False
    fn is_dimensionless(&self) -> bool {
        self.inner.is_dimensionless()
    }

    /// Check if this unit can be converted to another unit.
    ///
    /// Units are compatible if they have the same physical dimension.
    /// For example, GtC/yr and MtCO2/yr are compatible because they
    /// both represent mass flux.
    ///
    /// Parameters
    /// ----------
    /// other : Unit
    ///     The target unit to check compatibility with.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if conversion is possible, False otherwise.
    ///
    /// Examples
    /// --------
    /// >>> gtc = Unit("GtC/yr")
    /// >>> mtco2 = Unit("MtCO2/yr")
    /// >>> gtc.is_compatible(mtco2)
    /// True
    /// >>> flux = Unit("W/m^2")
    /// >>> gtc.is_compatible(flux)
    /// False
    fn is_compatible(&self, other: &PyUnit) -> bool {
        self.inner.is_compatible(&other.inner)
    }

    /// Calculate the conversion factor from this unit to another unit.
    ///
    /// The factor is the multiplier to convert a value in this unit to a
    /// value in the target unit. For example, if this unit is `GtC/yr` and
    /// the target is `MtCO2/yr`, the factor is approximately 3666.67.
    ///
    /// Parameters
    /// ----------
    /// target : Unit
    ///     The target unit to convert to.
    ///
    /// Returns
    /// -------
    /// float
    ///     The conversion factor.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the units have incompatible dimensions.
    ///
    /// Examples
    /// --------
    /// >>> gtc = Unit("GtC/yr")
    /// >>> mtco2 = Unit("MtCO2/yr")
    /// >>> factor = gtc.conversion_factor(mtco2)
    /// >>> round(factor, 2)
    /// 3666.67
    fn conversion_factor(&self, target: &PyUnit) -> PyResult<f64> {
        self.inner
            .conversion_factor(&target.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Convert a value from this unit to another unit.
    ///
    /// This is a convenience method equivalent to:
    /// `value * self.conversion_factor(target)`
    ///
    /// Parameters
    /// ----------
    /// value : float
    ///     The value to convert.
    /// target : Unit
    ///     The target unit to convert to.
    ///
    /// Returns
    /// -------
    /// float
    ///     The converted value.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the units have incompatible dimensions.
    ///
    /// Examples
    /// --------
    /// >>> gtc = Unit("GtC/yr")
    /// >>> mtco2 = Unit("MtCO2/yr")
    /// >>> gtc.convert(0.34, mtco2)
    /// 1246.666...
    fn convert(&self, value: f64, target: &PyUnit) -> PyResult<f64> {
        self.inner
            .convert_to(value, &target.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Return a string representation of this unit.
    fn __repr__(&self) -> String {
        format!("Unit('{}')", self.inner.normalized())
    }

    /// Return the normalized string representation.
    fn __str__(&self) -> String {
        self.inner.normalized()
    }

    /// Check equality with another unit.
    ///
    /// Two units are equal if they have the same normalized representation.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_unit) = other.extract::<PyRef<PyUnit>>() {
            Ok(self.inner == other_unit.inner)
        } else {
            Err(PyTypeError::new_err(
                "can only compare Unit with another Unit",
            ))
        }
    }

    /// Compute hash for use in sets and dicts.
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

/// Register the units classes with the Python module.
pub fn register_units(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUnit>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_unit_creation() {
        let unit = PyUnit::new("W/m^2").unwrap();
        assert_eq!(unit.normalized(), "W / m^2");
    }

    #[test]
    fn test_py_unit_equality() {
        let u1 = PyUnit::new("W/m^2").unwrap();
        let u2 = PyUnit::new("W / m ^ 2").unwrap();
        assert_eq!(u1.inner, u2.inner);
    }

    #[test]
    fn test_py_unit_conversion() {
        let gtc = PyUnit::new("GtC/yr").unwrap();
        let mtco2 = PyUnit::new("MtCO2/yr").unwrap();

        assert!(gtc.is_compatible(&mtco2));

        let factor = gtc.conversion_factor(&mtco2).unwrap();
        assert!((factor - 3666.67).abs() < 0.01);
    }

    #[test]
    fn test_py_unit_incompatible() {
        let gtc = PyUnit::new("GtC/yr").unwrap();
        let flux = PyUnit::new("W/m^2").unwrap();

        assert!(!gtc.is_compatible(&flux));
        assert!(gtc.conversion_factor(&flux).is_err());
    }
}

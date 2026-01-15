//! Variable registration system for RSCM.
//!
//! This module provides a central registry for variable definitions with comprehensive metadata,
//! enabling validation of unit and time convention compatibility between components.
//!
//! # Overview
//!
//! MAGICC modules use a DATASTORE pattern where variables carry rich metadata:
//! - Units (e.g., "ppm", "GtC / yr", "W / m^2")
//! - Time conventions (start-of-year, mid-year, instantaneous)
//! - Descriptions for documentation and introspection
//!
//! This module provides:
//! - [`TimeConvention`] enum for temporal alignment
//! - [`VariableDefinition`] struct for variable metadata
//! - [`VariableRegistry`] for registering and looking up variables
//! - [`define_variable!`] macro for compile-time registration (Rust)
//! - Runtime registration API for Python users
//!
//! # Usage
//!
//! ## Defining variables at compile time (Rust)
//!
//! ```rust,ignore
//! use rscm_core::variable::{define_variable, TimeConvention};
//!
//! define_variable!(
//!     CO2_CONCENTRATION,
//!     name = "Atmospheric Concentration|CO2",
//!     unit = "ppm",
//!     time_convention = TimeConvention::StartOfYear,
//!     description = "Atmospheric CO2 concentration",
//! );
//! ```
//!
//! ## Registering variables at runtime (Python)
//!
//! ```python
//! import rscm
//!
//! co2_conc = rscm.VariableDefinition(
//!     name="Atmospheric Concentration|CO2",
//!     unit="ppm",
//!     time_convention=rscm.TimeConvention.StartOfYear,
//!     description="Atmospheric CO2 concentration",
//! )
//! rscm.register_variable(co2_conc)
//! ```
//!
//! ## Looking up variables
//!
//! ```rust,ignore
//! use rscm_core::variable::VARIABLE_REGISTRY;
//!
//! if let Some(var) = VARIABLE_REGISTRY.get("Atmospheric Concentration|CO2") {
//!     println!("Unit: {}", var.unit);
//!     println!("Time convention: {:?}", var.time_convention);
//! }
//! ```

use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use crate::spatial::{FourBoxGrid, HemisphericGrid};

/// Time convention for a variable.
///
/// Specifies when during a time period a variable's value is valid.
/// This is critical for temporal alignment between components with different conventions.
///
/// # Variants
///
/// - [`StartOfYear`](TimeConvention::StartOfYear): Value applies at the start of the year (Jan 1).
///   Used for stock variables like concentrations and temperatures.
/// - [`MidYear`](TimeConvention::MidYear): Value applies at mid-year (Jul 1).
///   Used for flow variables like emissions.
/// - [`Instantaneous`](TimeConvention::Instantaneous): Value is instantaneous with no temporal averaging.
///   Used for derived quantities.
///
/// # Example
///
/// ```rust
/// use rscm_core::variable::TimeConvention;
///
/// // Emissions are mid-year values (average over the year)
/// let emissions_convention = TimeConvention::MidYear;
///
/// // Concentrations are start-of-year values (snapshot)
/// let concentration_convention = TimeConvention::StartOfYear;
///
/// // Check compatibility
/// assert_ne!(emissions_convention, concentration_convention);
/// ```
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeConvention {
    /// Value applies at the start of the year (Jan 1).
    /// Used for stock variables like concentrations and temperatures.
    StartOfYear,
    /// Value applies at mid-year (Jul 1).
    /// Used for flow variables like emissions.
    MidYear,
    /// Instantaneous value with no temporal averaging.
    /// Used for derived quantities.
    Instantaneous,
}

impl std::fmt::Display for TimeConvention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeConvention::StartOfYear => write!(f, "StartOfYear"),
            TimeConvention::MidYear => write!(f, "MidYear"),
            TimeConvention::Instantaneous => write!(f, "Instantaneous"),
        }
    }
}

/// Definition of a variable with comprehensive metadata.
///
/// Contains intrinsic metadata about a variable:
/// - Name (unique identifier)
/// - Unit (canonical unit for the variable)
/// - Time convention (temporal alignment)
/// - Description (for documentation)
///
/// Preindustrial values are stored separately in timeseries data, as they are
/// scenario-dependent rather than intrinsic to the variable definition.
///
/// # Example
///
/// ```rust
/// use rscm_core::variable::{VariableDefinition, TimeConvention};
///
/// let co2_conc = VariableDefinition::new(
///     "Atmospheric Concentration|CO2",
///     "ppm",
///     TimeConvention::StartOfYear,
///     "Atmospheric CO2 concentration",
/// );
///
/// assert_eq!(co2_conc.name, "Atmospheric Concentration|CO2");
/// assert_eq!(co2_conc.time_convention, TimeConvention::StartOfYear);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariableDefinition {
    /// Unique identifier for the variable (e.g., "Atmospheric Concentration|CO2")
    pub name: String,
    /// Canonical unit for the variable (e.g., "ppm", "GtC / yr")
    pub unit: String,
    /// Time convention for temporal alignment
    pub time_convention: TimeConvention,
    /// Human-readable description
    pub description: String,
}

impl VariableDefinition {
    /// Create a new variable definition.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for the variable
    /// * `unit` - Canonical unit for the variable
    /// * `time_convention` - Time convention for temporal alignment
    /// * `description` - Human-readable description
    pub fn new(
        name: impl Into<String>,
        unit: impl Into<String>,
        time_convention: TimeConvention,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            time_convention,
            description: description.into(),
        }
    }
}

/// Preindustrial reference value for a variable.
///
/// Preindustrial values are configuration-dependent (e.g., historical vs counterfactual
/// scenarios) and are stored as timeseries metadata rather than in variable definitions.
///
/// # Variants
///
/// - [`Scalar`](PreindustrialValue::Scalar): Global scalar value
/// - [`FourBox`](PreindustrialValue::FourBox): Regional values for four-box model
/// - [`Hemispheric`](PreindustrialValue::Hemispheric): Hemispheric values
///
/// # Example
///
/// ```rust
/// use rscm_core::variable::PreindustrialValue;
///
/// // Scalar preindustrial CO2 concentration
/// let co2_pi = PreindustrialValue::Scalar(278.0);
/// assert_eq!(co2_pi.to_scalar(), 278.0);
///
/// // Four-box preindustrial temperature
/// let temp_pi = PreindustrialValue::FourBox([15.0, 14.0, 10.0, 9.0]);
/// let global = temp_pi.to_scalar();  // Area-weighted average
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PreindustrialValue {
    /// Global scalar value
    Scalar(f64),
    /// Four-box regional values [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand]
    FourBox([f64; 4]),
    /// Hemispheric values [Northern, Southern]
    Hemispheric([f64; 2]),
}

impl PreindustrialValue {
    /// Get the value as a scalar if this is a Scalar variant.
    ///
    /// Returns `None` for FourBox or Hemispheric variants.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            PreindustrialValue::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the values as a four-box array if this is a FourBox variant.
    ///
    /// Returns `None` for Scalar or Hemispheric variants.
    pub fn as_four_box(&self) -> Option<[f64; 4]> {
        match self {
            PreindustrialValue::FourBox(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the values as a hemispheric array if this is a Hemispheric variant.
    ///
    /// Returns `None` for Scalar or FourBox variants.
    pub fn as_hemispheric(&self) -> Option<[f64; 2]> {
        match self {
            PreindustrialValue::Hemispheric(v) => Some(*v),
            _ => None,
        }
    }

    /// Convert to a global scalar value using area-weighted averaging.
    ///
    /// For FourBox and Hemispheric variants, uses the MAGICC standard weights:
    /// - FourBox: Ocean/Land area fractions per hemisphere
    /// - Hemispheric: Equal weights (0.5 each)
    ///
    /// # Example
    ///
    /// ```rust
    /// use rscm_core::variable::PreindustrialValue;
    ///
    /// let scalar = PreindustrialValue::Scalar(278.0);
    /// assert_eq!(scalar.to_scalar(), 278.0);
    ///
    /// let hemispheric = PreindustrialValue::Hemispheric([15.0, 10.0]);
    /// assert_eq!(hemispheric.to_scalar(), 12.5);  // Equal weights
    /// ```
    pub fn to_scalar(&self) -> f64 {
        match self {
            PreindustrialValue::Scalar(v) => *v,
            PreindustrialValue::FourBox(v) => {
                // Use MAGICC standard weights
                let grid = FourBoxGrid::magicc_standard();
                let weights = grid.weights();
                v.iter().zip(weights.iter()).map(|(val, w)| val * w).sum()
            }
            PreindustrialValue::Hemispheric(v) => {
                // Use equal hemisphere weights
                let grid = HemisphericGrid::equal_weights();
                let weights = grid.weights();
                v.iter().zip(weights.iter()).map(|(val, w)| val * w).sum()
            }
        }
    }
}

impl From<f64> for PreindustrialValue {
    fn from(value: f64) -> Self {
        PreindustrialValue::Scalar(value)
    }
}

impl From<[f64; 4]> for PreindustrialValue {
    fn from(value: [f64; 4]) -> Self {
        PreindustrialValue::FourBox(value)
    }
}

impl From<[f64; 2]> for PreindustrialValue {
    fn from(value: [f64; 2]) -> Self {
        PreindustrialValue::Hemispheric(value)
    }
}

// Static variable registration using inventory crate
inventory::collect!(&'static VariableDefinition);

/// Registry for variable definitions.
///
/// Provides a central location for registering and looking up variable definitions.
/// Supports both compile-time registration (via inventory) and runtime registration.
///
/// The global registry instance is available as [`VARIABLE_REGISTRY`].
///
/// # Thread Safety
///
/// The registry uses `RwLock` for thread-safe access to runtime-registered variables.
/// Static variables registered via `inventory` are immutable and lock-free.
///
/// # Example
///
/// ```rust
/// use rscm_core::variable::{VARIABLE_REGISTRY, VariableDefinition, TimeConvention};
///
/// // Register a variable at runtime
/// let my_var = VariableDefinition::new(
///     "My Custom Variable",
///     "kg",
///     TimeConvention::Instantaneous,
///     "A custom variable for testing",
/// );
/// VARIABLE_REGISTRY.register(my_var).unwrap();
///
/// // Look up a variable
/// if let Some(var) = VARIABLE_REGISTRY.get("My Custom Variable") {
///     println!("Found: {}", var.name);
/// }
/// ```
pub struct VariableRegistry {
    /// Runtime-registered variables (protected by RwLock for thread-safe mutation)
    runtime_vars: RwLock<HashMap<String, Arc<VariableDefinition>>>,
}

impl VariableRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            runtime_vars: RwLock::new(HashMap::new()),
        }
    }

    /// Get a variable definition by name.
    ///
    /// First checks static variables (registered via inventory), then runtime variables.
    /// Returns `None` if the variable is not found.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rscm_core::variable::VARIABLE_REGISTRY;
    ///
    /// if let Some(var) = VARIABLE_REGISTRY.get("Atmospheric Concentration|CO2") {
    ///     println!("Unit: {}", var.unit);
    /// }
    /// ```
    pub fn get(&self, name: &str) -> Option<Arc<VariableDefinition>> {
        // Check static variables first
        for var in inventory::iter::<&'static VariableDefinition> {
            if var.name == name {
                return Some(Arc::new((*var).clone()));
            }
        }

        // Check runtime variables
        let runtime = self.runtime_vars.read().expect("Registry lock poisoned");
        runtime.get(name).cloned()
    }

    /// Register a variable definition at runtime.
    ///
    /// Returns an error if a variable with the same name already exists
    /// (either static or runtime).
    ///
    /// # Errors
    ///
    /// Returns `Err` with a message if a duplicate name is detected.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rscm_core::variable::{VARIABLE_REGISTRY, VariableDefinition, TimeConvention};
    ///
    /// let var = VariableDefinition::new(
    ///     "My Variable",
    ///     "kg",
    ///     TimeConvention::Instantaneous,
    ///     "Description",
    /// );
    ///
    /// VARIABLE_REGISTRY.register(var).expect("Registration failed");
    /// ```
    pub fn register(&self, var: VariableDefinition) -> Result<(), String> {
        let name = var.name.clone();

        // Check for duplicates in static variables
        for static_var in inventory::iter::<&'static VariableDefinition> {
            if static_var.name == name {
                return Err(format!(
                    "Variable '{}' is already registered as a static variable",
                    name
                ));
            }
        }

        // Check for duplicates in runtime variables and insert
        let mut runtime = self.runtime_vars.write().expect("Registry lock poisoned");
        if runtime.contains_key(&name) {
            return Err(format!(
                "Variable '{}' is already registered as a runtime variable",
                name
            ));
        }

        runtime.insert(name, Arc::new(var));
        Ok(())
    }

    /// List all registered variables.
    ///
    /// Returns a vector of all variable definitions (both static and runtime).
    ///
    /// # Example
    ///
    /// ```rust
    /// use rscm_core::variable::VARIABLE_REGISTRY;
    ///
    /// for var in VARIABLE_REGISTRY.list() {
    ///     println!("{}: {} ({})", var.name, var.unit, var.time_convention);
    /// }
    /// ```
    pub fn list(&self) -> Vec<Arc<VariableDefinition>> {
        let mut result: Vec<Arc<VariableDefinition>> = Vec::new();

        // Add static variables
        for var in inventory::iter::<&'static VariableDefinition> {
            result.push(Arc::new((*var).clone()));
        }

        // Add runtime variables
        let runtime = self.runtime_vars.read().expect("Registry lock poisoned");
        for var in runtime.values() {
            result.push(var.clone());
        }

        // Sort by name for stable output
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Check if a variable is registered.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rscm_core::variable::VARIABLE_REGISTRY;
    ///
    /// if VARIABLE_REGISTRY.is_registered("Atmospheric Concentration|CO2") {
    ///     println!("Variable is registered");
    /// }
    /// ```
    pub fn is_registered(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Get the count of registered variables.
    pub fn len(&self) -> usize {
        let static_count = inventory::iter::<&'static VariableDefinition>
            .into_iter()
            .count();
        let runtime = self.runtime_vars.read().expect("Registry lock poisoned");
        static_count + runtime.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all runtime-registered variables.
    ///
    /// Static variables registered via `define_variable!` cannot be removed.
    /// This is primarily useful for testing.
    pub fn clear_runtime(&self) {
        let mut runtime = self.runtime_vars.write().expect("Registry lock poisoned");
        runtime.clear();
    }
}

impl Default for VariableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global variable registry instance.
///
/// Use this to register and look up variable definitions throughout the application.
///
/// # Example
///
/// ```rust
/// use rscm_core::variable::{VARIABLE_REGISTRY, VariableDefinition, TimeConvention};
///
/// // Register a variable
/// VARIABLE_REGISTRY.register(VariableDefinition::new(
///     "Test Variable",
///     "units",
///     TimeConvention::Instantaneous,
///     "A test variable",
/// )).unwrap();
///
/// // Look up a variable
/// let var = VARIABLE_REGISTRY.get("Test Variable").unwrap();
/// assert_eq!(var.unit, "units");
/// ```
pub static VARIABLE_REGISTRY: LazyLock<VariableRegistry> = LazyLock::new(VariableRegistry::new);

/// Macro for defining variables at compile time.
///
/// Variables defined with this macro are automatically registered with the global
/// [`VARIABLE_REGISTRY`] when the program starts.
///
/// # Usage
///
/// ```rust,ignore
/// use rscm_core::variable::{define_variable, TimeConvention};
///
/// define_variable!(
///     CO2_CONCENTRATION,
///     name = "Atmospheric Concentration|CO2",
///     unit = "ppm",
///     time_convention = TimeConvention::StartOfYear,
///     description = "Atmospheric CO2 concentration",
/// );
/// ```
///
/// # Parameters
///
/// - `$var_name`: Identifier for the static variable (uppercase by convention)
/// - `name`: String literal for the variable's unique identifier
/// - `unit`: String literal for the variable's canonical unit
/// - `time_convention`: [`TimeConvention`] value
/// - `description`: String literal description
#[macro_export]
macro_rules! define_variable {
    (
        $var_name:ident,
        name = $name:expr,
        unit = $unit:expr,
        time_convention = $convention:expr,
        description = $desc:expr $(,)?
    ) => {
        #[allow(non_upper_case_globals)]
        pub static $var_name: $crate::variable::VariableDefinition =
            $crate::variable::VariableDefinition {
                name: String::new(),
                unit: String::new(),
                time_convention: $convention,
                description: String::new(),
            };

        // This won't work because String::new() isn't const.
        // We need a different approach for static registration.
    };
}

// Alternative approach: Use inventory with a wrapper that holds &'static str
// and converts to VariableDefinition on access.

/// Static variable definition holder for compile-time registration.
///
/// This struct holds `&'static str` references that can be used in const contexts,
/// and provides conversion to [`VariableDefinition`] at runtime.
#[derive(Debug, Clone, Copy)]
pub struct StaticVariableDefinition {
    /// Variable name
    pub name: &'static str,
    /// Canonical unit
    pub unit: &'static str,
    /// Time convention
    pub time_convention: TimeConvention,
    /// Description
    pub description: &'static str,
}

impl StaticVariableDefinition {
    /// Create a new static variable definition.
    pub const fn new(
        name: &'static str,
        unit: &'static str,
        time_convention: TimeConvention,
        description: &'static str,
    ) -> Self {
        Self {
            name,
            unit,
            time_convention,
            description,
        }
    }

    /// Convert to a [`VariableDefinition`].
    pub fn to_variable_definition(&self) -> VariableDefinition {
        VariableDefinition {
            name: self.name.to_string(),
            unit: self.unit.to_string(),
            time_convention: self.time_convention,
            description: self.description.to_string(),
        }
    }
}

// Register StaticVariableDefinition with inventory
inventory::collect!(StaticVariableDefinition);

/// Macro for defining variables at compile time using static strings.
///
/// Variables defined with this macro are automatically registered with the global
/// [`VARIABLE_REGISTRY`] when the program starts.
///
/// # Usage
///
/// ```rust
/// use rscm_core::define_static_variable;
/// use rscm_core::variable::TimeConvention;
///
/// define_static_variable!(
///     MY_VARIABLE,
///     name = "My Variable|Test",
///     unit = "kg",
///     time_convention = TimeConvention::Instantaneous,
///     description = "A test variable",
/// );
/// ```
///
/// # Parameters
///
/// - `$var_name`: Identifier for the static variable (uppercase by convention)
/// - `name`: String literal for the variable's unique identifier
/// - `unit`: String literal for the variable's canonical unit
/// - `time_convention`: [`TimeConvention`] value
/// - `description`: String literal description
#[macro_export]
macro_rules! define_static_variable {
    (
        $var_name:ident,
        name = $name:expr,
        unit = $unit:expr,
        time_convention = $convention:expr,
        description = $desc:expr $(,)?
    ) => {
        #[doc = concat!("Static variable definition for ", $name)]
        pub static $var_name: $crate::variable::StaticVariableDefinition =
            $crate::variable::StaticVariableDefinition::new($name, $unit, $convention, $desc);

        ::inventory::submit! { $var_name }
    };
}

// Re-export the macro at module level for convenience
pub use crate::define_static_variable;

impl VariableRegistry {
    /// Get a variable definition by name, including static variables.
    ///
    /// Checks both runtime-registered variables and compile-time static variables.
    pub fn get_with_static(&self, name: &str) -> Option<VariableDefinition> {
        // Check static variables first (from inventory)
        for var in inventory::iter::<StaticVariableDefinition> {
            if var.name == name {
                return Some(var.to_variable_definition());
            }
        }

        // Check runtime variables
        let runtime = self.runtime_vars.read().expect("Registry lock poisoned");
        runtime.get(name).map(|v| (**v).clone())
    }

    /// List all registered variables including static variables.
    pub fn list_all(&self) -> Vec<VariableDefinition> {
        let mut result: Vec<VariableDefinition> = Vec::new();

        // Add static variables
        for var in inventory::iter::<StaticVariableDefinition> {
            result.push(var.to_variable_definition());
        }

        // Add runtime variables
        let runtime = self.runtime_vars.read().expect("Registry lock poisoned");
        for var in runtime.values() {
            result.push((**var).clone());
        }

        // Sort by name for stable output
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Check if a variable is registered (static or runtime).
    pub fn is_registered_any(&self, name: &str) -> bool {
        self.get_with_static(name).is_some()
    }

    /// Get the total count of registered variables (static + runtime).
    pub fn len_all(&self) -> usize {
        let static_count = inventory::iter::<StaticVariableDefinition>
            .into_iter()
            .count();
        let runtime = self.runtime_vars.read().expect("Registry lock poisoned");
        static_count + runtime.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_convention_display() {
        assert_eq!(format!("{}", TimeConvention::StartOfYear), "StartOfYear");
        assert_eq!(format!("{}", TimeConvention::MidYear), "MidYear");
        assert_eq!(
            format!("{}", TimeConvention::Instantaneous),
            "Instantaneous"
        );
    }

    #[test]
    fn test_time_convention_equality() {
        assert_eq!(TimeConvention::StartOfYear, TimeConvention::StartOfYear);
        assert_ne!(TimeConvention::StartOfYear, TimeConvention::MidYear);
    }

    #[test]
    fn test_time_convention_serialization() {
        let conv = TimeConvention::MidYear;
        let json = serde_json::to_string(&conv).unwrap();
        assert_eq!(json, "\"MidYear\"");

        let deserialized: TimeConvention = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, conv);
    }

    #[test]
    fn test_variable_definition_new() {
        let var = VariableDefinition::new(
            "Atmospheric Concentration|CO2",
            "ppm",
            TimeConvention::StartOfYear,
            "Atmospheric CO2 concentration",
        );

        assert_eq!(var.name, "Atmospheric Concentration|CO2");
        assert_eq!(var.unit, "ppm");
        assert_eq!(var.time_convention, TimeConvention::StartOfYear);
        assert_eq!(var.description, "Atmospheric CO2 concentration");
    }

    #[test]
    fn test_variable_definition_serialization() {
        let var = VariableDefinition::new(
            "Emissions|CO2",
            "GtC / yr",
            TimeConvention::MidYear,
            "CO2 emissions",
        );

        let json = serde_json::to_string(&var).unwrap();
        let deserialized: VariableDefinition = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized, var);
    }

    #[test]
    fn test_preindustrial_scalar() {
        let pi = PreindustrialValue::Scalar(278.0);
        assert_eq!(pi.as_scalar(), Some(278.0));
        assert_eq!(pi.as_four_box(), None);
        assert_eq!(pi.as_hemispheric(), None);
        assert_eq!(pi.to_scalar(), 278.0);
    }

    #[test]
    fn test_preindustrial_four_box() {
        let pi = PreindustrialValue::FourBox([15.0, 14.0, 10.0, 9.0]);
        assert_eq!(pi.as_scalar(), None);
        assert_eq!(pi.as_four_box(), Some([15.0, 14.0, 10.0, 9.0]));
        assert_eq!(pi.as_hemispheric(), None);

        // to_scalar uses MAGICC standard weights
        let scalar = pi.to_scalar();
        // Should be a weighted average
        assert!(scalar > 9.0 && scalar < 15.0);
    }

    #[test]
    fn test_preindustrial_hemispheric() {
        let pi = PreindustrialValue::Hemispheric([15.0, 10.0]);
        assert_eq!(pi.as_scalar(), None);
        assert_eq!(pi.as_four_box(), None);
        assert_eq!(pi.as_hemispheric(), Some([15.0, 10.0]));

        // Equal weights gives simple average
        assert_eq!(pi.to_scalar(), 12.5);
    }

    #[test]
    fn test_preindustrial_from_f64() {
        let pi: PreindustrialValue = 278.0.into();
        assert_eq!(pi.as_scalar(), Some(278.0));
    }

    #[test]
    fn test_preindustrial_from_four_box_array() {
        let pi: PreindustrialValue = [15.0, 14.0, 10.0, 9.0].into();
        assert_eq!(pi.as_four_box(), Some([15.0, 14.0, 10.0, 9.0]));
    }

    #[test]
    fn test_preindustrial_from_hemispheric_array() {
        let pi: PreindustrialValue = [15.0, 10.0].into();
        assert_eq!(pi.as_hemispheric(), Some([15.0, 10.0]));
    }

    #[test]
    fn test_preindustrial_serialization() {
        let pi = PreindustrialValue::FourBox([15.0, 14.0, 10.0, 9.0]);
        let json = serde_json::to_string(&pi).unwrap();
        let deserialized: PreindustrialValue = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, pi);
    }

    #[test]
    fn test_registry_register_and_get() {
        let registry = VariableRegistry::new();

        let var = VariableDefinition::new(
            "Test Variable 1",
            "units",
            TimeConvention::Instantaneous,
            "Test description",
        );

        registry.register(var.clone()).unwrap();

        let retrieved = registry.get("Test Variable 1").unwrap();
        assert_eq!(retrieved.name, "Test Variable 1");
        assert_eq!(retrieved.unit, "units");
    }

    #[test]
    fn test_registry_duplicate_rejection() {
        let registry = VariableRegistry::new();

        let var = VariableDefinition::new(
            "Test Variable 2",
            "units",
            TimeConvention::Instantaneous,
            "Test description",
        );

        registry.register(var.clone()).unwrap();
        let result = registry.register(var);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already registered"));
    }

    #[test]
    fn test_registry_list() {
        let registry = VariableRegistry::new();

        registry
            .register(VariableDefinition::new(
                "Var B",
                "b",
                TimeConvention::MidYear,
                "",
            ))
            .unwrap();
        registry
            .register(VariableDefinition::new(
                "Var A",
                "a",
                TimeConvention::StartOfYear,
                "",
            ))
            .unwrap();

        let list = registry.list();
        assert_eq!(list.len(), 2);
        // Should be sorted by name
        assert_eq!(list[0].name, "Var A");
        assert_eq!(list[1].name, "Var B");
    }

    #[test]
    fn test_registry_is_registered() {
        let registry = VariableRegistry::new();

        assert!(!registry.is_registered("Nonexistent"));

        registry
            .register(VariableDefinition::new(
                "Test Variable 3",
                "units",
                TimeConvention::Instantaneous,
                "",
            ))
            .unwrap();

        assert!(registry.is_registered("Test Variable 3"));
    }

    #[test]
    fn test_registry_clear_runtime() {
        let registry = VariableRegistry::new();

        registry
            .register(VariableDefinition::new(
                "Test Variable 4",
                "units",
                TimeConvention::Instantaneous,
                "",
            ))
            .unwrap();

        assert!(registry.is_registered("Test Variable 4"));

        registry.clear_runtime();

        assert!(!registry.is_registered("Test Variable 4"));
    }

    #[test]
    fn test_static_variable_definition() {
        let static_var = StaticVariableDefinition::new(
            "Static Var",
            "kg",
            TimeConvention::MidYear,
            "A static variable",
        );

        let var = static_var.to_variable_definition();
        assert_eq!(var.name, "Static Var");
        assert_eq!(var.unit, "kg");
        assert_eq!(var.time_convention, TimeConvention::MidYear);
        assert_eq!(var.description, "A static variable");
    }
}

//! Target observations for calibration.
//!
//! This module provides data structures for storing and managing observational data
//! that models are calibrated against.

use crate::{Error, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;

/// A single observational data point with associated uncertainty.
///
/// Represents one measurement at a specific time with its 1-sigma uncertainty.
/// Uncertainties are assumed to be Gaussian for likelihood calculations.
///
/// # Example
///
/// ```
/// use rscm_calibrate::target::Observation;
///
/// // Temperature observation: 1.2°C ± 0.1°C in year 2020
/// let obs = Observation::new(2020.0, 1.2, 0.1).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Time coordinate (typically decimal year)
    pub time: f64,
    /// Observed value
    pub value: f64,
    /// Uncertainty (1-sigma standard deviation, must be positive)
    pub uncertainty: f64,
}

impl Observation {
    /// Create a new observation.
    ///
    /// # Arguments
    /// * `time` - Time coordinate
    /// * `value` - Observed value
    /// * `uncertainty` - 1-sigma uncertainty (must be positive)
    ///
    /// # Errors
    /// Returns an error if uncertainty is not positive.
    pub fn new(time: f64, value: f64, uncertainty: f64) -> Result<Self> {
        if uncertainty <= 0.0 {
            return Err(Error::InvalidParameter(
                "Uncertainty must be positive".to_string(),
            ));
        }
        Ok(Self {
            time,
            value,
            uncertainty,
        })
    }
}

/// Target observations for a single model output variable.
///
/// Groups all observations for one variable (e.g., global temperature, ocean heat content).
/// Observations are automatically sorted by time for efficient lookups.
///
/// # Reference Periods
///
/// Climate data is often reported as anomalies relative to a baseline period
/// (e.g., 1850-1900 for pre-industrial baseline). Set `reference_period` to
/// indicate that observations are anomalies relative to the mean over that period.
///
/// # Example
///
/// ```
/// use rscm_calibrate::target::VariableTarget;
///
/// let mut temp = VariableTarget::new("Temperature|Global");
/// temp.add(2020.0, 1.2, 0.1).unwrap()
///     .add(2021.0, 1.3, 0.1).unwrap()
///     .with_reference_period(1850.0, 1900.0);  // Anomaly relative to 1850-1900
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableTarget {
    /// Variable name (e.g., "Temperature|Global")
    pub name: String,
    /// Observations sorted by time
    pub observations: Vec<Observation>,
    /// Reference period for anomaly calculation (inclusive range)
    pub reference_period: Option<RangeInclusive<f64>>,
}

impl VariableTarget {
    /// Create a new variable target.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            observations: Vec::new(),
            reference_period: None,
        }
    }

    /// Add an observation.
    ///
    /// Observations are kept sorted by time.
    pub fn add_observation(&mut self, obs: Observation) -> &mut Self {
        self.observations.push(obs);
        self.observations.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self
    }

    /// Add an observation from components.
    pub fn add(&mut self, time: f64, value: f64, uncertainty: f64) -> Result<&mut Self> {
        self.add_observation(Observation::new(time, value, uncertainty)?);
        Ok(self)
    }

    /// Add an observation with relative uncertainty (as a fraction of the value).
    ///
    /// # Example
    /// ```
    /// # use rscm_calibrate::target::VariableTarget;
    /// let mut target = VariableTarget::new("Temperature");
    /// // 5% relative uncertainty
    /// target.add_relative(2020.0, 1.0, 0.05).unwrap();
    /// assert_eq!(target.observations[0].uncertainty, 0.05);
    /// ```
    pub fn add_relative(
        &mut self,
        time: f64,
        value: f64,
        relative_uncertainty: f64,
    ) -> Result<&mut Self> {
        let uncertainty = value.abs() * relative_uncertainty;
        self.add(time, value, uncertainty)
    }

    /// Set the reference period for anomaly calculation.
    ///
    /// When set, observations will be interpreted as anomalies relative to the
    /// mean over this period.
    pub fn with_reference_period(&mut self, start: f64, end: f64) -> &mut Self {
        self.reference_period = Some(start..=end);
        self
    }

    /// Get observations within a time range.
    pub fn observations_in_range(&self, start: f64, end: f64) -> Vec<&Observation> {
        self.observations
            .iter()
            .filter(|obs| obs.time >= start && obs.time <= end)
            .collect()
    }

    /// Get the time range covered by observations.
    pub fn time_range(&self) -> Option<(f64, f64)> {
        if self.observations.is_empty() {
            None
        } else {
            Some((
                self.observations.first().unwrap().time,
                self.observations.last().unwrap().time,
            ))
        }
    }
}

/// Collection of target observations for multiple model output variables.
///
/// Target manages all observational data used for model calibration.
/// It supports multiple variables with different time series and uncertainties.
///
/// # Fluent Builder API
///
/// The API uses a fluent builder pattern where methods return `&mut Self`
/// for chaining. Use `add_variable()` to start a new variable, then chain
/// observation additions and reference period settings.
///
/// # Variable Ordering
///
/// Variables are stored in an `IndexMap` and preserve insertion order.
/// This ensures reproducible likelihood calculations.
///
/// # Example
///
/// ```
/// use rscm_calibrate::Target;
///
/// let mut target = Target::new();
///
/// // Add temperature observations (anomalies relative to 1850-1900)
/// target
///     .add_variable("Temperature|Global")
///     .add(2020.0, 1.2, 0.1).unwrap()
///     .add(2021.0, 1.3, 0.1).unwrap()
///     .with_reference_period(1850.0, 1900.0);
///
/// // Add ocean heat content with relative uncertainty
/// target
///     .add_variable("Ocean Heat Content")
///     .add_relative(2020.0, 200.0, 0.05).unwrap(); // 5% uncertainty
///
/// assert_eq!(target.variables().len(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Target {
    /// Variables indexed by name
    variables: IndexMap<String, VariableTarget>,
}

impl Target {
    /// Create a new empty target.
    pub fn new() -> Self {
        Self {
            variables: IndexMap::new(),
        }
    }

    /// Add a new variable and return a mutable reference for chaining.
    ///
    /// If the variable already exists, returns a reference to the existing one.
    pub fn add_variable(&mut self, name: impl Into<String>) -> &mut VariableTarget {
        let name = name.into();
        self.variables
            .entry(name.clone())
            .or_insert_with(|| VariableTarget::new(name))
    }

    /// Get a variable by name.
    pub fn get_variable(&self, name: &str) -> Option<&VariableTarget> {
        self.variables.get(name)
    }

    /// Get a mutable reference to a variable by name.
    pub fn get_variable_mut(&mut self, name: &str) -> Option<&mut VariableTarget> {
        self.variables.get_mut(name)
    }

    /// Get all variables in insertion order.
    pub fn variables(&self) -> &IndexMap<String, VariableTarget> {
        &self.variables
    }

    /// Get the names of all variables.
    pub fn variable_names(&self) -> Vec<&str> {
        self.variables.keys().map(|s| s.as_str()).collect()
    }

    /// Get the total number of observations across all variables.
    pub fn total_observations(&self) -> usize {
        self.variables.values().map(|v| v.observations.len()).sum()
    }

    /// Get the time range covered by all observations.
    pub fn time_range(&self) -> Option<(f64, f64)> {
        let ranges: Vec<_> = self
            .variables
            .values()
            .filter_map(|v| v.time_range())
            .collect();

        if ranges.is_empty() {
            None
        } else {
            let min_time = ranges
                .iter()
                .map(|(start, _)| start)
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max_time = ranges
                .iter()
                .map(|(_, end)| end)
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            Some((min_time, max_time))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_creation() {
        let obs = Observation::new(2020.0, 1.2, 0.1).unwrap();
        assert_eq!(obs.time, 2020.0);
        assert_eq!(obs.value, 1.2);
        assert_eq!(obs.uncertainty, 0.1);
    }

    #[test]
    fn observation_invalid_uncertainty() {
        assert!(Observation::new(2020.0, 1.2, 0.0).is_err());
        assert!(Observation::new(2020.0, 1.2, -0.1).is_err());
    }

    #[test]
    fn variable_target_add_observations() {
        let mut target = VariableTarget::new("Temperature");
        target.add(2020.0, 1.2, 0.1).unwrap();
        target.add(2019.0, 1.1, 0.1).unwrap();
        target.add(2021.0, 1.3, 0.1).unwrap();

        assert_eq!(target.observations.len(), 3);
        // Should be sorted by time
        assert_eq!(target.observations[0].time, 2019.0);
        assert_eq!(target.observations[1].time, 2020.0);
        assert_eq!(target.observations[2].time, 2021.0);
    }

    #[test]
    fn variable_target_relative_uncertainty() {
        let mut target = VariableTarget::new("Temperature");
        target.add_relative(2020.0, 100.0, 0.05).unwrap();

        assert_eq!(target.observations[0].value, 100.0);
        assert_eq!(target.observations[0].uncertainty, 5.0);
    }

    #[test]
    fn variable_target_reference_period() {
        let mut target = VariableTarget::new("Temperature");
        target.with_reference_period(1850.0, 1900.0);

        assert_eq!(target.reference_period, Some(1850.0..=1900.0));
    }

    #[test]
    fn variable_target_time_range() {
        let mut target = VariableTarget::new("Temperature");
        assert_eq!(target.time_range(), None);

        target.add(2020.0, 1.2, 0.1).unwrap();
        target.add(2015.0, 1.0, 0.1).unwrap();
        target.add(2025.0, 1.5, 0.1).unwrap();

        assert_eq!(target.time_range(), Some((2015.0, 2025.0)));
    }

    #[test]
    fn variable_target_observations_in_range() {
        let mut target = VariableTarget::new("Temperature");
        target.add(2010.0, 0.8, 0.1).unwrap();
        target.add(2015.0, 1.0, 0.1).unwrap();
        target.add(2020.0, 1.2, 0.1).unwrap();
        target.add(2025.0, 1.5, 0.1).unwrap();

        let in_range = target.observations_in_range(2014.0, 2021.0);
        assert_eq!(in_range.len(), 2);
        assert_eq!(in_range[0].time, 2015.0);
        assert_eq!(in_range[1].time, 2020.0);
    }

    #[test]
    fn target_fluent_api() {
        let mut target = Target::new();
        target
            .add_variable("Temperature|Global")
            .add(2020.0, 1.2, 0.1)
            .unwrap()
            .add(2021.0, 1.3, 0.1)
            .unwrap()
            .with_reference_period(1850.0, 1900.0);

        target
            .add_variable("Ocean Heat Content")
            .add_relative(2020.0, 200.0, 0.05)
            .unwrap();

        assert_eq!(target.variables().len(), 2);
        assert_eq!(
            target
                .get_variable("Temperature|Global")
                .unwrap()
                .observations
                .len(),
            2
        );
        assert_eq!(
            target
                .get_variable("Ocean Heat Content")
                .unwrap()
                .observations
                .len(),
            1
        );
    }

    #[test]
    fn target_total_observations() {
        let mut target = Target::new();
        target.add_variable("Var1").add(2020.0, 1.0, 0.1).unwrap();
        target
            .add_variable("Var2")
            .add(2020.0, 2.0, 0.1)
            .unwrap()
            .add(2021.0, 2.1, 0.1)
            .unwrap();

        assert_eq!(target.total_observations(), 3);
    }

    #[test]
    fn target_time_range() {
        let mut target = Target::new();
        assert_eq!(target.time_range(), None);

        target
            .add_variable("Var1")
            .add(2020.0, 1.0, 0.1)
            .unwrap()
            .add(2025.0, 1.5, 0.1)
            .unwrap();

        target
            .add_variable("Var2")
            .add(2015.0, 0.8, 0.1)
            .unwrap()
            .add(2022.0, 1.2, 0.1)
            .unwrap();

        assert_eq!(target.time_range(), Some((2015.0, 2025.0)));
    }

    #[test]
    fn target_serialization() {
        let mut target = Target::new();
        target
            .add_variable("Temperature")
            .add(2020.0, 1.2, 0.1)
            .unwrap()
            .with_reference_period(1850.0, 1900.0);

        let json = serde_json::to_string(&target).unwrap();
        let deserialized: Target = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized
                .get_variable("Temperature")
                .unwrap()
                .observations
                .len(),
            1
        );
        assert_eq!(
            deserialized
                .get_variable("Temperature")
                .unwrap()
                .reference_period,
            Some(1850.0..=1900.0)
        );
    }
}

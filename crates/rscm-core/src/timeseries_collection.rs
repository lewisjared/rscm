//! Named collections of timeseries for managing model state.
//!
//! This module provides [`TimeseriesCollection`], the central data structure for
//! storing all timeseries data during a model simulation. It holds both prescribed
//! (supplied to the model) and modelled (produced by a component) variables.
//!
//! # Overview
//!
//! A `TimeseriesCollection` maps variable names to their timeseries data.
//! A variable name reads from general to specific: the quantity, then the species,
//! then the sector, then any further breakdown, such as `"Emissions|CO2|Fossil"`.
//! For example:
//! - `"Emissions|CO2"` - CO2 emissions in Mt CO2/yr
//! - `"Atmospheric Concentrations|CO2"` - CO2 concentration in ppm
//! - `"Effective Radiative Forcing|CO2"` - CO2 forcing in W/m^2
//!
//! # Grids
//!
//! The collection supports three grids via [`TimeseriesData`]:
//! - **Scalar**: Global mean values
//! - **FourBox**: Regional values (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
//! - **Hemispheric**: Hemispheric values (Northern, Southern)

use crate::component::GridType;
use crate::errors::RSCMResult;
use crate::spatial::{FourBoxGrid, HemisphericGrid, SpatialGrid};
use crate::state::{FourBoxSlice, HemisphericSlice};
use crate::timeseries::{FloatValue, GridTimeseries, Timeseries};
use serde::{Deserialize, Serialize};

/// Container for timeseries data on different grids.
///
/// This enum wraps the different grids supported by the model,
/// providing a unified interface for storing and accessing timeseries data
/// regardless of grid.
///
/// # Variants
///
/// - **Scalar**: Single global value per step (most common)
/// - **FourBox**: Four regional values following MAGICC conventions
/// - **Hemispheric**: Two hemispheric values
///
/// # Example
///
/// ```ignore
/// match data {
///     TimeseriesData::Scalar(ts) => {
///         let global_value = ts.at_scalar(time_index).unwrap();
///     }
///     TimeseriesData::FourBox(ts) => {
///         let no_value = ts.at(time_index, FourBoxRegion::NorthernOcean);
///     }
///     TimeseriesData::Hemispheric(ts) => {
///         let nh_value = ts.at(time_index, HemisphericRegion::Northern);
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeseriesData {
    /// Scalar (global mean) timeseries with a single value per step.
    Scalar(Timeseries<FloatValue>),
    /// Four-box regional timeseries with values for:
    /// - Northern Ocean
    /// - Northern Land
    /// - Southern Ocean
    /// - Southern Land
    FourBox(GridTimeseries<FloatValue, FourBoxGrid>),
    /// Hemispheric timeseries with values for:
    /// - Northern Hemisphere
    /// - Southern Hemisphere
    Hemispheric(GridTimeseries<FloatValue, HemisphericGrid>),
}

impl From<Timeseries<FloatValue>> for TimeseriesData {
    fn from(timeseries: Timeseries<FloatValue>) -> Self {
        TimeseriesData::Scalar(timeseries)
    }
}

impl From<GridTimeseries<FloatValue, FourBoxGrid>> for TimeseriesData {
    fn from(timeseries: GridTimeseries<FloatValue, FourBoxGrid>) -> Self {
        TimeseriesData::FourBox(timeseries)
    }
}

impl From<GridTimeseries<FloatValue, HemisphericGrid>> for TimeseriesData {
    fn from(timeseries: GridTimeseries<FloatValue, HemisphericGrid>) -> Self {
        TimeseriesData::Hemispheric(timeseries)
    }
}

impl TimeseriesData {
    /// Get the grid size (number of regions)
    pub fn grid_size(&self) -> usize {
        match self {
            TimeseriesData::Scalar(_) => 1,
            TimeseriesData::FourBox(ts) => ts.grid().size(),
            TimeseriesData::Hemispheric(ts) => ts.grid().size(),
        }
    }

    /// The grid the series is stored on.
    pub fn grid(&self) -> GridType {
        match self {
            TimeseriesData::Scalar(_) => GridType::Scalar,
            TimeseriesData::FourBox(_) => GridType::FourBox,
            TimeseriesData::Hemispheric(_) => GridType::Hemispheric,
        }
    }

    /// Get the grid name
    pub fn grid_name(&self) -> &'static str {
        match self {
            TimeseriesData::Scalar(ts) => ts.grid().grid_name(),
            TimeseriesData::FourBox(ts) => ts.grid().grid_name(),
            TimeseriesData::Hemispheric(ts) => ts.grid().grid_name(),
        }
    }

    /// Get the time series length
    pub fn len(&self) -> usize {
        match self {
            TimeseriesData::Scalar(ts) => ts.len(),
            TimeseriesData::FourBox(ts) => ts.len(),
            TimeseriesData::Hemispheric(ts) => ts.len(),
        }
    }

    /// Check if the timeseries is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the index of the latest valid step
    pub fn latest(&self) -> usize {
        match self {
            TimeseriesData::Scalar(ts) => ts.latest(),
            TimeseriesData::FourBox(ts) => ts.latest(),
            TimeseriesData::Hemispheric(ts) => ts.latest(),
        }
    }

    /// Get the index corresponding to a time value
    ///
    /// Returns `None` if the time is not found in the time axis.
    pub fn index_of_time(&self, time: crate::timeseries::Time) -> Option<usize> {
        match self {
            TimeseriesData::Scalar(ts) => ts.time_axis().index_of(time),
            TimeseriesData::FourBox(ts) => ts.time_axis().index_of(time),
            TimeseriesData::Hemispheric(ts) => ts.time_axis().index_of(time),
        }
    }

    /// Get the scalar timeseries if this is a Scalar variant
    pub fn as_scalar(&self) -> Option<&Timeseries<FloatValue>> {
        match self {
            TimeseriesData::Scalar(ts) => Some(ts),
            _ => None,
        }
    }

    /// Get the four-box timeseries if this is a FourBox variant
    pub fn as_four_box(&self) -> Option<&GridTimeseries<FloatValue, FourBoxGrid>> {
        match self {
            TimeseriesData::FourBox(ts) => Some(ts),
            _ => None,
        }
    }

    /// Get the hemispheric timeseries if this is a Hemispheric variant
    pub fn as_hemispheric(&self) -> Option<&GridTimeseries<FloatValue, HemisphericGrid>> {
        match self {
            TimeseriesData::Hemispheric(ts) => Some(ts),
            _ => None,
        }
    }

    /// Get a mutable reference to the scalar timeseries if this is a Scalar variant
    pub fn as_scalar_mut(&mut self) -> Option<&mut Timeseries<FloatValue>> {
        match self {
            TimeseriesData::Scalar(ts) => Some(ts),
            _ => None,
        }
    }

    /// Get a mutable reference to the four-box timeseries if this is a FourBox variant
    pub fn as_four_box_mut(&mut self) -> Option<&mut GridTimeseries<FloatValue, FourBoxGrid>> {
        match self {
            TimeseriesData::FourBox(ts) => Some(ts),
            _ => None,
        }
    }

    /// Get a mutable reference to the hemispheric timeseries if this is a Hemispheric variant
    pub fn as_hemispheric_mut(
        &mut self,
    ) -> Option<&mut GridTimeseries<FloatValue, HemisphericGrid>> {
        match self {
            TimeseriesData::Hemispheric(ts) => Some(ts),
            _ => None,
        }
    }

    /// Set a scalar value at the given time index
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a Scalar timeseries
    pub fn set_scalar(
        &mut self,
        name: &str,
        time_index: usize,
        value: FloatValue,
    ) -> RSCMResult<()> {
        match self {
            TimeseriesData::Scalar(ts) => {
                ts.set(time_index, crate::spatial::ScalarRegion::Global, value);
                Ok(())
            }
            TimeseriesData::FourBox(_) => Err(crate::errors::RSCMError::GridOutputMismatch {
                variable: name.to_string(),
                expected_grid: "scalar".to_string(),
                component_grid: "four_box".to_string(),
            }),
            TimeseriesData::Hemispheric(_) => Err(crate::errors::RSCMError::GridOutputMismatch {
                variable: name.to_string(),
                expected_grid: "scalar".to_string(),
                component_grid: "hemispheric".to_string(),
            }),
        }
    }

    /// Set a FourBox slice of values at the given time index
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a four-box timeseries
    pub fn set_four_box(
        &mut self,
        name: &str,
        time_index: usize,
        slice: &FourBoxSlice,
    ) -> RSCMResult<()> {
        match self {
            TimeseriesData::FourBox(ts) => {
                ts.set_from_slice(time_index, slice);
                Ok(())
            }
            TimeseriesData::Scalar(_) => Err(crate::errors::RSCMError::GridOutputMismatch {
                variable: name.to_string(),
                expected_grid: "four_box".to_string(),
                component_grid: "scalar".to_string(),
            }),
            TimeseriesData::Hemispheric(_) => Err(crate::errors::RSCMError::GridOutputMismatch {
                variable: name.to_string(),
                expected_grid: "four_box".to_string(),
                component_grid: "hemispheric".to_string(),
            }),
        }
    }

    /// Set a Hemispheric slice of values at the given time index
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a Hemispheric timeseries
    pub fn set_hemispheric(
        &mut self,
        name: &str,
        time_index: usize,
        slice: &HemisphericSlice,
    ) -> RSCMResult<()> {
        match self {
            TimeseriesData::Hemispheric(ts) => {
                ts.set_from_slice(time_index, slice);
                Ok(())
            }
            TimeseriesData::Scalar(_) => Err(crate::errors::RSCMError::GridOutputMismatch {
                variable: name.to_string(),
                expected_grid: "hemispheric".to_string(),
                component_grid: "scalar".to_string(),
            }),
            TimeseriesData::FourBox(_) => Err(crate::errors::RSCMError::GridOutputMismatch {
                variable: name.to_string(),
                expected_grid: "hemispheric".to_string(),
                component_grid: "four_box".to_string(),
            }),
        }
    }
}

/// A named timeseries, the item stored in a [`TimeseriesCollection`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesItem {
    /// The timeseries data (scalar, four-box, or hemispheric)
    #[serde(alias = "timeseries")]
    pub data: TimeseriesData,
    /// Variable name (e.g., "Emissions|CO2", "Surface Air Temperature Change")
    pub name: String,
}

/// A named collection of timeseries that holds all model state.
///
/// `TimeseriesCollection` is the central data structure for managing model state
/// during simulation. It provides:
/// - Named access to variables via string keys
/// - Support for multiple grids
/// - Iteration over all contained timeseries
///
/// # Usage
///
/// ```ignore
/// use rscm_core::timeseries_collection::TimeseriesCollection;
/// use rscm_core::timeseries::Timeseries;
///
/// let mut collection = TimeseriesCollection::new();
///
/// // Add a scalar timeseries
/// collection.add_timeseries("Emissions|CO2".to_string(), emissions_ts);
///
/// // Retrieve by name
/// if let Some(data) = collection.get_data("Emissions|CO2") {
///     let scalar = data.as_scalar().unwrap();
///     // Use the timeseries...
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesCollection {
    /// Internal storage for timeseries items, kept sorted by name for stable serialisation
    #[serde(deserialize_with = "sorted_items")]
    timeseries: Vec<TimeseriesItem>,
}

fn sorted_items<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<TimeseriesItem>, D::Error> {
    let mut items = Vec::<TimeseriesItem>::deserialize(deserializer)?;
    items.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    Ok(items)
}

impl Default for TimeseriesCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeseriesCollection {
    /// Create a new empty collection.
    pub fn new() -> Self {
        Self {
            timeseries: Vec::new(),
        }
    }

    /// Add a new scalar timeseries to the collection
    ///
    /// # Panics
    /// Panics if a timeseries with the same name already exists in the collection
    pub fn add_timeseries(&mut self, name: String, timeseries: Timeseries<FloatValue>) {
        self.add_data(name, TimeseriesData::Scalar(timeseries));
    }

    /// Add a new four-box grid timeseries to the collection
    ///
    /// # Panics
    /// Panics if a timeseries with the same name already exists in the collection
    pub fn add_four_box_timeseries(
        &mut self,
        name: String,
        timeseries: GridTimeseries<FloatValue, FourBoxGrid>,
    ) {
        self.add_data(name, TimeseriesData::FourBox(timeseries));
    }

    /// Add a new hemispheric grid timeseries to the collection
    ///
    /// # Panics
    /// Panics if a timeseries with the same name already exists in the collection
    pub fn add_hemispheric_timeseries(
        &mut self,
        name: String,
        timeseries: GridTimeseries<FloatValue, HemisphericGrid>,
    ) {
        self.add_data(name, TimeseriesData::Hemispheric(timeseries));
    }

    /// Add a timeseries on any grid to the collection
    ///
    /// # Panics
    /// Panics if a timeseries with the same name already exists in the collection
    pub fn add_data(&mut self, name: String, data: impl Into<TimeseriesData>) {
        self.insert(TimeseriesItem {
            data: data.into(),
            name,
        });
    }

    /// Get a timeseries item by variable name.
    ///
    /// Returns the full [`TimeseriesItem`] including metadata. Use [`get_data`](Self::get_data)
    /// if you only need the timeseries data.
    pub fn get_by_name(&self, name: &str) -> Option<&TimeseriesItem> {
        self.position(name).map(|index| &self.timeseries[index])
    }

    /// Get a mutable reference to a timeseries item by variable name.
    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut TimeseriesItem> {
        self.position(name).map(|index| &mut self.timeseries[index])
    }

    /// The item at an index from `position`, valid until the next insertion.
    pub(crate) fn item(&self, index: usize) -> &TimeseriesItem {
        &self.timeseries[index]
    }

    pub(crate) fn data_mut(&mut self, index: usize) -> &mut TimeseriesData {
        &mut self.timeseries[index].data
    }

    /// Index of a name, scanning on a miss in case an item was renamed out of order.
    pub(crate) fn position(&self, name: &str) -> Option<usize> {
        self.timeseries
            .binary_search_by(|x| x.name.as_str().cmp(name))
            .ok()
            .or_else(|| self.timeseries.iter().position(|x| x.name == name))
    }

    /// Get the timeseries data for a variable by name.
    ///
    /// This is the most common way to access timeseries data. Returns
    /// the [`TimeseriesData`] enum which can be pattern-matched or converted
    /// to the appropriate grid.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(data) = collection.get_data("Surface Air Temperature Change") {
    ///     match data {
    ///         TimeseriesData::Scalar(ts) => println!("Global: {:?}", ts.latest_value()),
    ///         TimeseriesData::FourBox(ts) => println!("Regional: {:?}", ts.latest_values()),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn get_data(&self, name: &str) -> Option<&TimeseriesData> {
        self.get_by_name(name).map(|item| &item.data)
    }

    /// Get mutable timeseries data for a variable by name.
    ///
    /// Use this when you need to update values in a timeseries during simulation.
    pub fn get_data_mut(&mut self, name: &str) -> Option<&mut TimeseriesData> {
        self.get_by_name_mut(name).map(|item| &mut item.data)
    }

    /// Iterate over all timeseries items in the collection.
    ///
    /// Items are returned in sorted order by name for deterministic iteration.
    pub fn iter(&self) -> impl Iterator<Item = &TimeseriesItem> {
        self.timeseries.iter()
    }

    /// Add all items from another collection into this collection
    ///
    /// # Panics
    /// Panics if any item name already exists in this collection
    pub fn extend(&mut self, other: TimeseriesCollection) {
        for item in other.timeseries {
            self.insert(item);
        }
    }

    /// Insert at the item's sorted position, panicking on a duplicate name.
    fn insert(&mut self, item: TimeseriesItem) {
        match self
            .timeseries
            .binary_search_by(|x| x.name.as_str().cmp(&item.name))
        {
            Ok(_) => panic!("timeseries {} already exists", item.name),
            Err(index) => self.timeseries.insert(index, item),
        }
    }
}

impl IntoIterator for TimeseriesCollection {
    type Item = TimeseriesItem;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.timeseries.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
    use crate::timeseries::TimeAxis;
    use numpy::array;
    use numpy::ndarray::{Array, Array2};
    use std::sync::Arc;

    #[test]
    fn adding_scalar() {
        let mut collection = TimeseriesCollection::new();

        let timeseries =
            Timeseries::from_values(array![1.0, 2.0, 3.0], Array::range(2020.0, 2023.0, 1.0));
        collection.add_timeseries(
            "Surface Air Temperature Change".to_string(),
            timeseries.clone(),
        );
        collection.add_timeseries("Emissions|CO2".to_string(), timeseries.clone());

        assert_eq!(
            collection
                .get_data("Surface Air Temperature Change")
                .unwrap()
                .grid_size(),
            1
        );
        assert!(collection
            .get_data("Surface Air Temperature Change")
            .unwrap()
            .as_scalar()
            .is_some());
    }

    #[test]
    fn adding_four_box() {
        let mut collection = TimeseriesCollection::new();

        let grid = FourBoxGrid::magicc_standard();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values =
            Array2::from_shape_vec((2, 4), vec![15.0, 14.0, 10.0, 9.0, 16.0, 15.0, 11.0, 10.0])
                .unwrap();

        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "degC".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        collection.add_four_box_timeseries("Temperature".to_string(), ts);

        assert_eq!(collection.get_data("Temperature").unwrap().grid_size(), 4);
        assert!(collection
            .get_data("Temperature")
            .unwrap()
            .as_four_box()
            .is_some());
    }

    #[test]
    fn adding_hemispheric() {
        let mut collection = TimeseriesCollection::new();

        let grid = HemisphericGrid::equal_weights();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values = Array2::from_shape_vec((2, 2), vec![15.0, 10.0, 16.0, 11.0]).unwrap();

        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "degC".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        collection.add_hemispheric_timeseries("Temperature".to_string(), ts);

        assert_eq!(collection.get_data("Temperature").unwrap().grid_size(), 2);
        assert!(collection
            .get_data("Temperature")
            .unwrap()
            .as_hemispheric()
            .is_some());
    }

    #[test]
    fn mixed_collection() {
        let mut collection = TimeseriesCollection::new();

        // Add scalar
        let scalar = Timeseries::from_values(array![280.0, 285.0], array![2000.0, 2001.0]);
        collection.add_timeseries("CO2".to_string(), scalar);

        // Add four-box
        let grid = FourBoxGrid::magicc_standard();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values =
            Array2::from_shape_vec((2, 4), vec![15.0, 14.0, 10.0, 9.0, 16.0, 15.0, 11.0, 10.0])
                .unwrap();
        let four_box = GridTimeseries::new(
            values,
            time_axis.clone(),
            grid,
            "degC".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        collection.add_four_box_timeseries("Temperature".to_string(), four_box);

        // Add hemispheric
        let grid = HemisphericGrid::equal_weights();
        let values = Array2::from_shape_vec((2, 2), vec![500.0, 450.0, 510.0, 460.0]).unwrap();
        let hemispheric = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "W/m^2".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        collection.add_hemispheric_timeseries("Radiation".to_string(), hemispheric);

        // Verify all exist with correct grid sizes
        assert_eq!(collection.get_data("CO2").unwrap().grid_size(), 1);
        assert_eq!(collection.get_data("Temperature").unwrap().grid_size(), 4);
        assert_eq!(collection.get_data("Radiation").unwrap().grid_size(), 2);
    }

    #[test]
    #[should_panic]
    fn adding_same_name() {
        let mut collection = TimeseriesCollection::new();

        let timeseries =
            Timeseries::from_values(array![1.0, 2.0, 3.0], Array::range(2020.0, 2023.0, 1.0));
        collection.add_timeseries("test".to_string(), timeseries.clone());
        collection.add_timeseries("test".to_string(), timeseries.clone());
    }
}

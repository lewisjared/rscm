use crate::spatial::{FourBoxGrid, HemisphericGrid, ScalarGrid, SpatialGrid};
use crate::timeseries::{FloatValue, GridTimeseries, Timeseries};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, PartialOrd, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[pyo3::pyclass]
pub enum VariableType {
    /// Values that are defined outside of the model
    Exogenous,
    /// Values that are determined within the model
    Endogenous,
}

/// Container for timeseries data that can be either scalar or grid-based
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeseriesData {
    /// Scalar (global-only) timeseries
    Scalar(Timeseries<FloatValue>),
    /// Four-box regional timeseries (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
    FourBox(GridTimeseries<FloatValue, FourBoxGrid>),
    /// Hemispheric timeseries (Northern, Southern)
    Hemispheric(GridTimeseries<FloatValue, HemisphericGrid>),
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesItem {
    #[serde(alias = "timeseries")]
    pub data: TimeseriesData,
    pub name: String,
    pub variable_type: VariableType,
}

/// A collection of time series data.
/// Allows for easy access to time series data by name across the whole model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesCollection {
    timeseries: Vec<TimeseriesItem>,
}

impl Default for TimeseriesCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeseriesCollection {
    pub fn new() -> Self {
        Self {
            timeseries: Vec::new(),
        }
    }

    /// Add a new scalar timeseries to the collection
    ///
    /// # Panics
    /// Panics if a timeseries with the same name already exists in the collection
    pub fn add_timeseries(
        &mut self,
        name: String,
        timeseries: Timeseries<FloatValue>,
        variable_type: VariableType,
    ) {
        if self.timeseries.iter().any(|x| x.name == name) {
            panic!("timeseries {} already exists", name)
        }
        self.timeseries.push(TimeseriesItem {
            data: TimeseriesData::Scalar(timeseries),
            name,
            variable_type,
        });
        // Ensure the order of the serialised timeseries is stable
        self.timeseries.sort_unstable_by_key(|x| x.name.clone());
    }

    /// Add a new four-box grid timeseries to the collection
    ///
    /// # Panics
    /// Panics if a timeseries with the same name already exists in the collection
    pub fn add_four_box_timeseries(
        &mut self,
        name: String,
        timeseries: GridTimeseries<FloatValue, FourBoxGrid>,
        variable_type: VariableType,
    ) {
        if self.timeseries.iter().any(|x| x.name == name) {
            panic!("timeseries {} already exists", name)
        }
        self.timeseries.push(TimeseriesItem {
            data: TimeseriesData::FourBox(timeseries),
            name,
            variable_type,
        });
        self.timeseries.sort_unstable_by_key(|x| x.name.clone());
    }

    /// Add a new hemispheric grid timeseries to the collection
    ///
    /// # Panics
    /// Panics if a timeseries with the same name already exists in the collection
    pub fn add_hemispheric_timeseries(
        &mut self,
        name: String,
        timeseries: GridTimeseries<FloatValue, HemisphericGrid>,
        variable_type: VariableType,
    ) {
        if self.timeseries.iter().any(|x| x.name == name) {
            panic!("timeseries {} already exists", name)
        }
        self.timeseries.push(TimeseriesItem {
            data: TimeseriesData::Hemispheric(timeseries),
            name,
            variable_type,
        });
        self.timeseries.sort_unstable_by_key(|x| x.name.clone());
    }

    pub fn get_by_name(&self, name: &str) -> Option<&TimeseriesItem> {
        self.timeseries.iter().find(|x| x.name == name)
    }

    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut TimeseriesItem> {
        self.timeseries.iter_mut().find(|x| x.name == name)
    }

    /// Get the timeseries data for a variable by name
    pub fn get_data(&self, name: &str) -> Option<&TimeseriesData> {
        self.get_by_name(name).map(|item| &item.data)
    }

    /// Get mutable timeseries data for a variable by name
    pub fn get_data_mut(&mut self, name: &str) -> Option<&mut TimeseriesData> {
        self.get_by_name_mut(name).map(|item| &mut item.data)
    }

    /// Get a scalar timeseries by name (backwards compatibility)
    ///
    /// Returns None if the variable is not scalar or doesn't exist.
    ///
    /// # Deprecated
    /// Use `get_data()` for grid-aware access
    #[deprecated(since = "0.3.0", note = "Use get_data() for grid-aware access")]
    pub fn get_timeseries_by_name(&self, name: &str) -> Option<&Timeseries<FloatValue>> {
        self.get_data(name).and_then(|data| data.as_scalar())
    }

    /// Get a mutable scalar timeseries by name (backwards compatibility)
    ///
    /// Returns None if the variable is not scalar or doesn't exist.
    ///
    /// # Deprecated
    /// Use `get_data_mut()` for grid-aware access
    #[deprecated(since = "0.3.0", note = "Use get_data_mut() for grid-aware access")]
    pub fn get_timeseries_by_name_mut(
        &mut self,
        name: &str,
    ) -> Option<&mut Timeseries<FloatValue>> {
        self.get_data_mut(name).and_then(|data| match data {
            TimeseriesData::Scalar(ts) => Some(ts),
            _ => None,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &TimeseriesItem> {
        self.timeseries.iter()
    }

    /// Add all items from another collection into this collection
    ///
    /// # Panics
    /// Panics if any item name already exists in this collection
    pub fn extend(&mut self, other: TimeseriesCollection) {
        for item in other.timeseries {
            if self.timeseries.iter().any(|x| x.name == item.name) {
                panic!("timeseries {} already exists", item.name)
            }
            self.timeseries.push(item);
        }
        self.timeseries.sort_unstable_by_key(|x| x.name.clone());
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
            "Surface Temperature".to_string(),
            timeseries.clone(),
            VariableType::Exogenous,
        );
        collection.add_timeseries(
            "Emissions|CO2".to_string(),
            timeseries.clone(),
            VariableType::Endogenous,
        );

        assert_eq!(
            collection
                .get_data("Surface Temperature")
                .unwrap()
                .grid_size(),
            1
        );
        assert!(collection
            .get_data("Surface Temperature")
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
            "°C".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        collection.add_four_box_timeseries(
            "Temperature|FourBox".to_string(),
            ts,
            VariableType::Endogenous,
        );

        assert_eq!(
            collection
                .get_data("Temperature|FourBox")
                .unwrap()
                .grid_size(),
            4
        );
        assert!(collection
            .get_data("Temperature|FourBox")
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
            "°C".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        collection.add_hemispheric_timeseries(
            "Temperature|Hemispheric".to_string(),
            ts,
            VariableType::Endogenous,
        );

        assert_eq!(
            collection
                .get_data("Temperature|Hemispheric")
                .unwrap()
                .grid_size(),
            2
        );
        assert!(collection
            .get_data("Temperature|Hemispheric")
            .unwrap()
            .as_hemispheric()
            .is_some());
    }

    #[test]
    fn mixed_collection() {
        let mut collection = TimeseriesCollection::new();

        // Add scalar
        let scalar = Timeseries::from_values(array![280.0, 285.0], array![2000.0, 2001.0]);
        collection.add_timeseries("CO2|Global".to_string(), scalar, VariableType::Endogenous);

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
            "°C".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        collection.add_four_box_timeseries(
            "Temperature|FourBox".to_string(),
            four_box,
            VariableType::Endogenous,
        );

        // Add hemispheric
        let grid = HemisphericGrid::equal_weights();
        let values = Array2::from_shape_vec((2, 2), vec![500.0, 450.0, 510.0, 460.0]).unwrap();
        let hemispheric = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "W/m²".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        collection.add_hemispheric_timeseries(
            "Radiation|Hemispheric".to_string(),
            hemispheric,
            VariableType::Endogenous,
        );

        // Verify all exist with correct grid sizes
        assert_eq!(collection.get_data("CO2|Global").unwrap().grid_size(), 1);
        assert_eq!(
            collection
                .get_data("Temperature|FourBox")
                .unwrap()
                .grid_size(),
            4
        );
        assert_eq!(
            collection
                .get_data("Radiation|Hemispheric")
                .unwrap()
                .grid_size(),
            2
        );
    }

    #[test]
    #[should_panic]
    fn adding_same_name() {
        let mut collection = TimeseriesCollection::new();

        let timeseries =
            Timeseries::from_values(array![1.0, 2.0, 3.0], Array::range(2020.0, 2023.0, 1.0));
        collection.add_timeseries(
            "test".to_string(),
            timeseries.clone(),
            VariableType::Exogenous,
        );
        collection.add_timeseries(
            "test".to_string(),
            timeseries.clone(),
            VariableType::Endogenous,
        );
    }
}

use crate::timeseries::{FloatValue, Timeseries};
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, PartialOrd, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[pyo3::pyclass]
pub enum VariableType {
    /// Values that are defined outside of the model
    Exogenous,
    /// Values that are determined within the model
    Endogenous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesItem {
    pub timeseries: Timeseries<FloatValue>,
    pub name: String,
    pub variable_type: VariableType,
}

/// A collection of time series data.
/// Allows for easy access to time series data by name across the whole model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesCollection {
    node_indexes: Vec<NodeIndex>,
    graph: Graph<TimeseriesItem, f64>,
}

impl Default for TimeseriesCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeseriesCollection {
    pub fn new() -> Self {
        Self {
            node_indexes: Vec::new(),
            graph: Graph::default(),
        }
    }

    /// Add a new timeseries to the collection
    ///
    /// Panics if a timeseries with the same name already exists in the collection
    /// TODO: Revisit if this is the correct way of handling this type of error
    pub fn add_timeseries(
        &mut self,
        name: String,
        timeseries: Timeseries<FloatValue>,
        variable_type: VariableType,
    ) {
        self.iter().for_each(|x| {
            if x.name == name {
                panic!("timeseries {} already exists", name)
            }
        });

        let node_index = self.graph.add_node(TimeseriesItem {
            timeseries,
            name,
            variable_type,
        });
        self.node_indexes.push(node_index);
    }

    pub fn add_nested_timeseries(&mut self, parent_name: String, child_name: String) {
        self.add_nested_timeseries_with_weight(parent_name, child_name, 1.0);
    }

    pub fn add_nested_timeseries_with_weight(
        &mut self,
        parent_name: String,
        child_name: String,
        weight: f64,
    ) {
        let parent = self
            .get_by_name(&parent_name)
            .expect("Parent timeseries not found");
        let timeseries = parent.timeseries.clone();
        {
            self.add_timeseries(child_name.clone(), timeseries, parent.variable_type);
        }

        self.graph.add_edge(
            *self.get_index(&parent_name),
            *self.get_index(&child_name),
            weight,
        );
    }

    fn get_index(&self, name: &str) -> &NodeIndex {
        self.node_indexes
            .iter()
            .find(|x| self.graph[**x].name == name)
            .expect("Timeseries not found")
    }

    pub fn get_by_name(&self, name: &str) -> Option<&TimeseriesItem> {
        self.node_indexes
            .iter()
            .find(|x| self.graph[**x].name == name)
            .map(|x| &self.graph[*x])
    }

    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut TimeseriesItem> {
        self.node_indexes
            .iter()
            .find(|x| self.graph[**x].name == name)
            .map(|x| &mut self.graph[*x])
    }

    pub fn get_timeseries_by_name(&self, name: &str) -> Option<&Timeseries<FloatValue>> {
        self.get_by_name(name).map(|item| &item.timeseries)
    }
    fn get_timeseries_by_name_mut(&mut self, name: &str) -> Option<&mut Timeseries<FloatValue>> {
        self.node_indexes
            .iter()
            .find(|x| self.graph[**x].name == name)
            .map(|x| &mut self.graph[*x])
            .map(|item| &mut item.timeseries)
    }

    pub fn set_value(&mut self, name: &str, time_index: usize, value: FloatValue) {
        self.get_timeseries_by_name_mut(name)
            .expect("Timeseries not found")
            .set(time_index, value);
    }

    pub fn iter(&self) -> impl Iterator<Item = &TimeseriesItem> {
        self.node_indexes.iter().map(move |x| &self.graph[*x])
    }
}

impl IntoIterator for TimeseriesCollection {
    type Item = TimeseriesItem;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.node_indexes
            .iter()
            .map(move |x| self.graph[*x].clone())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::array;
    use numpy::ndarray::Array;

    #[test]
    fn adding() {
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
    }

    #[test]
    fn hierarchical_timeseries() {
        let mut collection = TimeseriesCollection::new();

        let timeseries =
            Timeseries::from_values(array![1.0, 2.0, 3.0], Array::range(2020.0, 2023.0, 1.0));
        collection.add_timeseries(
            "Emissions|CO2".to_string(),
            timeseries.clone(),
            VariableType::Exogenous,
        );

        collection.add_nested_timeseries_with_weight(
            "Emissions|CO2".to_string(),
            "Emissions|CO2|Fossil and Industrial".to_string(),
            1.0,
        );
        collection.add_nested_timeseries_with_weight(
            "Emissions|CO2".to_string(),
            "Emissions|CO2|LULUCF".to_string(),
            1.0,
        );

        let ts = collection
            .get_timeseries_by_name("Emissions|CO2|Fossil and Industrial")
            .unwrap();
        assert_eq!(ts.time_axis().values(), array![2020.0, 2021.0, 2022.0]);
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

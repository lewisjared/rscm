use crate::timeseries::{FloatValue, Timeseries};
use petgraph::dot::{Config, Dot};
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

/// Propoate changes to the children
///
/// A closure is used to determine the new value of the child timeseries.
/// This is somewhat inefficient as it requires unnecessary traversals of the graph,
/// but it is simple and should be sufficient for now.
fn propagate_changes<F>(
    graph: &mut Graph<TimeseriesItem, FloatValue>,
    direction: petgraph::Direction,
    node: NodeIndex,
    time_index: usize,
    f: F,
) where
    F: Fn(FloatValue, FloatValue, FloatValue, FloatValue) -> FloatValue,
{
    // Calculate the sum of the edge weights
    let edge_sum = graph
        .edges_directed(node, direction)
        .map(|edge| *edge.weight())
        .sum::<FloatValue>();

    // Calculate the sum of the neighbour values
    let neighbour_sum = graph
        .neighbors_directed(node, direction)
        .map(|node| graph[node].timeseries.at(time_index).unwrap())
        .sum::<FloatValue>();

    let mut child_walker = graph.neighbors_directed(node, direction).detach();

    while let Some((edge, node)) = child_walker.next(graph) {
        let edge_weight = *graph.edge_weight(edge).unwrap();
        let node = &mut graph[node];
        let current_value = node.timeseries.at(time_index).unwrap();

        // Update the child timeseries with the new value calculated by the closure
        node.timeseries.set(
            time_index,
            f(current_value, edge_weight, edge_sum, neighbour_sum),
        );
    }
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
    ) -> NodeIndex {
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
        node_index
    }

    pub fn add_nested_timeseries(&mut self, parent_name: String, child_name: String) {
        self.add_nested_timeseries_with_weight(parent_name, child_name, 1.0);
    }

    /// Adds a nested timeseries to the collection
    ///
    /// Adds a new timeseries to the collection
    /// and creates a parent-child relationship between the two
    /// where the parent is always the sum of the children values.
    /// All parent-child timeseries share a common time axis to simplify operations.
    ///
    /// The weight is used to determine how the parent timeseries
    /// is divided up between the children and reciprocally,
    /// how the parent total is impacted by the children.
    ///
    /// parent = sum(children * weight)
    ///
    /// # Missing values
    ///
    /// Timeseries differentiate between missing values (nan) and zero quantities.
    /// If any of the children have missing values,
    /// the value for the parent will be nan.
    ///
    /// # Initialisation
    ///
    /// If the parent already has values then the first child will be initialised
    /// with those same values, subsequent children will be initialised with zeros
    /// to maintain the relationship parent total.
    ///
    /// This initialisation functionality can be a bit confusing.
    /// This function may be moved to a builder pattern once the API is more stable.
    /// The builder will be responsible for defining the expected timeseries and
    /// the relationships between them.
    /// This would also give more freedom to use more efficient data structures if
    /// performance becomes an issue.
    pub fn add_nested_timeseries_with_weight(
        &mut self,
        parent_name: String,
        child_name: String,
        weight: f64,
    ) {
        assert_eq!(weight, 1.0, "Only unity weights are currently supported");
        let parent = self
            .get_by_name(&parent_name)
            .expect("Parent timeseries not found");

        let parent_index = self.get_index(&parent_name);
        let has_neighbours = self.graph.neighbors(parent_index).count() > 0;

        let mut timeseries = parent.timeseries.clone();
        if has_neighbours {
            // Other children already exist, initialise with zeros to maintain the parent values
            // This is a choice. We could also initialise with nans to indicate missing values,
            // which would then propagate to parents
            timeseries.fill(0.0)
        }

        // Add the new child node and add an edge from the parent
        let child_node = self.add_timeseries(child_name, timeseries, parent.variable_type);
        self.graph.add_edge(parent_index, child_node, weight);
    }

    fn get_index(&self, name: &str) -> NodeIndex {
        *self
            .node_indexes
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
        let timeseries = self
            .get_timeseries_by_name_mut(name)
            .expect("Timeseries not found");
        let old_value = timeseries.at(time_index).unwrap();

        timeseries.set(time_index, value);

        let target = self.get_index(name);
        // Propagate the change to the parent
        propagate_changes(
            &mut self.graph,
            petgraph::Direction::Incoming,
            target,
            time_index,
            |parent_value, edge_weight, _, _| parent_value + (value - old_value) / edge_weight,
        );

        // Propagate the change to the children timeseries
        // The children should maintain their relative proportions,
        // i.e. apply a scale factor ignoring edge weights
        propagate_changes(
            &mut self.graph,
            petgraph::Direction::Outgoing,
            target,
            time_index,
            |child_value, _, _, children_sum| value * child_value / children_sum,
        );
    }

    pub fn iter(&self) -> impl Iterator<Item = &TimeseriesItem> {
        self.node_indexes.iter().map(move |x| &self.graph[*x])
    }

    pub fn to_dot(&self) -> petgraph::dot::Dot<&Graph<TimeseriesItem, f64>> {
        Dot::with_attr_getters(
            &self.graph,
            &[Config::NodeNoLabel, Config::EdgeNoLabel],
            &|_, er| format!("label = {:?}", er.weight()),
            &|_, (_, component)| format!("label = {:?}", component.name),
        )
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

        // Independant timeseries
        collection.add_timeseries(
            "Surface Temperature".to_string(),
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
        // First child is allocated the parent's value
        assert_eq!(
            collection
                .get_timeseries_by_name("Emissions|CO2|Fossil and Industrial")
                .unwrap()
                .at(0)
                .unwrap(),
            1.0
        );
        // Subsequent child have 0 values
        assert_eq!(
            collection
                .get_timeseries_by_name("Emissions|CO2|LULUCF")
                .unwrap()
                .at(0)
                .unwrap(),
            0.0
        );

        collection.set_value("Emissions|CO2|Fossil and Industrial", 0, 2.0);
        collection.set_value("Emissions|CO2|LULUCF", 0, 27.0);
        assert_eq!(
            collection
                .get_timeseries_by_name("Emissions|CO2")
                .unwrap()
                .at(0)
                .unwrap(),
            2.0 + 27.0
        );

        // Updating the parent should update the children
        // Scale the emissions back to 3.0
        collection.set_value("Emissions|CO2", 0, 3.0);
        assert_eq!(
            collection
                .get_timeseries_by_name("Emissions|CO2|Fossil and Industrial")
                .unwrap()
                .at(0)
                .unwrap(),
            2.0 / (2.0 + 27.0) * 3.0
        );
        assert_eq!(
            collection
                .get_timeseries_by_name("Emissions|CO2|LULUCF")
                .unwrap()
                .at(0)
                .unwrap(),
            27.0 / (2.0 + 27.0) * 3.0
        );

        println!("{:?}", collection.to_dot());
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

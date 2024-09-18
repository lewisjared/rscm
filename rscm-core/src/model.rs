/// A model consists of a series of coupled components which are solved together.
/// The model orchastrates the passing of state between different components.
/// Each component is solved for a given time step in an order determined by their
/// dependencies.
/// Once all components and state is solved for, the model will move to the next time step.
/// The state from previous steps is preserved as it is useful as output or in the case where
/// a component needs previous values.
///
/// The model also holds all of the exogenous variables required by the model.
/// The required variables are identified when building the model.
/// If a required exogenous variable isn't provided, then the build step will fail.
use crate::component::{
    Component, InputState, OutputState, RequirementDefinition, RequirementType, State,
};
use crate::errors::RSCMResult;
use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
use crate::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use crate::timeseries_collection::{TimeseriesCollection, VariableType};
use numpy::ndarray::Array;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::{Bfs, IntoNeighbors, IntoNodeIdentifiers, Visitable};
use petgraph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;

type C = Arc<dyn Component>;
type CGraph = Graph<C, RequirementDefinition>;

#[derive(Debug)]
struct VariableDefinition {
    name: String,
    unit: String,
}

impl VariableDefinition {
    fn from_requirement_definition(definition: &RequirementDefinition) -> Self {
        Self {
            name: definition.name.clone(),
            unit: definition.unit.clone(),
        }
    }
}

/// A null component that does nothing
///
/// Used as an initial component to ensure that the model is connected
#[derive(Debug, Serialize, Deserialize)]
struct NullComponent {}

#[typetag::serde]
impl Component for NullComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        Ok(OutputState::from(input_state.clone()))
    }
}

/// Build a new model from a set of components
///
/// The builder generates a graph that defines the inter-component dependencies
/// and determines what variables are endogenous and exogenous to the model.
/// This graph is used by the model to define the order in which components are solved.
///
/// # Examples
/// TODO: figure out how to share example components throughout the docs
pub struct ModelBuilder {
    components: Vec<C>,
    exogenous_variables: TimeseriesCollection,
    initial_values: InputState,
    pub time_axis: Arc<TimeAxis>,
}

/// Checks if the new definition is valid
///
/// If any definitions share a name then the units must be equivalent
///
/// Panics if the parameter definition is inconsistent with any existing definitions.
fn verify_definition(
    definitions: &mut HashMap<String, VariableDefinition>,
    definition: &RequirementDefinition,
) {
    let existing = definitions.get(&definition.name);
    match existing {
        Some(existing) => {
            assert_eq!(existing.unit, definition.unit);
        }
        None => {
            definitions.insert(
                definition.name.clone(),
                VariableDefinition::from_requirement_definition(definition),
            );
        }
    }
}

/// Check that a component graph is valid
///
/// We require a directed acyclic graph which doesn't contain any cycles (other than a self-referential node).
/// This avoids the case where component `A` depends on a component `B`,
/// but component `B` also depends on component `A`.
fn is_valid_graph<G>(g: G) -> bool
where
    G: IntoNodeIdentifiers + IntoNeighbors + Visitable,
{
    use petgraph::visit::{depth_first_search, DfsEvent};

    depth_first_search(g, g.node_identifiers(), |event| match event {
        DfsEvent::BackEdge(a, b) => {
            // If the cycle is self-referential then that is fine
            match a == b {
                true => Ok(()),
                false => Err(()),
            }
        }
        _ => Ok(()),
    })
    .is_err()
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            components: vec![],
            initial_values: InputState::empty(),
            exogenous_variables: TimeseriesCollection::new(),
            time_axis: Arc::new(TimeAxis::from_values(Array::range(2000.0, 2100.0, 1.0))),
        }
    }

    /// Register a component with the builder
    pub fn with_component(&mut self, component: Arc<dyn Component + Send + Sync>) -> &mut Self {
        self.components.push(component);
        self
    }

    /// Supply exogenous data to be used by the model
    ///
    /// Any unneeded timeseries will be ignored.
    pub fn with_exogenous_variable(
        &mut self,
        name: &str,
        timeseries: Timeseries<FloatValue>,
    ) -> &mut Self {
        self.exogenous_variables.add_timeseries(
            name.to_string(),
            timeseries,
            VariableType::Exogenous,
        );
        self
    }

    /// Supply exogenous data to be used by the model
    ///
    /// Any unneeded timeseries will be ignored.
    pub fn with_exogenous_collection(&mut self, collection: TimeseriesCollection) -> &mut Self {
        collection.into_iter().for_each(|x| {
            self.exogenous_variables
                .add_timeseries(x.name, x.timeseries, x.variable_type)
        });
        self
    }

    /// Adds some state to the set of initial values
    ///
    /// These initial values are used to provide some initial values at `t_0`.
    /// Initial values are used for requirements which have a type of `RequirementType::InputAndOutput`.
    /// These requirements use state from the current timestep in order to generate a value for the
    /// next timestep.
    /// Building a model where any variables which have `RequirementType::InputAndOutput`, but
    /// do not have an initial value will result in an error.
    pub fn with_initial_values(&mut self, initial_values: InputState) -> &mut Self {
        self.initial_values.merge(initial_values);
        self
    }

    /// Specify the time axis that will be used by the model
    ///
    /// This time axis defines the time steps (including bounds) on which the model will be iterated.
    pub fn with_time_axis(&mut self, time_axis: TimeAxis) -> &mut Self {
        self.time_axis = Arc::new(time_axis);
        self
    }

    /// Builds the component graph for the registered components and creates a concrete model
    ///
    /// Panics if the required data to build a model is not available.
    pub fn build(&self) -> Model {
        // todo: refactor once this is more stable
        let mut graph: CGraph = Graph::new();
        let mut endrogoneous: HashMap<String, NodeIndex> = HashMap::new();
        let mut exogenous: Vec<String> = vec![];
        let mut definitions: HashMap<String, VariableDefinition> = HashMap::new();
        let initial_node = graph.add_node(Arc::new(NullComponent {}));

        self.components.iter().for_each(|component| {
            let node = graph.add_node(component.clone());
            let mut has_dependencies = false;

            let requires = component.inputs();
            let provides = component.outputs();

            requires.iter().for_each(|requirement| {
                verify_definition(&mut definitions, requirement);

                if exogenous.contains(&requirement.name) {
                    // Link to the node that provides the requirement
                    graph.add_edge(endrogoneous[&requirement.name], node, requirement.clone());
                    has_dependencies = true;
                } else {
                    // Add a new variable that must be defined outside of the model
                    exogenous.push(requirement.name.clone())
                }
            });

            if !has_dependencies {
                // If the node has no dependencies on other components,
                // create a link to the initial node.
                // This ensures that we have a single connected graph
                // There might be smarter ways to iterate over the nodes, but this is fine for now
                graph.add_edge(
                    initial_node,
                    node,
                    RequirementDefinition::new("", "", RequirementType::EmptyLink),
                );
            }

            provides.iter().for_each(|requirement| {
                verify_definition(&mut definitions, requirement);

                let val = endrogoneous.get(&requirement.name);

                match val {
                    None => {
                        endrogoneous.insert(requirement.name.clone(), node);
                    }
                    Some(node_index) => {
                        graph.add_edge(*node_index, node, requirement.clone());
                        endrogoneous.insert(requirement.name.clone(), node);
                    }
                }
            });
        });

        // Check that the component graph doesn't contain any loops
        assert!(!is_valid_graph(&graph));

        // Create the timeseries collection using the information from the components
        let mut collection = TimeseriesCollection::new();
        for (name, definition) in definitions {
            assert_eq!(definition.name, name);

            if exogenous.contains(&name) {
                // Exogenous variable is expected to be supplied
                if self.initial_values.has(&name) {
                    // An initial value was provided
                    let mut ts = Timeseries::new_empty(
                        self.time_axis.clone(),
                        definition.unit,
                        InterpolationStrategy::from(LinearSplineStrategy::new(true)),
                    );
                    ts.set(0, *self.initial_values.get(&name));

                    // Note that timeseries that are initialised are defined as Endogenous
                    // all but the first time point come from the model.
                    // This could potentially be defined as a different VariableType if needed.
                    collection.add_timeseries(name, ts, VariableType::Endogenous)
                } else {
                    // Check if the timeseries is available in the provided exogenous variables
                    // then interpolate to the right timebase
                    let timeseries = self.exogenous_variables.get_timeseries_by_name(&name);

                    match timeseries {
                        Some(timeseries) => collection.add_timeseries(
                            name,
                            timeseries
                                .to_owned()
                                .interpolate_into(self.time_axis.clone()),
                            VariableType::Exogenous,
                        ),
                        None => println!("No exogenous data for {}", definition.name),
                    }
                }
            } else {
                // Create a placeholder for data that will be generated by the model
                collection.add_timeseries(
                    definition.name,
                    Timeseries::new_empty(
                        self.time_axis.clone(),
                        definition.unit,
                        InterpolationStrategy::from(LinearSplineStrategy::new(true)),
                    ),
                    VariableType::Endogenous,
                )
            }
        }

        // Add the components to the graph
        Model::new(graph, initial_node, collection, self.time_axis.clone())
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A coupled set of components that are solved on a common time axis.
///
/// These components are solved over time steps defined by the ['time_axis'].
/// Components may pass state between themselves.
/// Each component may require information from other components to be solved (endogenous) or
/// predefined data (exogenous).
///
/// For example, a component to calculate the Effective Radiative Forcing(ERF) of CO_2 may
/// require CO_2 concentrations as input state and provide CO_2 ERF.
/// The component is agnostic about where/how that state is defined.
/// If the model has no components which provide CO_2 concentrations,
/// then a CO_2 concentration timeseries must be defined externally.
/// If the model also contains a carbon cycle component which produced CO_2 concentrations,
/// then the ERF component will be solved after the carbon cycle model.
#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    /// A directed graph with components as nodes and the edges defining the state dependencies
    /// between nodes.
    /// This graph is traversed on every time step to ensure that any state dependencies are
    /// solved before another component needs the state.
    components: CGraph,
    /// The base node of the graph from where to begin traversing.
    initial_node: NodeIndex,
    /// The model state
    ///
    /// Variable names within the model are unique and these variable names are used by
    /// components to request state.
    collection: TimeseriesCollection,
    time_axis: Arc<TimeAxis>,
    time_index: usize,
}

impl Model {
    pub fn new(
        components: CGraph,
        initial_node: NodeIndex,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
    ) -> Self {
        Self {
            components,
            initial_node,
            collection,
            time_axis,
            time_index: 0,
        }
    }

    /// Gets the time value at the current step
    pub fn current_time(&self) -> Time {
        self.time_axis.at(self.time_index).unwrap()
    }
    pub fn current_time_bounds(&self) -> (Time, Time) {
        self.time_axis.at_bounds(self.time_index).unwrap()
    }

    /// Solve a single component for the current timestep
    ///
    /// The updated state from the component is then pushed into the model's timeseries collection
    /// to be later used by other components.
    /// The output state defines the values at the next time index as it represents the state
    /// at the start of the next timestep.
    fn step_model_component(&mut self, component: C) {
        let input_state = component.extract_state(&self.collection, self.current_time());

        let (start, end) = self.current_time_bounds();

        let result = component.solve(start, end, &input_state);

        match result {
            Ok(output_state) => output_state.iter().for_each(|(key, value)| {
                let ts = self.collection.get_timeseries_by_name_mut(key).unwrap();
                // The next time index is used as this output state represents the value of a
                // variable at the end of the current time step.
                // This is the same as the start of the next timestep.
                ts.set(self.time_index + 1, *value)
            }),
            Err(err) => {
                println!("Solving failed: {}", err)
            }
        }
    }

    /// Step the model forward a step by solving each component for the current time step.
    ///
    /// A breadth-first search across the component graph starting at the initial node
    /// will solve the components in a way that ensures any models with dependencies are solved
    /// after the dependent component is first solved.
    fn step_model(&mut self) {
        let mut bfs = Bfs::new(&self.components, self.initial_node);
        while let Some(nx) = bfs.next(&self.components) {
            let c = self.components.index(nx);
            self.step_model_component(c.clone())
        }
    }

    /// Steps the model forward one time step
    ///
    /// This solves the current time step and then updates the index.
    pub fn step(&mut self) {
        assert!(self.time_index < self.time_axis.len() - 1);
        self.step_model();

        self.time_index += 1;
    }

    /// Steps the model until the end of the time axis
    pub fn run(&mut self) {
        while self.time_index < self.time_axis.len() - 1 {
            self.step();
        }
    }

    /// Create a diagram the represents the component graph
    ///
    /// Useful for debugging
    pub fn as_dot(&self) -> Dot<&CGraph> {
        Dot::with_attr_getters(
            &self.components,
            &[Config::NodeNoLabel, Config::EdgeNoLabel],
            &|_, er| format!("label = {:?}", er.weight().name),
            &|_, (_, component)| format!("label = \"{:?}\"", component),
        )
    }

    /// Returns true if the model has no more time steps to process
    pub fn finished(&self) -> bool {
        self.time_index == self.time_axis.len() - 1
    }

    pub fn timeseries(&self) -> &TimeseriesCollection {
        &self.collection
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example_components::{TestComponent, TestComponentParameters};
    use crate::interpolate::strategies::PreviousStrategy;
    use is_close::is_close;
    use numpy::array;
    use numpy::ndarray::Array;
    use std::iter::zip;

    fn get_emissions() -> Timeseries<FloatValue> {
        Timeseries::new(
            array![0.0, 10.0],
            Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
            "GtC / yr".to_string(),
            InterpolationStrategy::from(PreviousStrategy::new(true)),
        )
    }

    #[test]
    fn step() {
        let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
        let mut model = ModelBuilder::new()
            .with_time_axis(time_axis)
            .with_component(Arc::new(TestComponent::from_parameters(
                TestComponentParameters { p: 0.5 },
            )))
            .with_exogenous_variable("Emissions|CO2", get_emissions())
            .build();

        assert_eq!(model.time_index, 0);
        model.step();
        model.step();
        assert_eq!(model.time_index, 2);
        assert_eq!(model.current_time(), 2022.0);
        model.run();
        assert_eq!(model.time_index, 4);
        assert!(model.finished());

        let concentrations = model
            .collection
            .get_timeseries_by_name("Concentrations|CO2")
            .unwrap();

        println!("{:?}", concentrations.values());

        // The first value for an endogenous timeseries without a y0 value is NaN.
        // This is because the values in the timeseries represents the state at the start
        // of a time step.
        // Since the values from t-1 aren't known we can't solve for y0
        assert!(concentrations.at(0).unwrap().is_nan());
        let mut iter = concentrations.values().into_iter();
        iter.next(); // Skip the first value
        assert!(iter.all(|x| !x.is_nan()));
    }

    #[test]
    fn dot() {
        let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
        let model = ModelBuilder::new()
            .with_time_axis(time_axis)
            .with_component(Arc::new(TestComponent::from_parameters(
                TestComponentParameters { p: 0.5 },
            )))
            .with_exogenous_variable("Emissions|CO2", get_emissions())
            .build();

        let exp = "digraph {
    0 [ ]
    1 [ label = \"TestComponent { parameters: TestComponentParameters { p: 0.5 } }\"]
    0 -> 1 [ ]
}
";

        let res = format!("{:?}", model.as_dot());
        assert_eq!(res, exp);
    }

    #[test]
    fn serialise_and_deserialise_model() {
        let mut model = ModelBuilder::new()
            .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
            .with_component(Arc::new(TestComponent::from_parameters(
                TestComponentParameters { p: 0.5 },
            )))
            .with_exogenous_variable("Emissions|CO2", get_emissions())
            .build();

        model.step();

        let serialised = serde_json::to_string_pretty(&model).unwrap();
        println!("{}", serialised);
        let serialised = toml::to_string(&model).unwrap();

        let expected = r#"initial_node = 0
time_index = 1

[components]
node_holes = []
edge_property = "directed"
edges = [[0, 1, { name = "", unit = "", requirement_type = "EmptyLink" }]]

[[components.nodes]]
type = "NullComponent"

[[components.nodes]]
type = "TestComponent"

[components.nodes.parameters]
p = 0.5

[[collection.timeseries]]
name = "Concentrations|CO2"
variable_type = "Endogenous"

[collection.timeseries.timeseries]
units = "ppm"
latest = 1
interpolation_strategy = "Linear"

[collection.timeseries.timeseries.values]
v = 1
dim = [5]
data = [nan, 0.65, nan, nan, nan]

[collection.timeseries.timeseries.time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]

[[collection.timeseries]]
name = "Emissions|CO2"
variable_type = "Exogenous"

[collection.timeseries.timeseries]
units = "GtC / yr"
latest = 5
interpolation_strategy = "Previous"

[collection.timeseries.timeseries.values]
v = 1
dim = [5]
data = [10.0, 10.0, 10.0, 10.0, 10.0]

[collection.timeseries.timeseries.time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]

[time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]
"#;

        assert_eq!(serialised, expected);

        let deserialised = toml::from_str::<Model>(&serialised).unwrap();

        assert!(zip(
            model
                .collection
                .get_timeseries_by_name("Emissions|CO2")
                .unwrap()
                .values(),
            deserialised
                .collection
                .get_timeseries_by_name("Emissions|CO2")
                .unwrap()
                .values()
        )
        .all(|(x0, x1)| { is_close!(*x0, *x1) || (x0.is_nan() && x0.is_nan()) }))
    }
}

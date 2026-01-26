//! Model struct and runtime execution.

use crate::component::GridType;
use crate::state::{ReadTransformInfo, StateValue, TransformContext};
use crate::timeseries::{Time, TimeAxis};
use crate::timeseries_collection::TimeseriesCollection;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::Bfs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;

use super::state_extraction::{extract_state, extract_state_with_transforms};
use super::transformations::aggregate_state_value;
use super::types::{CGraph, RequiredTransformation, C};

/// A coupled set of components that are solved on a common time axis.
///
/// These components are solved over time steps defined by the [`TimeAxis`].
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
    /// The model state.
    ///
    /// Variable names within the model are unique and these variable names are used by
    /// components to request state.
    collection: TimeseriesCollection,
    time_axis: Arc<TimeAxis>,
    time_index: usize,
    /// Custom weights for grid aggregation, keyed by grid type.
    ///
    /// Used for grid transformations during model execution.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    grid_weights: HashMap<GridType, Vec<f64>>,
    /// Read-side transformations: variable name -> transformation needed when component reads.
    ///
    /// When a component reads a variable at a coarser grid than the schema declares,
    /// this maps the variable name to the transformation needed (e.g., FourBox -> Scalar).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    read_transforms: HashMap<String, RequiredTransformation>,
    /// Write-side transformations: variable name -> transformation needed when component writes.
    ///
    /// When a component writes a variable at a finer grid than the schema declares,
    /// this maps the variable name to the transformation needed (e.g., FourBox -> Scalar).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    write_transforms: HashMap<String, RequiredTransformation>,
}

impl Model {
    /// Create a new Model with the given components and collection.
    pub fn new(
        components: CGraph,
        initial_node: NodeIndex,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
    ) -> Self {
        Self::with_grid_weights(
            components,
            initial_node,
            collection,
            time_axis,
            HashMap::new(),
        )
    }

    /// Create a new Model with custom grid weights.
    ///
    /// The grid_weights are used for grid transformations during model execution.
    pub fn with_grid_weights(
        components: CGraph,
        initial_node: NodeIndex,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
        grid_weights: HashMap<GridType, Vec<f64>>,
    ) -> Self {
        Self::with_transforms(
            components,
            initial_node,
            collection,
            time_axis,
            grid_weights,
            HashMap::new(),
            HashMap::new(),
        )
    }

    /// Create a new Model with grid weights and transformations.
    ///
    /// This is the full constructor that includes both grid weights and the
    /// read/write transformations for automatic grid aggregation.
    pub fn with_transforms(
        components: CGraph,
        initial_node: NodeIndex,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
        grid_weights: HashMap<GridType, Vec<f64>>,
        read_transforms: HashMap<String, RequiredTransformation>,
        write_transforms: HashMap<String, RequiredTransformation>,
    ) -> Self {
        Self {
            components,
            initial_node,
            collection,
            time_axis,
            time_index: 0,
            grid_weights,
            read_transforms,
            write_transforms,
        }
    }

    /// Get the configured grid weights.
    ///
    /// Returns the custom weights if configured, or None if using defaults.
    pub fn get_grid_weights(&self, grid_type: GridType) -> Option<&Vec<f64>> {
        self.grid_weights.get(&grid_type)
    }

    /// Get all required transformations for introspection.
    ///
    /// Returns a vector of all required transformations, both read-side and write-side.
    /// This is useful for debugging and understanding what grid aggregations will occur.
    pub fn required_transformations(&self) -> Vec<&RequiredTransformation> {
        self.read_transforms
            .values()
            .chain(self.write_transforms.values())
            .collect()
    }

    /// Get read-side transformations.
    ///
    /// These transformations aggregate data when components read variables at coarser
    /// resolutions than the schema declares.
    pub fn read_transforms(&self) -> &HashMap<String, RequiredTransformation> {
        &self.read_transforms
    }

    /// Get write-side transformations.
    ///
    /// These transformations aggregate data when components write variables at finer
    /// resolutions than the schema declares.
    pub fn write_transforms(&self) -> &HashMap<String, RequiredTransformation> {
        &self.write_transforms
    }

    /// Gets the time value at the current step.
    pub fn current_time(&self) -> Time {
        self.time_axis.at(self.time_index).unwrap()
    }

    /// Gets the time bounds at the current step.
    pub fn current_time_bounds(&self) -> (Time, Time) {
        self.time_axis.at_bounds(self.time_index).unwrap()
    }

    /// Solve a single component for the current timestep.
    ///
    /// The updated state from the component is then pushed into the model's timeseries collection
    /// to be later used by other components.
    /// The output state defines the values at the next time index as it represents the state
    /// at the start of the next timestep.
    fn step_model_component(&mut self, component: C) {
        // Build transform context for read-side aggregation
        let input_names = component.input_names();
        let input_state = if self.read_transforms.is_empty() {
            extract_state(&self.collection, input_names, self.current_time())
        } else {
            // Build transform context with only the transforms relevant to this component's inputs
            let mut read_transform_info = HashMap::new();
            for name in &input_names {
                if let Some(transform) = self.read_transforms.get(name) {
                    read_transform_info.insert(
                        name.clone(),
                        ReadTransformInfo {
                            source_grid: transform.source_grid,
                            weights: self.grid_weights.get(&transform.source_grid).cloned(),
                        },
                    );
                }
            }

            if read_transform_info.is_empty() {
                extract_state(&self.collection, input_names, self.current_time())
            } else {
                let context = TransformContext {
                    read_transforms: read_transform_info,
                };
                extract_state_with_transforms(
                    &self.collection,
                    input_names,
                    self.current_time(),
                    context,
                )
            }
        };

        let (start, end) = self.current_time_bounds();

        let result = component.solve(start, end, &input_state);

        match result {
            Ok(output_state) => {
                for (key, state_value) in output_state.iter() {
                    let data = self.collection.get_data_mut(key).unwrap();

                    // Apply write-side transformation if needed (component produces finer grid
                    // than schema expects)
                    let final_value = if let Some(transform) = self.write_transforms.get(key) {
                        let weights = self.grid_weights.get(&transform.source_grid);
                        match aggregate_state_value(
                            state_value,
                            transform.source_grid,
                            transform.target_grid,
                            weights,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                println!("Write-side aggregation failed for {}: {}", key, e);
                                continue;
                            }
                        }
                    } else {
                        state_value.clone()
                    };

                    // The next time index is used as this output state represents the value of a
                    // variable at the end of the current time step.
                    // This is the same as the start of the next timestep.
                    let result = match &final_value {
                        StateValue::Scalar(v) => data.set_scalar(key, self.time_index + 1, *v),
                        StateValue::FourBox(slice) => {
                            data.set_four_box(key, self.time_index + 1, slice)
                        }
                        StateValue::Hemispheric(slice) => {
                            data.set_hemispheric(key, self.time_index + 1, slice)
                        }
                    };
                    if let Err(e) = result {
                        println!("Failed to set output {}: {}", key, e);
                    }
                }
            }
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

    /// Steps the model forward one time step.
    ///
    /// This solves the current time step and then updates the index.
    pub fn step(&mut self) {
        assert!(self.time_index < self.time_axis.len() - 1);
        self.step_model();

        self.time_index += 1;
    }

    /// Steps the model until the end of the time axis.
    pub fn run(&mut self) {
        while self.time_index < self.time_axis.len() - 1 {
            self.step();
        }
    }

    /// Create a diagram that represents the component graph.
    ///
    /// Useful for debugging.
    pub fn as_dot(&self) -> Dot<'_, &CGraph> {
        Dot::with_attr_getters(
            &self.components,
            &[Config::NodeNoLabel, Config::EdgeNoLabel],
            &|_, er| format!("label = {:?}", er.weight().name),
            &|_, (_, component)| {
                // Escape quotes and backslashes for DOT format
                let debug_str = format!("{:?}", component);
                let escaped = debug_str.replace('\\', "\\\\").replace('"', "\\\"");
                format!("label = \"{}\"", escaped)
            },
        )
    }

    /// Returns true if the model has no more time steps to process.
    pub fn finished(&self) -> bool {
        self.time_index == self.time_axis.len() - 1
    }

    /// Returns a reference to the timeseries collection.
    pub fn timeseries(&self) -> &TimeseriesCollection {
        &self.collection
    }
}

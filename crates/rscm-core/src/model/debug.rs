//! Debug introspection for model execution order and variable flow.
//!
//! Provides structured output (JSON) and rich terminal display for understanding
//! how a model's components are wired together and in what order they execute.

use crate::component::{GridType, RequirementType};
use crate::state::VariableSource;
use petgraph::visit::Bfs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use super::runtime::Model;
use super::types::RequiredTransformation;

/// Source classification for a variable within a component's context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VarSource {
    Exogenous,
    Upstream,
    OwnState,
}

impl fmt::Display for VarSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VarSource::Exogenous => write!(f, "exo"),
            VarSource::Upstream => write!(f, "upstream"),
            VarSource::OwnState => write!(f, "own_state"),
        }
    }
}

impl From<VariableSource> for VarSource {
    fn from(vs: VariableSource) -> Self {
        match vs {
            VariableSource::Exogenous => VarSource::Exogenous,
            VariableSource::UpstreamOutput => VarSource::Upstream,
            VariableSource::OwnState => VarSource::OwnState,
        }
    }
}

/// A variable declaration in the debug output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarInfo {
    pub name: String,
    pub unit: String,
    #[serde(default, skip_serializing_if = "is_scalar")]
    pub grid: GridType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<VarSource>,
}

fn is_scalar(g: &GridType) -> bool {
    *g == GridType::Scalar
}

/// Debug information for a single component in the execution graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentDebugInfo {
    /// Position in the execution order (0-indexed).
    pub order: usize,
    /// Component type name.
    pub name: String,
    /// Input variables consumed by this component.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<VarInfo>,
    /// Output variables produced by this component.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<VarInfo>,
    /// State variables (read previous, write new).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub states: Vec<VarInfo>,
}

/// Grid transformation applied at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformInfo {
    pub variable: String,
    pub from: GridType,
    pub to: GridType,
    pub direction: String,
}

/// Unit conversion applied at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionInfo {
    pub variable: String,
    pub component: String,
    pub factor: f64,
}

/// Complete debug snapshot of a model's execution graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDebugInfo {
    /// Components in execution order.
    pub components: Vec<ComponentDebugInfo>,
    /// Grid transformations applied at runtime.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grid_transforms: Vec<TransformInfo>,
    /// Unit conversions applied at runtime.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unit_conversions: Vec<ConversionInfo>,
}

impl ModelDebugInfo {
    /// Serialise to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("ModelDebugInfo is always serialisable")
    }

    /// Serialise to compact JSON (no extra whitespace).
    pub fn to_json_compact(&self) -> String {
        serde_json::to_string(self).expect("ModelDebugInfo is always serialisable")
    }

    /// Render a rich, coloured terminal representation.
    ///
    /// When the `rich-debug` feature is enabled (default), uses `owo-colors`
    /// for styled output. Otherwise falls back to plain text via [`Display`].
    pub fn to_rich(&self) -> String {
        #[cfg(feature = "rich-debug")]
        {
            self.to_rich_coloured()
        }
        #[cfg(not(feature = "rich-debug"))]
        {
            format!("{}", self)
        }
    }

    /// Coloured output implementation using `owo-colors`.
    #[cfg(feature = "rich-debug")]
    fn to_rich_coloured(&self) -> String {
        use owo_colors::OwoColorize;

        let mut out = String::new();

        // Header
        out.push_str(&format!(
            "{}\n\n",
            "--- Model Execution Graph ---".cyan().bold()
        ));

        for c in &self.components {
            // Component header: "[order] ComponentName"
            out.push_str(&format!(
                "{} {}\n",
                format_args!("[{}]", c.order).yellow().bold(),
                c.name.white().bold()
            ));

            // Inputs
            for v in &c.inputs {
                let source_tag = v
                    .source
                    .as_ref()
                    .map(|s| format!(" {}", format_args!("({})", s).dimmed()))
                    .unwrap_or_default();
                let grid_tag = rich_grid_tag(v.grid);
                out.push_str(&format!(
                    "  {} {} {}{}{}\n",
                    "<-".green(),
                    v.name,
                    format_args!("[{}]", v.unit).dimmed(),
                    grid_tag,
                    source_tag
                ));
            }

            // States
            for v in &c.states {
                let grid_tag = rich_grid_tag(v.grid);
                out.push_str(&format!(
                    "  {} {} {}{}\n",
                    "<>".magenta(),
                    v.name,
                    format_args!("[{}]", v.unit).dimmed(),
                    grid_tag
                ));
            }

            // Outputs
            for v in &c.outputs {
                let grid_tag = rich_grid_tag(v.grid);
                out.push_str(&format!(
                    "  {} {} {}{}\n",
                    "->".blue(),
                    v.name,
                    format_args!("[{}]", v.unit).dimmed(),
                    grid_tag
                ));
            }

            out.push('\n');
        }

        // Grid transforms
        if !self.grid_transforms.is_empty() {
            out.push_str(&format!("{}\n", "Grid Transforms".cyan().bold()));
            for t in &self.grid_transforms {
                out.push_str(&format!(
                    "  {} {}\n",
                    t.variable,
                    format_args!("{} -> {} ({})", t.from, t.to, t.direction).dimmed()
                ));
            }
            out.push('\n');
        }

        // Unit conversions
        if !self.unit_conversions.is_empty() {
            out.push_str(&format!("{}\n", "Unit Conversions".cyan().bold()));
            for c in &self.unit_conversions {
                out.push_str(&format!(
                    "  {} {}\n",
                    c.variable,
                    format_args!("(in {}, x{})", c.component, c.factor).dimmed()
                ));
            }
            out.push('\n');
        }

        out
    }
}

/// Format a grid tag for rich output. Returns empty string for Scalar.
#[cfg(feature = "rich-debug")]
fn rich_grid_tag(grid: GridType) -> String {
    use owo_colors::OwoColorize;

    match grid {
        GridType::Scalar => String::new(),
        _ => format!(" {}", format_args!("[{}]", grid).yellow()),
    }
}

impl fmt::Display for ModelDebugInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Plain-text version (no ANSI codes)
        writeln!(f, "--- Model Execution Graph ---")?;
        writeln!(f)?;

        for c in &self.components {
            writeln!(f, "[{}] {}", c.order, c.name)?;

            for v in &c.inputs {
                let source_tag = v
                    .source
                    .as_ref()
                    .map(|s| format!(" ({})", s))
                    .unwrap_or_default();
                let grid_tag = if v.grid != GridType::Scalar {
                    format!(" [{}]", v.grid)
                } else {
                    String::new()
                };
                writeln!(f, "  <- {} [{}]{}{}", v.name, v.unit, grid_tag, source_tag)?;
            }

            for v in &c.states {
                let grid_tag = if v.grid != GridType::Scalar {
                    format!(" [{}]", v.grid)
                } else {
                    String::new()
                };
                writeln!(f, "  <> {} [{}]{}", v.name, v.unit, grid_tag)?;
            }

            for v in &c.outputs {
                let grid_tag = if v.grid != GridType::Scalar {
                    format!(" [{}]", v.grid)
                } else {
                    String::new()
                };
                writeln!(f, "  -> {} [{}]{}", v.name, v.unit, grid_tag)?;
            }

            writeln!(f)?;
        }

        if !self.grid_transforms.is_empty() {
            writeln!(f, "Grid Transforms")?;
            for t in &self.grid_transforms {
                writeln!(
                    f,
                    "  {} {} -> {} ({})",
                    t.variable, t.from, t.to, t.direction
                )?;
            }
            writeln!(f)?;
        }

        if !self.unit_conversions.is_empty() {
            writeln!(f, "Unit Conversions")?;
            for c in &self.unit_conversions {
                writeln!(f, "  {} (in {}, x{})", c.variable, c.component, c.factor)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

impl Model {
    /// Generate debug information about the model's execution graph.
    ///
    /// Walks the component graph in BFS order (the actual execution order)
    /// and collects inputs, outputs, states, grid types, and variable sources
    /// for each component.
    pub fn debug_info(&self) -> ModelDebugInfo {
        let mut components = Vec::new();
        let mut order = 0;

        let mut bfs = Bfs::new(&self.components, self.initial_node);
        while let Some(nx) = bfs.next(&self.components) {
            let component = &self.components[nx];

            // Skip the NullComponent (graph root)
            let debug_str = format!("{:?}", component);
            let component_name = debug_str
                .split(['{', ' ', '('])
                .next()
                .unwrap_or("Unknown")
                .to_string();

            if component_name == "NullComponent" {
                continue;
            }

            let defs = component.definitions();
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            let mut states = Vec::new();

            for def in &defs {
                match def.requirement_type {
                    RequirementType::Input => {
                        let source = self
                            .variable_sources()
                            .get(&(def.name.clone(), component_name.clone()))
                            .copied()
                            .map(VarSource::from);

                        inputs.push(VarInfo {
                            name: def.name.clone(),
                            unit: def.unit.clone(),
                            grid: def.grid_type,
                            source,
                        });
                    }
                    RequirementType::Output => {
                        outputs.push(VarInfo {
                            name: def.name.clone(),
                            unit: def.unit.clone(),
                            grid: def.grid_type,
                            source: None,
                        });
                    }
                    RequirementType::State => {
                        states.push(VarInfo {
                            name: def.name.clone(),
                            unit: def.unit.clone(),
                            grid: def.grid_type,
                            source: None,
                        });
                    }
                    RequirementType::EmptyLink => {}
                }
            }

            components.push(ComponentDebugInfo {
                order,
                name: component_name,
                inputs,
                outputs,
                states,
            });
            order += 1;
        }

        // Collect grid transforms
        let grid_transforms = collect_transforms(self.read_transforms(), self.write_transforms());

        // Collect unit conversions
        let unit_conversions: Vec<ConversionInfo> = self
            .unit_conversions()
            .iter()
            .map(|((var, comp), factor)| ConversionInfo {
                variable: var.clone(),
                component: comp.clone(),
                factor: *factor,
            })
            .collect();

        ModelDebugInfo {
            components,
            grid_transforms,
            unit_conversions,
        }
    }
}

fn collect_transforms(
    read: &HashMap<String, RequiredTransformation>,
    write: &HashMap<String, RequiredTransformation>,
) -> Vec<TransformInfo> {
    let mut transforms = Vec::new();

    for t in read.values() {
        transforms.push(TransformInfo {
            variable: t.variable.clone(),
            from: t.source_grid,
            to: t.target_grid,
            direction: "read".to_string(),
        });
    }

    for t in write.values() {
        transforms.push(TransformInfo {
            variable: t.variable.clone(),
            from: t.source_grid,
            to: t.target_grid,
            direction: "write".to_string(),
        });
    }

    transforms.sort_by(|a, b| a.variable.cmp(&b.variable));
    transforms
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Component, InputState, OutputState, RequirementDefinition};
    use crate::errors::RSCMResult;
    use crate::model::ModelBuilder;
    use crate::timeseries::{Time, TimeAxis, Timeseries};
    use ndarray::array;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    #[derive(Debug, Serialize, Deserialize)]
    struct EmissionsComponent {
        factor: f64,
    }

    #[typetag::serde]
    impl Component for EmissionsComponent {
        fn definitions(&self) -> Vec<RequirementDefinition> {
            vec![
                RequirementDefinition::scalar_input("Emissions|CO2", "GtCO2 / yr"),
                RequirementDefinition::scalar_output("Concentration|CO2", "ppm"),
            ]
        }

        fn solve(
            &self,
            _t_current: Time,
            _t_next: Time,
            _input_state: &InputState,
        ) -> RSCMResult<OutputState> {
            Ok(OutputState::new())
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct ForcingComponent {
        sensitivity: f64,
    }

    #[typetag::serde]
    impl Component for ForcingComponent {
        fn definitions(&self) -> Vec<RequirementDefinition> {
            vec![
                RequirementDefinition::scalar_input("Concentration|CO2", "ppm"),
                RequirementDefinition::scalar_output("ERF|CO2", "W / m^2"),
            ]
        }

        fn solve(
            &self,
            _t_current: Time,
            _t_next: Time,
            _input_state: &InputState,
        ) -> RSCMResult<OutputState> {
            Ok(OutputState::new())
        }
    }

    fn build_test_model() -> Model {
        let time_axis = TimeAxis::from_values(array![2020.0, 2021.0, 2022.0]);
        let emissions_ts =
            Timeseries::from_values(array![10.0, 11.0, 12.0], array![2020.0, 2021.0, 2022.0]);

        let mut builder = ModelBuilder::new();
        builder.with_time_axis(time_axis);
        builder.with_component(Arc::new(EmissionsComponent { factor: 0.47 }));
        builder.with_component(Arc::new(ForcingComponent { sensitivity: 5.35 }));
        builder.with_exogenous_variable("Emissions|CO2", emissions_ts);
        builder.build().unwrap()
    }

    #[test]
    fn test_debug_info_execution_order() {
        let model = build_test_model();
        let info = model.debug_info();

        assert_eq!(info.components.len(), 2);
        assert_eq!(info.components[0].name, "EmissionsComponent");
        assert_eq!(info.components[0].order, 0);
        assert_eq!(info.components[1].name, "ForcingComponent");
        assert_eq!(info.components[1].order, 1);
    }

    #[test]
    fn test_debug_info_inputs_outputs() {
        let model = build_test_model();
        let info = model.debug_info();

        // EmissionsComponent: 1 input (Emissions|CO2), 1 output (Concentration|CO2)
        let emissions = &info.components[0];
        assert_eq!(emissions.inputs.len(), 1);
        assert_eq!(emissions.inputs[0].name, "Emissions|CO2");
        assert_eq!(emissions.outputs.len(), 1);
        assert_eq!(emissions.outputs[0].name, "Concentration|CO2");

        // ForcingComponent: 1 input (Concentration|CO2), 1 output (ERF|CO2)
        let forcing = &info.components[1];
        assert_eq!(forcing.inputs.len(), 1);
        assert_eq!(forcing.inputs[0].name, "Concentration|CO2");
        assert_eq!(forcing.outputs.len(), 1);
        assert_eq!(forcing.outputs[0].name, "ERF|CO2");
    }

    #[test]
    fn test_debug_info_variable_sources() {
        let model = build_test_model();
        let info = model.debug_info();

        // Emissions|CO2 is exogenous to EmissionsComponent
        let emissions_input = &info.components[0].inputs[0];
        assert_eq!(emissions_input.source, Some(VarSource::Exogenous));

        // Concentration|CO2 is upstream to ForcingComponent
        let forcing_input = &info.components[1].inputs[0];
        assert_eq!(forcing_input.source, Some(VarSource::Upstream));
    }

    #[test]
    fn test_debug_info_json_roundtrip() {
        let model = build_test_model();
        let info = model.debug_info();
        let json = info.to_json();

        let parsed: ModelDebugInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.components.len(), info.components.len());
        assert_eq!(parsed.components[0].name, info.components[0].name);
    }

    #[test]
    fn test_debug_info_display_plain() {
        let model = build_test_model();
        let info = model.debug_info();
        let plain = format!("{}", info);

        assert!(plain.contains("[0] EmissionsComponent"));
        assert!(plain.contains("[1] ForcingComponent"));
        assert!(plain.contains("<- Emissions|CO2"));
        assert!(plain.contains("-> Concentration|CO2"));
        assert!(plain.contains("-> ERF|CO2"));
    }

    #[test]
    fn test_debug_info_rich_output() {
        let model = build_test_model();
        let info = model.debug_info();
        let rich = info.to_rich();

        // With rich-debug feature enabled, should contain ANSI escape codes
        #[cfg(feature = "rich-debug")]
        assert!(rich.contains("\x1b["));

        // Should always contain component names regardless of feature
        assert!(rich.contains("EmissionsComponent"));
        assert!(rich.contains("ForcingComponent"));
    }

    #[test]
    fn test_debug_info_compact_json() {
        let model = build_test_model();
        let info = model.debug_info();
        let compact = info.to_json_compact();

        // Compact JSON should not contain newlines
        assert!(!compact.contains('\n'));
        // But should still be parseable
        let parsed: ModelDebugInfo = serde_json::from_str(&compact).unwrap();
        assert_eq!(parsed.components.len(), 2);
    }
}

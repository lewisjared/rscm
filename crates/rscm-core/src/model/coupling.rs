//! What the builder decided about a model's coupling.
//!
//! [`Model::coupling`] reports the order components run in, every edge between them
//! with whether it lags, each variable's source and timeseries type, and every conversion.

use crate::component::{DeclaredVariable, GridType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use super::plan::Conversion;
use super::runtime::Model;

/// A component and the variables it declares.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentCoupling {
    /// Position in the execution order (0-indexed).
    pub order: usize,
    /// The component's label.
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<DeclaredVariable>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<DeclaredVariable>,
    /// State variables (read previous, write new).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub states: Vec<DeclaredVariable>,
}

/// A variable passed from the component that produces it to one that reads it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouplingEdge {
    pub producer: String,
    pub consumer: String,
    pub variable: String,
    /// Whether the consumer reads the previous step's value.
    pub lagged: bool,
}

/// A model's coupling: execution order, edges, each input's variable source and every conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coupling {
    /// Components in the order the runtime steps them.
    pub components: Vec<ComponentCoupling>,
    /// Edges ordered by consumer, then variable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<CouplingEdge>,
    /// Unit and grid conversions of each read and write, ordered by variable, component and direction.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conversions: Vec<Conversion>,
}

impl Coupling {
    /// Serialise to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Coupling is always serialisable")
    }

    /// Serialise to compact JSON (no extra whitespace).
    pub fn to_json_compact(&self) -> String {
        serde_json::to_string(self).expect("Coupling is always serialisable")
    }

    /// Render for a terminal, coloured when the `rich-debug` feature is enabled.
    pub fn to_rich(&self) -> String {
        self.render(cfg!(feature = "rich-debug"))
    }

    fn render(&self, colour: bool) -> String {
        let paint = |text: &str, style: Style| style.paint(text, colour);
        let tag = |v: &DeclaredVariable| {
            let grid = if v.grid == GridType::Scalar {
                String::new()
            } else {
                format!(" {}", paint(&format!("[{}]", v.grid), Style::Grid))
            };
            format!(
                "{} {}{} {}",
                v.name,
                paint(&format!("[{}]", v.unit), Style::Dim),
                grid,
                paint(&v.timeseries_type.to_string(), Style::Dim),
            )
        };
        let producers: HashMap<(&str, &str), &str> = self
            .edges
            .iter()
            .map(|e| {
                (
                    (e.consumer.as_str(), e.variable.as_str()),
                    e.producer.as_str(),
                )
            })
            .collect();

        let mut out = format!("{}\n\n", paint("--- Model Coupling ---", Style::Header));
        for c in &self.components {
            out.push_str(&format!(
                "{} {}\n",
                paint(&format!("[{}]", c.order), Style::Order),
                paint(&c.name, Style::Name)
            ));
            for v in &c.inputs {
                let source = match (v.source, producers.get(&(c.name.as_str(), v.name.as_str()))) {
                    (Some(s), Some(p)) => format!(" ({s} from {p})"),
                    (Some(s), None) => format!(" ({s})"),
                    (None, _) => String::new(),
                };
                out.push_str(&format!(
                    "  {} {}{}\n",
                    paint("<-", Style::In),
                    tag(v),
                    paint(&source, Style::Dim)
                ));
            }
            for v in &c.states {
                out.push_str(&format!("  {} {}\n", paint("<>", Style::State), tag(v)));
            }
            for v in &c.outputs {
                out.push_str(&format!("  {} {}\n", paint("->", Style::Out), tag(v)));
            }
            out.push('\n');
        }

        let mut section = |title: &str, lines: Vec<String>| {
            if !lines.is_empty() {
                out.push_str(&format!("{}\n", paint(title, Style::Header)));
                for line in lines {
                    out.push_str(&format!("  {line}\n"));
                }
                out.push('\n');
            }
        };
        section(
            "Lagged Edges",
            self.edges
                .iter()
                .filter(|e| e.lagged)
                .map(|e| format!("{} -> {}: {}", e.producer, e.consumer, e.variable))
                .collect(),
        );
        section(
            "Grid Transforms",
            self.conversions
                .iter()
                .filter_map(|c| {
                    let grid = c.grid?;
                    let detail = format!(
                        "{} -> {} ({} by {})",
                        grid.from, grid.to, c.direction, c.component
                    );
                    Some(format!("{} {}", c.variable, paint(&detail, Style::Dim)))
                })
                .collect(),
        );
        section(
            "Unit Conversions",
            self.conversions
                .iter()
                .filter_map(|c| {
                    let unit = c.unit.as_ref()?;
                    let detail = format!(
                        "{} -> {} ({} by {}, x{})",
                        unit.from, unit.to, c.direction, c.component, unit.factor
                    );
                    Some(format!("{} {}", c.variable, paint(&detail, Style::Dim)))
                })
                .collect(),
        );
        out
    }
}

impl fmt::Display for Coupling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render(false))
    }
}

#[derive(Clone, Copy)]
enum Style {
    Header,
    Order,
    Name,
    In,
    State,
    Out,
    Dim,
    Grid,
}

impl Style {
    #[cfg(feature = "rich-debug")]
    fn paint(self, text: &str, colour: bool) -> String {
        use owo_colors::OwoColorize;

        if !colour || text.is_empty() {
            return text.to_string();
        }
        match self {
            Style::Header => text.cyan().bold().to_string(),
            Style::Order => text.yellow().bold().to_string(),
            Style::Name => text.white().bold().to_string(),
            Style::In => text.green().to_string(),
            Style::State => text.magenta().to_string(),
            Style::Out => text.blue().to_string(),
            Style::Dim => text.dimmed().to_string(),
            Style::Grid => text.yellow().to_string(),
        }
    }

    #[cfg(not(feature = "rich-debug"))]
    fn paint(self, text: &str, _colour: bool) -> String {
        text.to_string()
    }
}

impl Model {
    /// Report the model's coupling: the order the runtime steps components in,
    /// every edge with whether it lags, each variable's source and timeseries type,
    /// and the conversions each component applies.
    pub fn coupling(&self) -> Coupling {
        self.plan().report()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{
        Component, InputState, OutputState, TimeseriesType, VariableDefinition,
    };
    use crate::errors::RSCMResult;
    use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
    use crate::model::ModelBuilder;
    use crate::spatial::ScalarGrid;
    use crate::state::VariableSource;
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
        fn definitions(&self) -> Vec<VariableDefinition> {
            vec![
                VariableDefinition::scalar_input("Emissions|CO2", "GtCO2/yr"),
                VariableDefinition::scalar_output("Atmospheric Concentrations|CO2", "ppm"),
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
        fn definitions(&self) -> Vec<VariableDefinition> {
            vec![
                VariableDefinition::scalar_input("Atmospheric Concentrations|CO2", "ppm"),
                VariableDefinition::scalar_output("Effective Radiative Forcing|CO2", "W/m^2"),
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
        let emissions_ts = Timeseries::new(
            array![[10.0], [11.0], [12.0]],
            Arc::new(time_axis.clone()),
            ScalarGrid,
            "GtCO2/yr".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let mut builder = ModelBuilder::new();
        builder.with_time_axis(time_axis);
        builder.with_component(Arc::new(EmissionsComponent { factor: 0.47 }));
        builder.with_component(Arc::new(ForcingComponent { sensitivity: 5.35 }));
        builder.with_prescribed_variable("Emissions|CO2", emissions_ts);
        builder.build().unwrap()
    }

    #[test]
    fn test_coupling_execution_order() {
        let coupling = build_test_model().coupling();

        assert_eq!(coupling.components.len(), 2);
        assert_eq!(coupling.components[0].name, "EmissionsComponent");
        assert_eq!(coupling.components[0].order, 0);
        assert_eq!(coupling.components[1].name, "ForcingComponent");
        assert_eq!(coupling.components[1].order, 1);
    }

    #[test]
    fn test_coupling_inputs_outputs() {
        let coupling = build_test_model().coupling();

        let emissions = &coupling.components[0];
        assert_eq!(emissions.inputs.len(), 1);
        assert_eq!(emissions.inputs[0].name, "Emissions|CO2");
        assert_eq!(emissions.outputs.len(), 1);
        assert_eq!(emissions.outputs[0].name, "Atmospheric Concentrations|CO2");

        let forcing = &coupling.components[1];
        assert_eq!(forcing.inputs.len(), 1);
        assert_eq!(forcing.inputs[0].name, "Atmospheric Concentrations|CO2");
        assert_eq!(forcing.outputs.len(), 1);
        assert_eq!(forcing.outputs[0].name, "Effective Radiative Forcing|CO2");
    }

    #[test]
    fn test_coupling_sources_and_types() {
        let coupling = build_test_model().coupling();

        let emissions = &coupling.components[0].inputs[0];
        assert_eq!(emissions.source, Some(VariableSource::Prescribed));
        assert_eq!(emissions.timeseries_type, TimeseriesType::Average);

        let concentration = &coupling.components[1].inputs[0];
        assert_eq!(concentration.source, Some(VariableSource::Upstream));
        assert_eq!(concentration.timeseries_type, TimeseriesType::Point);
    }

    #[test]
    fn test_coupling_edges() {
        let coupling = build_test_model().coupling();

        assert_eq!(
            coupling.edges,
            vec![CouplingEdge {
                producer: "EmissionsComponent".into(),
                consumer: "ForcingComponent".into(),
                variable: "Atmospheric Concentrations|CO2".into(),
                lagged: false,
            }]
        );
    }

    #[test]
    fn test_coupling_json_roundtrip() {
        let coupling = build_test_model().coupling();
        let parsed: Coupling = serde_json::from_str(&coupling.to_json()).unwrap();

        assert_eq!(parsed.components.len(), coupling.components.len());
        assert_eq!(parsed.components[0].name, coupling.components[0].name);
        assert_eq!(parsed.edges, coupling.edges);
    }

    #[test]
    fn test_coupling_display_plain() {
        let plain = build_test_model().coupling().to_string();

        assert!(plain.contains("[0] EmissionsComponent"));
        assert!(plain.contains("[1] ForcingComponent"));
        assert!(plain.contains("<- Emissions|CO2 [GtCO2/yr] average (prescribed)"));
        assert!(plain.contains(
            "<- Atmospheric Concentrations|CO2 [ppm] point (upstream from EmissionsComponent)"
        ));
        assert!(plain.contains("-> Effective Radiative Forcing|CO2 [W/m^2] point"));
        assert!(!plain.contains("\x1b["));
    }

    #[test]
    fn test_coupling_rich_output() {
        let rich = build_test_model().coupling().to_rich();

        #[cfg(feature = "rich-debug")]
        assert!(rich.contains("\x1b["));
        assert!(rich.contains("EmissionsComponent"));
        assert!(rich.contains("ForcingComponent"));
    }

    #[test]
    fn test_coupling_compact_json() {
        let compact = build_test_model().coupling().to_json_compact();

        assert!(!compact.contains('\n'));
        let parsed: Coupling = serde_json::from_str(&compact).unwrap();
        assert_eq!(parsed.components.len(), 2);
    }
}

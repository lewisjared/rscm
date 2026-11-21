//! A model's spec and state as one serialisable [`Record`], and restoring or branching from one.

use super::builder::ModelBuilder;
use super::runtime::Model;
use super::types::C;
use crate::component::{ComponentState, TimeseriesType};
use crate::errors::{RSCMError, RSCMResult};
use crate::timeseries_collection::TimeseriesCollection;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The layout version of a [`Record`]. A record of another format cannot be read.
pub const RECORD_FORMAT: u32 = 1;

/// A model's spec and state at one step: a checkpoint mid-run, or the output of a finished run.
///
/// Restoring rebuilds the model from `builder`, then loads the saved values and states into it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// The layout version, [`RECORD_FORMAT`] when written.
    pub format: u32,
    /// The rscm version that wrote the record.
    pub rscm: String,
    /// The builder the model was built from.
    pub builder: Arc<ModelBuilder>,
    /// The index of the next step to solve.
    pub time_index: usize,
    /// Every stored timeseries.
    pub values: TimeseriesCollection,
    /// Whether each stored variable is a point or an average.
    pub timeseries_types: BTreeMap<String, TimeseriesType>,
    /// Each component's internal state, keyed by label.
    pub states: BTreeMap<String, serde_json::Value>,
}

/// What a branch changes from its record.
#[derive(Debug, Clone, Default)]
pub struct Branch {
    /// Prescribed data replacing the record's from the step after the branch point.
    pub prescribed: TimeseriesCollection,
    /// Components replacing those of the same label, each of the same type.
    pub parameters: BTreeMap<String, C>,
}

#[derive(Deserialize)]
struct Header {
    format: u32,
}

fn check_format(format: u32) -> RSCMResult<()> {
    if format == RECORD_FORMAT {
        Ok(())
    } else {
        Err(RSCMError::Error(format!(
            "The record has format {format}, but this rscm reads format {RECORD_FORMAT}"
        )))
    }
}

impl Record {
    /// Serialise the record as JSON.
    pub fn to_json(&self) -> RSCMResult<String> {
        serde_json::to_string(self)
            .map_err(|e| RSCMError::Error(format!("Could not serialise the record: {e}")))
    }

    /// Read a record from JSON, failing if its format is not [`RECORD_FORMAT`].
    pub fn from_json(json: &str) -> RSCMResult<Self> {
        let read_error = |e| RSCMError::Error(format!("Could not read the record: {e}"));
        let header: Header = serde_json::from_str(json).map_err(read_error)?;
        check_format(header.format)?;
        serde_json::from_str(json).map_err(read_error)
    }
}

/// How the saved values are loaded into the rebuilt model.
#[derive(Clone, Copy)]
enum Load {
    /// Every step is the record's.
    All,
    /// Steps up to the branch point are the record's, later ones the rebuilt model's.
    History,
}

impl Model {
    /// The record of the model at its current step, to resume or branch from later.
    pub fn checkpoint(&self) -> RSCMResult<Record> {
        let states = self
            .labels
            .iter()
            .map(|(node, label)| {
                let state = serde_json::to_value(&self.component_states[node]).map_err(|e| {
                    RSCMError::Error(format!("Could not serialise the state of {label}: {e}"))
                })?;
                Ok((label.clone(), state))
            })
            .collect::<RSCMResult<_>>()?;
        Ok(Record {
            format: RECORD_FORMAT,
            rscm: env!("CARGO_PKG_VERSION").to_string(),
            builder: self.builder.clone(),
            time_index: self.time_index,
            values: self.collection.clone(),
            timeseries_types: self.timeseries_types.clone(),
            states,
        })
    }

    /// The record of a run, the same as [`Self::checkpoint`], for reading results after [`Self::run`].
    pub fn output(&self) -> RSCMResult<Record> {
        self.checkpoint()
    }

    /// Rebuild a model from a record and load its values and states, logging any warnings.
    pub fn restore(record: &Record) -> RSCMResult<Model> {
        let (model, warnings) = Self::restore_with_warnings(record)?;
        for message in warnings {
            tracing::warn!("{message}");
        }
        Ok(model)
    }

    /// Like [`Self::restore`], but returns the warnings, such as a record from another rscm version.
    ///
    /// Fails if the record's format differs, or if the rebuilt model stores a different set of variables,
    /// or any in another unit, grid or timeseries type.
    pub fn restore_with_warnings(record: &Record) -> RSCMResult<(Model, Vec<String>)> {
        Self::rebuild(record, &record.builder, Load::All)
    }

    /// Rebuild a model from a record with new future prescribed data or component parameters,
    /// logging any warnings.
    pub fn branch(record: &Record, branch: Branch) -> RSCMResult<Model> {
        let (model, warnings) = Self::branch_with_warnings(record, branch)?;
        for message in warnings {
            tracing::warn!("{message}");
        }
        Ok(model)
    }

    /// Like [`Self::branch`], but returns the warnings.
    ///
    /// Values up to the record's step stay as saved, so new prescribed data takes effect after it.
    /// A point variable keeps its value at the record's step and an average its values before it.
    /// Fails for prescribed data the model does not take, or as [`Self::restore_with_warnings`] does,
    /// so new parameters cannot change what is stored.
    pub fn branch_with_warnings(
        record: &Record,
        branch: Branch,
    ) -> RSCMResult<(Model, Vec<String>)> {
        let mut builder = ModelBuilder::clone(&record.builder);
        let inputs = builder.prescribed_inputs();
        for item in branch.prescribed.iter() {
            let name = &item.name;
            let prescribed = builder.prescribed_variables.get_data(name).is_some()
                || inputs.iter().any(|input| &input.definition.name == name);
            if !prescribed {
                return Err(RSCMError::Error(format!(
                    "Cannot branch with prescribed data for '{name}', which the model does not take as prescribed"
                )));
            }
        }
        builder.with_prescribed_collection(branch.prescribed);
        for (label, component) in branch.parameters {
            builder.replace_component(&label, component)?;
        }
        Self::rebuild(record, &builder, Load::History)
    }

    fn rebuild(
        record: &Record,
        builder: &ModelBuilder,
        load: Load,
    ) -> RSCMResult<(Model, Vec<String>)> {
        check_format(record.format)?;
        let mut warnings = Vec::new();
        let version = env!("CARGO_PKG_VERSION");
        if record.rscm != version {
            warnings.push(format!(
                "The record was written by rscm {}, this is rscm {version}, so results may differ",
                record.rscm
            ));
        }
        let (mut model, build_warnings) = builder.build_with_warnings()?;
        warnings.extend(build_warnings);
        model.check_storage(record)?;
        if record.time_index >= model.timeseries_len() {
            return Err(RSCMError::Error(format!(
                "The record is at step {}, past the end of its time axis",
                record.time_index
            )));
        }
        model.load_values(record, load);
        model.load_states(record)?;
        model.time_index = record.time_index;
        Ok((model, warnings))
    }

    /// Fail unless the model stores the record's variables in the same unit, grid and timeseries type.
    fn check_storage(&self, record: &Record) -> RSCMResult<()> {
        let mismatch = |name: &str, what: String| {
            Err(RSCMError::Error(format!(
                "Cannot restore the record: '{name}' {what}"
            )))
        };
        for item in record.values.iter() {
            if self.collection.get_by_name(&item.name).is_none() {
                return mismatch(
                    &item.name,
                    "is in the record but not the rebuilt model".into(),
                );
            }
        }
        for item in self.collection.iter() {
            let name = &item.name;
            let Some(saved) = record.values.get_data(name) else {
                return mismatch(name, "is in the rebuilt model but not the record".into());
            };
            let rebuilt = &item.data;
            if saved.unit() != rebuilt.unit() {
                return mismatch(
                    name,
                    format!(
                        "is stored in '{}' by the rebuilt model but '{}' in the record",
                        rebuilt.unit(),
                        saved.unit()
                    ),
                );
            }
            if saved.grid() != rebuilt.grid() {
                return mismatch(
                    name,
                    format!(
                        "is stored on the {} grid by the rebuilt model but the {} grid in the record",
                        rebuilt.grid(),
                        saved.grid()
                    ),
                );
            }
            if saved.len() != rebuilt.len() {
                return mismatch(
                    name,
                    format!(
                        "has {} steps in the rebuilt model but {} in the record",
                        rebuilt.len(),
                        saved.len()
                    ),
                );
            }
            let rebuilt_type = self.timeseries_types.get(name);
            let saved_type = record.timeseries_types.get(name);
            if rebuilt_type != saved_type {
                let show =
                    |t: Option<&TimeseriesType>| t.map_or("untyped".into(), |t| t.to_string());
                return mismatch(
                    name,
                    format!(
                        "is a {} in the rebuilt model but a {} in the record",
                        show(rebuilt_type),
                        show(saved_type)
                    ),
                );
            }
        }
        Ok(())
    }

    fn load_values(&mut self, record: &Record, load: Load) {
        for index in 0..self.collection.items().len() {
            let name = &self.collection.items()[index].name;
            let saved = record
                .values
                .get_data(name)
                .expect("checked by check_storage");
            let end = match (load, self.timeseries_types.get(name)) {
                (Load::All, _) => saved.len(),
                (Load::History, Some(TimeseriesType::Average)) => record.time_index,
                (Load::History, _) => record.time_index + 1,
            };
            self.collection.data_mut(index).copy_steps_from(saved, end);
        }
    }

    fn load_states(&mut self, record: &Record) -> RSCMResult<()> {
        if let Some(label) = record
            .states
            .keys()
            .find(|label| !self.labels.values().any(|l| l == *label))
        {
            return Err(RSCMError::Error(format!(
                "Cannot restore the record: it has a state for '{label}', which the rebuilt model lacks"
            )));
        }
        for (node, label) in &self.labels {
            let Some(saved) = record.states.get(label) else {
                return Err(RSCMError::Error(format!(
                    "Cannot restore the record: it has no state for '{label}'"
                )));
            };
            let state: Box<dyn ComponentState> =
                serde_json::from_value(saved.clone()).map_err(|e| {
                    RSCMError::Error(format!("Could not read the state of {label}: {e}"))
                })?;
            self.component_states.insert(*node, state);
        }
        Ok(())
    }
}

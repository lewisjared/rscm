//! State management for model components.
//!
//! This module provides types for managing input and output state in model components,
//! including:
//!
//! - [`StateValue`]: Enum for scalar, four-box, or hemispheric values
//! - [`InputState`]: Input state passed to components during solve
//! - [`OutputState`]: Output state returned from components
//! - Window types for efficient timeseries access
//! - Slice types for typed regional values

mod aggregating;
mod slices;
mod windows;

// Re-export slice types
pub use slices::{FourBoxSlice, HemisphericSlice};

// Re-export window types
pub(crate) use windows::ReadIndex;
pub use windows::{GridTimeseriesWindow, TimeseriesWindow};

// Re-export aggregating window types
pub use aggregating::{
    AggregatingFourBoxToHemisphericWindow, AggregatingFourBoxWindow, AggregatingHemisphericWindow,
    HemisphericWindow, ReadTransformInfo, ScalarWindow,
};

use crate::component::{GridType, TimeseriesType};
use crate::spatial::FourBoxGrid;
use crate::timeseries::{FloatValue, Time, TimeAxis};
use crate::timeseries_collection::TimeseriesItem;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::HashMap;

// =============================================================================
// State Value Types
// =============================================================================

/// Represents a value that can be either scalar or spatially-resolved
///
/// `StateValue` is the enum used for both input state retrieval and output state
/// in components. It provides type-safe handling of scalar and grid-based values.
///
/// # Examples
///
/// ```rust
/// use rscm_core::state::{StateValue, FourBoxSlice, HemisphericSlice};
///
/// // Scalar value
/// let scalar = StateValue::Scalar(288.0);
/// assert_eq!(scalar.to_scalar(), 288.0);
///
/// // Four-box value
/// let four_box = StateValue::FourBox(FourBoxSlice::from_array([15.0, 14.0, 10.0, 9.0]));
/// assert_eq!(four_box.to_scalar(), 12.0); // Mean of all regions
///
/// // Hemispheric value
/// let hemispheric = StateValue::Hemispheric(HemisphericSlice::from_array([15.0, 10.0]));
/// assert_eq!(hemispheric.to_scalar(), 12.5); // Mean of both hemispheres
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateValue {
    /// A single scalar value (global average or non-spatial variable)
    Scalar(FloatValue),
    /// Four-box regional values (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
    FourBox(FourBoxSlice),
    /// Hemispheric values (Northern, Southern)
    Hemispheric(HemisphericSlice),
}

impl StateValue {
    /// Convert to a scalar value, aggregating if necessary
    ///
    /// For Scalar variants, returns the value directly.
    /// For FourBox variants, computes the mean of all 4 regional values.
    /// For Hemispheric variants, computes the mean of both hemispheres.
    ///
    /// Note: This simple averaging may not be physically appropriate for all variables.
    /// Use grid-aware aggregation methods when the grid weights are known.
    pub fn to_scalar(&self) -> FloatValue {
        match self {
            StateValue::Scalar(v) => *v,
            StateValue::FourBox(slice) => {
                let values = slice.as_array();
                values.iter().sum::<FloatValue>() / 4.0
            }
            StateValue::Hemispheric(slice) => {
                let values = slice.as_array();
                values.iter().sum::<FloatValue>() / 2.0
            }
        }
    }

    /// The grid this value is laid out on.
    pub fn grid(&self) -> GridType {
        match self {
            StateValue::Scalar(_) => GridType::Scalar,
            StateValue::FourBox(_) => GridType::FourBox,
            StateValue::Hemispheric(_) => GridType::Hemispheric,
        }
    }

    /// Check if this is a scalar value
    pub fn is_scalar(&self) -> bool {
        matches!(self, StateValue::Scalar(_))
    }

    /// Check if this is a four-box grid value
    pub fn is_four_box(&self) -> bool {
        matches!(self, StateValue::FourBox(_))
    }

    /// Check if this is a Hemispheric grid value
    pub fn is_hemispheric(&self) -> bool {
        matches!(self, StateValue::Hemispheric(_))
    }

    /// Get the scalar value if this is a Scalar variant
    pub fn as_scalar(&self) -> Option<FloatValue> {
        match self {
            StateValue::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the FourBoxSlice if this is a FourBox variant
    pub fn as_four_box(&self) -> Option<&FourBoxSlice> {
        match self {
            StateValue::FourBox(slice) => Some(slice),
            _ => None,
        }
    }

    /// Get the HemisphericSlice if this is a Hemispheric variant
    pub fn as_hemispheric(&self) -> Option<&HemisphericSlice> {
        match self {
            StateValue::Hemispheric(slice) => Some(slice),
            _ => None,
        }
    }
}

impl From<FloatValue> for StateValue {
    fn from(value: FloatValue) -> Self {
        StateValue::Scalar(value)
    }
}

impl From<FourBoxSlice> for StateValue {
    fn from(slice: FourBoxSlice) -> Self {
        StateValue::FourBox(slice)
    }
}

impl From<HemisphericSlice> for StateValue {
    fn from(slice: HemisphericSlice) -> Self {
        StateValue::Hemispheric(slice)
    }
}

/// Where one component's input is read from, as the builder wired it.
///
/// With the variable's point or average type, this decides the step index `get()` reads.
/// Prescribed and upstream inputs read the same index, so the distinction only describes the coupling.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariableSource {
    /// Supplied to the model rather than produced by a component.
    /// `get()` reads a point at index N+1 and an average at index N.
    #[default]
    Prescribed,

    /// Produced earlier in the same step by an upstream component.
    /// `get()` reads a point at index N+1 and an average at index N.
    Upstream,

    /// Read from the previous step because the model lags this edge.
    /// `get()` reads index N, and `interval()` gives the previous step's (N-1, N).
    Lagged,

    /// The component's own state from the previous step.
    /// `get()` reads a point at index N and an average at index N-1.
    OwnState,
}

impl std::fmt::Display for VariableSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VariableSource::Prescribed => "prescribed",
            VariableSource::Upstream => "upstream",
            VariableSource::Lagged => "lagged",
            VariableSource::OwnState => "own_state",
        })
    }
}

/// Transform context for automatic grid aggregation.
///
/// This holds the information needed to transform grid data during read operations.
/// It is passed to InputState when the model has configured grid transformations.
#[derive(Debug, Clone, Default)]
pub struct TransformContext {
    /// Map from variable name to the read transformation info
    pub read_transforms: HashMap<String, ReadTransformInfo>,
}

/// Input state for a component
///
/// A state is a collection of values
/// that can be used to represent the state of a system at a given time.
///
/// This is very similar to a Hashmap (with likely worse performance),
/// but provides strong type separation.
#[derive(Debug, Clone)]
pub struct InputState<'a> {
    current_time: Time,
    /// Index of `current_time` in the time axis that most items share.
    time_index: Option<usize>,
    state: Items<'a>,
    /// Optional transform context for grid aggregation
    transform_context: Option<Cow<'a, TransformContext>>,
    /// Position after the last item read, which is checked first.
    next: Cell<usize>,
}

/// Where a component's inputs sit in a collection and how each is read, in declaration order.
#[derive(Debug, Clone, Default)]
pub(crate) struct InputPlan {
    pub(crate) indices: Vec<usize>,
    /// Positions in [`lookup_order`] of their names.
    pub(crate) order: Vec<u32>,
    pub(crate) transforms: Vec<Option<ReadTransformInfo>>,
}

#[derive(Debug, Clone)]
enum Items<'a> {
    Listed(Vec<&'a TimeseriesItem>),
    Planned {
        items: &'a [TimeseriesItem],
        plan: &'a InputPlan,
    },
}

impl<'a> Items<'a> {
    fn get(&self, position: usize) -> Option<&'a TimeseriesItem> {
        match self {
            Items::Listed(items) => items.get(position).copied(),
            Items::Planned { items, plan } => plan.indices.get(position).map(|&i| &items[i]),
        }
    }

    fn iter(&self) -> impl Iterator<Item = &'a TimeseriesItem> + '_ {
        (0..).map_while(|position| self.get(position))
    }
}

/// Order names by length and then by name, which settles most comparisons by length alone.
pub(crate) fn lookup_order(a: &str, b: &str) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

impl<'a> InputState<'a> {
    pub fn build(values: Vec<&'a TimeseriesItem>, current_time: Time) -> Self {
        Self {
            current_time,
            time_index: None,
            state: Items::Listed(values),
            transform_context: None,
            next: Cell::new(0),
        }
    }

    /// Build an InputState with transform context for grid aggregation.
    pub fn build_with_transforms(
        values: Vec<&'a TimeseriesItem>,
        current_time: Time,
        transform_context: TransformContext,
    ) -> Self {
        Self {
            current_time,
            time_index: None,
            state: Items::Listed(values),
            transform_context: Some(Cow::Owned(transform_context)),
            next: Cell::new(0),
        }
    }

    /// Build an InputState that borrows an optional transform context.
    pub fn build_with_transforms_ref(
        values: Vec<&'a TimeseriesItem>,
        current_time: Time,
        transform_context: Option<&'a TransformContext>,
    ) -> Self {
        Self {
            current_time,
            time_index: None,
            state: Items::Listed(values),
            transform_context: transform_context.map(Cow::Borrowed),
            next: Cell::new(0),
        }
    }

    pub fn empty() -> Self {
        Self {
            current_time: f64::NAN,
            time_index: None,
            state: Items::Listed(vec![]),
            transform_context: None,
            next: Cell::new(0),
        }
    }

    /// Test if the state contains a value with the given name
    pub fn has(&self, name: &str) -> bool {
        self.find(name).is_some()
    }

    /// Build an InputState over the planned `items` of a collection, without collecting them.
    pub(crate) fn planned(
        items: &'a [TimeseriesItem],
        plan: &'a InputPlan,
        current_time: Time,
    ) -> Self {
        Self {
            current_time,
            time_index: None,
            state: Items::Planned { items, plan },
            transform_context: None,
            next: Cell::new(0),
        }
    }

    /// Set the index of the current time in the time axis that most items share.
    pub fn with_time_index(mut self, time_index: usize) -> Self {
        self.time_index = Some(time_index);
        self
    }

    /// Index of the current time in `axis`.
    pub(crate) fn time_index(&self, axis: &TimeAxis, name: &str) -> usize {
        self.time_index
            .filter(|&index| axis.at(index) == Some(self.current_time))
            .or_else(|| axis.index_of(self.current_time))
            .unwrap_or_else(|| {
                panic!(
                    "Time {} not found in timeseries '{}' time axis",
                    self.current_time, name
                )
            })
    }

    /// Item named `name`, by binary search when the items are in [`lookup_order`].
    fn find(&self, name: &str) -> Option<&'a TimeseriesItem> {
        self.position(name).and_then(|index| self.state.get(index))
    }

    /// Position of `name`, checking the one after the last read first,
    /// so reads in declaration order skip the search.
    fn position(&self, name: &str) -> Option<usize> {
        let next = self.next.get();
        let position = if self.state.get(next).is_some_and(|item| item.name == name) {
            Some(next)
        } else {
            match &self.state {
                Items::Listed(items) => items
                    .binary_search_by(|item| lookup_order(&item.name, name))
                    .ok()
                    .or_else(|| items.iter().position(|item| item.name == name)),
                Items::Planned { items, plan } => plan
                    .order
                    .binary_search_by(|&p| {
                        lookup_order(&items[plan.indices[p as usize]].name, name)
                    })
                    .ok()
                    .map(|k| plan.order[k] as usize),
            }
        }?;
        self.next.set(position + 1);
        Some(position)
    }

    /// Item named `name` and its read transform, panicking if there is no such item.
    fn lookup(&self, name: &str) -> (&'a TimeseriesItem, Option<&ReadTransformInfo>) {
        let index = self
            .position(name)
            .unwrap_or_else(|| panic!("Variable '{}' not found in input state", name));
        let transform = match &self.state {
            Items::Planned { plan, .. } => plan.transforms[index].as_ref(),
            Items::Listed(_) => self
                .transform_context
                .as_ref()
                .and_then(|ctx| ctx.read_transforms.get(name)),
        };
        let item = self.state.get(index).expect("a position within the items");
        (item, transform)
    }

    pub fn iter(&self) -> impl Iterator<Item = &'a TimeseriesItem> + '_ {
        self.state.iter()
    }

    /// Get the current time
    pub fn current_time(&self) -> Time {
        self.current_time
    }

    /// Get a scalar TimeseriesWindow for the named variable
    ///
    /// This provides zero-cost access to current, previous, and historical values.
    /// If the underlying data is stored on a finer grid (FourBox or Hemispheric)
    /// and a read transform is configured, the data will be automatically aggregated
    /// to a scalar value on each access.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or cannot be accessed as scalar.
    pub fn get_scalar_window(&self, name: &str) -> ScalarWindow<'_> {
        let (item, transform) = self.lookup(name);
        self.scalar_window(name, item, transform)
            .with_timeseries_type(timeseries_type(transform))
    }

    fn scalar_window(
        &self,
        name: &str,
        item: &'a TimeseriesItem,
        transform: Option<&ReadTransformInfo>,
    ) -> ScalarWindow<'_> {
        // If there's a transform, use the source grid and unit conversion factor
        if let Some(transform) = transform {
            let unit_factor = transform.unit_conversion_factor;

            match transform.source_grid {
                GridType::FourBox => {
                    let ts = item.data.as_four_box().unwrap_or_else(|| {
                        panic!(
                            "Variable '{}' requires a four-box to scalar transform but is not a four-box timeseries",
                            name
                        )
                    });

                    let time_index = self.time_index(ts.time_axis_ref(), name);

                    return ScalarWindow::FromFourBox(
                        AggregatingFourBoxWindow::with_borrowed_weights(
                            ts,
                            time_index,
                            &item.name,
                            transform.weights.as_deref(),
                            unit_factor,
                            transform.variable_source,
                        ),
                    );
                }
                GridType::Hemispheric => {
                    let ts = item
                        .data
                        .as_hemispheric()
                        .unwrap_or_else(|| {
                            panic!(
                                "Variable '{}' requires Hemispheric->Scalar transform but is not Hemispheric",
                                name
                            )
                        });

                    let time_index = self.time_index(ts.time_axis_ref(), name);

                    return ScalarWindow::FromHemispheric(
                        AggregatingHemisphericWindow::with_borrowed_weights(
                            ts,
                            time_index,
                            &item.name,
                            transform.weights.as_deref(),
                            unit_factor,
                            transform.variable_source,
                        ),
                    );
                }
                GridType::Scalar => {
                    // Scalar->Scalar with possible unit conversion
                    let ts = item.data.as_scalar().unwrap_or_else(|| {
                        panic!("Variable '{}' is not a scalar timeseries", name)
                    });

                    let time_index = self.time_index(ts.time_axis_ref(), name);

                    return ScalarWindow::Direct(TimeseriesWindow::with_source(
                        ts,
                        time_index,
                        &item.name,
                        unit_factor,
                        transform.variable_source,
                    ));
                }
            }
        }

        // Direct scalar access (no transform needed)
        let ts = item
            .data
            .as_scalar()
            .unwrap_or_else(|| panic!("Variable '{}' is not a scalar timeseries", name));

        let time_index = self.time_index(ts.time_axis_ref(), name);

        let source = transform.map_or(VariableSource::Prescribed, |t| t.variable_source);

        ScalarWindow::Direct(TimeseriesWindow::with_source(
            ts, time_index, &item.name, 1.0, // no unit conversion for direct access
            source,
        ))
    }

    /// Get a FourBox GridTimeseriesWindow for the named variable
    ///
    /// If a read transform with unit conversion is configured for this variable,
    /// the conversion factor will be applied to all returned values.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or is not a four-box timeseries.
    pub fn get_four_box_window(&self, name: &str) -> GridTimeseriesWindow<'_, FourBoxGrid> {
        let (item, transform) = self.lookup(name);

        let ts = item
            .data
            .as_four_box()
            .unwrap_or_else(|| panic!("Variable '{}' is not a four-box timeseries", name));

        // Find the index corresponding to current_time in the timeseries
        let time_index = self.time_index(ts.time_axis_ref(), name);

        // Check if there's a unit conversion factor and variable source
        let (unit_factor, source) = transform
            .map(|t| (t.unit_conversion_factor, t.variable_source))
            .unwrap_or((1.0, VariableSource::Prescribed));

        GridTimeseriesWindow::with_source(ts, time_index, &item.name, unit_factor, source)
            .with_timeseries_type(timeseries_type(transform))
    }

    /// Get a Hemispheric GridTimeseriesWindow for the named variable
    ///
    /// If the underlying data is stored on the FourBox grid and a read transform
    /// is configured, the data will be automatically aggregated to Hemispheric
    /// on each access.
    ///
    /// If a read transform with unit conversion is configured for this variable,
    /// the conversion factor will be applied to all returned values.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or cannot be accessed as Hemispheric.
    pub fn get_hemispheric_window(&self, name: &str) -> HemisphericWindow<'_> {
        let (item, transform) = self.lookup(name);
        self.hemispheric_window(name, item, transform)
            .with_timeseries_type(timeseries_type(transform))
    }

    fn hemispheric_window(
        &self,
        name: &str,
        item: &'a TimeseriesItem,
        transform: Option<&ReadTransformInfo>,
    ) -> HemisphericWindow<'_> {
        // If there's a transform from FourBox, aggregate
        if let Some(transform) = transform {
            let unit_factor = transform.unit_conversion_factor;

            if transform.source_grid == GridType::FourBox {
                let ts = item.data.as_four_box().unwrap_or_else(|| {
                    panic!(
                        "Variable '{}' requires a four-box to hemispheric transform but is not a four-box timeseries",
                        name
                    )
                });

                let time_index = self.time_index(ts.time_axis_ref(), name);

                return HemisphericWindow::FromFourBox(
                    AggregatingFourBoxToHemisphericWindow::with_source(
                        ts,
                        time_index,
                        &item.name,
                        unit_factor,
                        transform.variable_source,
                    ),
                );
            }

            // Hemispheric->Hemispheric with possible unit conversion
            if transform.source_grid == GridType::Hemispheric {
                let ts = item.data.as_hemispheric().unwrap_or_else(|| {
                    panic!("Variable '{}' is not a Hemispheric timeseries", name)
                });

                let time_index = self.time_index(ts.time_axis_ref(), name);

                return HemisphericWindow::Direct(GridTimeseriesWindow::with_source(
                    ts,
                    time_index,
                    &item.name,
                    unit_factor,
                    transform.variable_source,
                ));
            }
        }

        // Direct hemispheric access (no transform)
        let ts = item
            .data
            .as_hemispheric()
            .unwrap_or_else(|| panic!("Variable '{}' is not a Hemispheric timeseries", name));

        let time_index = self.time_index(ts.time_axis_ref(), name);

        let source = transform.map_or(VariableSource::Prescribed, |t| t.variable_source);

        HemisphericWindow::Direct(GridTimeseriesWindow::with_source(
            ts, time_index, &item.name, 1.0, source,
        ))
    }
}

/// Point or average type from a read transform, a point without one.
fn timeseries_type(transform: Option<&ReadTransformInfo>) -> TimeseriesType {
    transform.map_or(TimeseriesType::Point, |t| t.timeseries_type)
}

impl<'a> IntoIterator for InputState<'a> {
    type Item = &'a TimeseriesItem;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.state.iter().collect::<Vec<_>>().into_iter()
    }
}

/// Output state from a component
///
/// A collection of named values that a component produces. Each value can be:
/// - `StateValue::Scalar` for global/non-spatial values
/// - `StateValue::FourBox` for four-box regional values
/// - `StateValue::Hemispheric` for hemispheric values
///
/// The model writes these values to the appropriate timeseries based on the
/// variable's grid in `VariableDefinition`.
pub type OutputState = HashMap<String, StateValue>;

/// Values a component writes in one step, held by the position of each declared output.
#[derive(Debug, Clone, Default)]
pub struct OutputBuffer {
    names: Vec<String>,
    /// Positions in [`lookup_order`] of their names.
    order: Vec<u32>,
    values: Vec<Option<StateValue>>,
    undeclared: Vec<(String, StateValue)>,
}

impl OutputBuffer {
    /// A buffer for the outputs `names`, usually `Component::output_names`.
    pub fn new(names: Vec<String>) -> Self {
        let mut order: Vec<u32> = (0..names.len() as u32).collect();
        order.sort_by(|&a, &b| lookup_order(&names[a as usize], &names[b as usize]));
        Self {
            values: vec![None; names.len()],
            names,
            order,
            undeclared: Vec::new(),
        }
    }

    /// Outputs `f` writes to a fresh buffer, for a `solve_with_state` over `solve_into`.
    pub fn collect(
        names: Vec<String>,
        f: impl FnOnce(&mut OutputWriter) -> crate::errors::RSCMResult<()>,
    ) -> crate::errors::RSCMResult<OutputState> {
        let mut buffer = Self::new(names);
        f(&mut buffer.writer())?;
        Ok(buffer.into_output_state())
    }

    /// Write access for one step.
    pub fn writer(&mut self) -> OutputWriter<'_> {
        OutputWriter {
            buffer: self,
            next: 0,
        }
    }

    pub(crate) fn names(&self) -> &[String] {
        &self.names
    }

    /// Take the value written to the declared output at `position`.
    pub(crate) fn take(&mut self, position: usize) -> Option<StateValue> {
        self.values[position].take()
    }

    /// Take the values written to names that are not declared.
    pub(crate) fn take_undeclared(&mut self) -> Vec<(String, StateValue)> {
        std::mem::take(&mut self.undeclared)
    }

    /// Drop every value written so far.
    pub(crate) fn clear(&mut self) {
        self.values.fill(None);
        self.undeclared.clear();
    }

    pub fn into_output_state(self) -> OutputState {
        self.names
            .into_iter()
            .zip(self.values)
            .filter_map(|(name, value)| Some((name, value?)))
            .chain(self.undeclared)
            .collect()
    }

    fn position(&self, name: &str, next: usize) -> Option<usize> {
        if self.names.get(next).is_some_and(|n| n == name) {
            return Some(next);
        }
        self.order
            .binary_search_by(|&p| lookup_order(&self.names[p as usize], name))
            .ok()
            .map(|k| self.order[k] as usize)
    }
}

/// Write access to a component's outputs for one step.
///
/// Writing outputs in declaration order skips the search for each name.
pub struct OutputWriter<'a> {
    buffer: &'a mut OutputBuffer,
    next: usize,
}

impl OutputWriter<'_> {
    /// Write `value` to the output `name`.
    pub fn set(&mut self, name: &str, value: impl Into<StateValue>) {
        let value = value.into();
        match self.buffer.position(name, self.next) {
            Some(position) => {
                self.buffer.values[position] = Some(value);
                self.next = position + 1;
            }
            None => self.buffer.undeclared.push((name.to_string(), value)),
        }
    }

    /// Write every value of `state`.
    pub fn extend(&mut self, state: OutputState) {
        for (name, value) in state {
            match self.buffer.position(&name, self.next) {
                Some(position) => self.buffer.values[position] = Some(value),
                None => self.buffer.undeclared.push((name, value)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeseries_collection::TimeseriesData;

    #[test]
    fn test_state_value_scalar() {
        let sv = StateValue::Scalar(42.0);
        assert!(sv.is_scalar());
        assert!(!sv.is_four_box());
        assert!(!sv.is_hemispheric());
        assert_eq!(sv.as_scalar(), Some(42.0));
        assert_eq!(sv.as_four_box(), None);
        assert_eq!(sv.as_hemispheric(), None);
        assert_eq!(sv.to_scalar(), 42.0);
    }

    #[test]
    fn test_state_value_four_box() {
        let slice = FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0]);
        let sv = StateValue::FourBox(slice);
        assert!(!sv.is_scalar());
        assert!(sv.is_four_box());
        assert!(!sv.is_hemispheric());
        assert_eq!(sv.as_scalar(), None);
        assert_eq!(sv.as_four_box(), Some(&slice));
        assert_eq!(sv.as_hemispheric(), None);
        assert_eq!(sv.to_scalar(), 2.5); // Mean of [1, 2, 3, 4]
    }

    #[test]
    fn test_state_value_hemispheric() {
        let slice = HemisphericSlice::from_array([10.0, 20.0]);
        let sv = StateValue::Hemispheric(slice);
        assert!(!sv.is_scalar());
        assert!(!sv.is_four_box());
        assert!(sv.is_hemispheric());
        assert_eq!(sv.as_scalar(), None);
        assert_eq!(sv.as_four_box(), None);
        assert_eq!(sv.as_hemispheric(), Some(&slice));
        assert_eq!(sv.to_scalar(), 15.0); // Mean of [10, 20]
    }

    #[test]
    fn test_state_value_from_impls() {
        // Test From<FloatValue> for StateValue
        let sv: StateValue = 42.0.into();
        assert!(sv.is_scalar());
        assert_eq!(sv.as_scalar(), Some(42.0));

        // Test From<FourBoxSlice> for StateValue
        let slice = FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0]);
        let sv: StateValue = slice.into();
        assert!(sv.is_four_box());

        // Test From<HemisphericSlice> for StateValue
        let slice = HemisphericSlice::from_array([10.0, 20.0]);
        let sv: StateValue = slice.into();
        assert!(sv.is_hemispheric());
    }

    #[test]
    fn test_input_state_grid_values() {
        use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use crate::spatial::FourBoxGrid;
        use crate::timeseries::{GridTimeseries, TimeAxis};
        use numpy::array;
        use numpy::ndarray::Array2;
        use std::sync::Arc;

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

        let item = TimeseriesItem {
            data: TimeseriesData::FourBox(ts),
            name: "Temperature".to_string(),
        };

        // The step from 2000.0 ends at index 1
        let state = InputState::build(vec![&item], 2000.0);

        // Test get_four_box_window returns values at index 1 using get_all()
        let window = state.get_four_box_window("Temperature");
        let values = window.get_all();
        assert_eq!(values, [16.0, 15.0, 11.0, 10.0]);
    }
}

#[cfg(test)]
mod input_state_window_tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
    use crate::spatial::{
        FourBoxGrid, FourBoxRegion, HemisphericGrid, HemisphericRegion, ScalarGrid,
    };
    use crate::timeseries::{GridTimeseries, TimeAxis, Timeseries};
    use crate::timeseries_collection::TimeseriesData;
    use numpy::array;
    use numpy::ndarray::{Array2, Axis};
    use std::sync::Arc;

    fn create_scalar_item(name: &str, values: Vec<FloatValue>) -> TimeseriesItem {
        // Create time axis that matches values length
        let n = values.len();
        let time_vals: Vec<f64> = (0..n).map(|i| 2000.0 + i as f64).collect();
        let time_axis = Arc::new(TimeAxis::from_values(ndarray::Array1::from_vec(time_vals)));
        let values_arr = ndarray::Array1::from_vec(values).insert_axis(Axis(1));
        let ts = Timeseries::new(
            values_arr,
            time_axis,
            ScalarGrid,
            "unit".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        TimeseriesItem {
            data: TimeseriesData::Scalar(ts),
            name: name.to_string(),
        }
    }

    fn create_four_box_item(name: &str) -> TimeseriesItem {
        let grid = FourBoxGrid::magicc_standard();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0, 2002.0]));
        let values = Array2::from_shape_vec(
            (3, 4),
            vec![
                15.0, 14.0, 10.0, 9.0, // 2000
                16.0, 15.0, 11.0, 10.0, // 2001
                17.0, 16.0, 12.0, 11.0, // 2002
            ],
        )
        .unwrap();
        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "C".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        TimeseriesItem {
            data: TimeseriesData::FourBox(ts),
            name: name.to_string(),
        }
    }

    fn create_hemispheric_item(name: &str) -> TimeseriesItem {
        let grid = HemisphericGrid::equal_weights();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values = Array2::from_shape_vec(
            (2, 2),
            vec![
                1000.0, 500.0, // 2000
                1100.0, 550.0, // 2001
            ],
        )
        .unwrap();
        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "mm/yr".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        TimeseriesItem {
            data: TimeseriesData::Hemispheric(ts),
            name: name.to_string(),
        }
    }

    #[test]
    fn test_get_scalar_window() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0, 290.0, 295.0, 300.0]);
        // Time 2001.0 corresponds to index 1 in the timeseries [2000, 2001, 2002, 2003, 2004]
        let state = InputState::build(vec![&item], 2001.0);

        let window = state.get_scalar_window("CO2");

        // A prescribed point's get() returns the value at the step's end
        assert_eq!(window.get(), 290.0);
        assert_eq!(window.interval().unwrap(), (285.0, 290.0));
        assert_eq!(window.history(1), Some(285.0));
        assert_eq!(window.len(), 5);
    }

    #[test]
    fn test_get_four_box_window() {
        let item = create_four_box_item("Temperature");
        // Time 2000.0 is index 0, so the step ends at index 1 in the timeseries [2000, 2001, 2002]
        let state = InputState::build(vec![&item], 2000.0);

        let window = state.get_four_box_window("Temperature");

        // get() returns values at index 1 (2001 values: [16.0, 15.0, 11.0, 10.0])
        assert_eq!(window.get(FourBoxRegion::NorthernOcean), 16.0);
        assert_eq!(window.get(FourBoxRegion::SouthernLand), 10.0);
        assert_eq!(window.get_all(), vec![16.0, 15.0, 11.0, 10.0]);
        // history(1) is index 0 (2000 values: [15.0, 14.0, 10.0, 9.0])
        assert_eq!(window.history(1), Some(vec![15.0, 14.0, 10.0, 9.0]));
    }

    #[test]
    fn test_get_hemispheric_window() {
        let item = create_hemispheric_item("Precipitation");
        // Time 2000.0 is index 0, so the step ends at index 1 in the timeseries [2000, 2001]
        let state = InputState::build(vec![&item], 2000.0);

        let window = state.get_hemispheric_window("Precipitation");

        // get() returns values at index 1 (2001 values: [1100.0, 550.0])
        assert_eq!(window.get(HemisphericRegion::Northern), 1100.0);
        assert_eq!(window.get(HemisphericRegion::Southern), 550.0);
    }

    #[test]
    #[should_panic(expected = "Variable 'NonExistent' not found")]
    fn test_get_scalar_window_missing_variable() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0]);
        let state = InputState::build(vec![&item], 2000.0);
        let _ = state.get_scalar_window("NonExistent");
    }

    #[test]
    #[should_panic(expected = "not a scalar timeseries")]
    fn test_get_scalar_window_wrong_type() {
        let item = create_four_box_item("Temperature");
        let state = InputState::build(vec![&item], 2000.0);
        // Attempting to get scalar window for a four-box variable should panic
        let _ = state.get_scalar_window("Temperature");
    }

    #[test]
    #[should_panic(expected = "not a four-box timeseries")]
    fn test_get_four_box_window_wrong_type() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0]);
        let state = InputState::build(vec![&item], 2000.0);
        let _ = state.get_four_box_window("CO2");
    }

    #[test]
    fn test_current_time_accessor() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0]);
        let state = InputState::build(vec![&item], 2023.5);
        assert_eq!(state.current_time(), 2023.5);
    }

    // =========================================================================
    // Unit conversion integration tests through InputState
    // =========================================================================

    #[test]
    fn test_input_state_scalar_window_with_unit_conversion() {
        use std::collections::HashMap;

        let item = create_scalar_item("Emissions", vec![1.0, 2.0, 3.0]);

        // Create transform context with unit conversion factor
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Emissions".to_string(),
            ReadTransformInfo::with_unit_conversion(1000.0), // Convert Gt to Mt
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2000.0, transform_ctx);

        let window = state.get_scalar_window("Emissions");

        // Raw value at index 1 is 2.0, converted should be 2000.0
        assert_eq!(window.get(), 2000.0);
        assert_eq!(window.interval().unwrap(), (1000.0, 2000.0));
        assert_eq!(window.history(1), Some(1000.0));
    }

    #[test]
    fn test_input_state_four_box_window_with_unit_conversion() {
        use std::collections::HashMap;

        let item = create_four_box_item("Temperature");

        // Create transform context with unit conversion factor
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Temperature".to_string(),
            ReadTransformInfo::with_unit_conversion(0.1), // Convert from deci-degrees to degrees
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2000.0, transform_ctx);

        let window = state.get_four_box_window("Temperature");

        // Raw values at index 1: [16.0, 15.0, 11.0, 10.0]
        // Converted: [1.6, 1.5, 1.1, 1.0]
        assert_eq!(window.get(FourBoxRegion::NorthernOcean), 1.6);
        assert_eq!(window.get_all(), vec![1.6, 1.5, 1.1, 1.0]);
    }

    #[test]
    fn test_input_state_four_box_to_scalar_with_unit_conversion() {
        use crate::component::GridType;
        use std::collections::HashMap;

        let item = create_four_box_item("Forcing");

        // Create transform context with grid aggregation AND unit conversion
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Forcing".to_string(),
            ReadTransformInfo::new(GridType::FourBox, None).with_conversion_factor(2.0),
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2000.0, transform_ctx);

        let window = state.get_scalar_window("Forcing");

        // Raw values at index 1: [16.0, 15.0, 11.0, 10.0], mean = 13.0
        // Converted: 13.0 * 2.0 = 26.0
        assert_eq!(window.get(), 26.0);
    }

    #[test]
    fn test_input_state_hemispheric_window_with_unit_conversion() {
        use crate::component::GridType;
        use std::collections::HashMap;

        let item = create_hemispheric_item("Precipitation");

        // Create transform context with unit conversion
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Precipitation".to_string(),
            ReadTransformInfo {
                source_grid: GridType::Hemispheric,
                weights: None,
                unit_conversion_factor: 0.001, // Convert mm/yr to m/yr
                variable_source: VariableSource::Prescribed,
                timeseries_type: TimeseriesType::Point,
            },
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2000.0, transform_ctx);

        let window = state.get_hemispheric_window("Precipitation");

        // Raw values at index 1: [1100.0, 550.0]
        // Converted: [1.1, 0.55]
        assert_eq!(window.get(HemisphericRegion::Northern), 1.1);
        assert_eq!(window.get(HemisphericRegion::Southern), 0.55);
    }

    #[test]
    fn test_input_state_four_box_to_hemispheric_with_unit_conversion() {
        use crate::component::GridType;
        use std::collections::HashMap;

        let item = create_four_box_item("Temperature");

        // Create transform context to aggregate FourBox -> Hemispheric with conversion
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Temperature".to_string(),
            ReadTransformInfo::new(GridType::FourBox, None).with_conversion_factor(2.0),
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2000.0, transform_ctx);

        let window = state.get_hemispheric_window("Temperature");

        // Raw values at index 1: [16.0, 15.0, 11.0, 10.0]
        // Hemispheric aggregation: Northern = (16+15)/2 = 15.5, Southern = (11+10)/2 = 10.5
        // With 2.0 conversion: [31.0, 21.0]
        assert_eq!(window.get(HemisphericRegion::Northern), 31.0);
        assert_eq!(window.get(HemisphericRegion::Southern), 21.0);
    }

    #[test]
    fn test_input_state_windows_carry_timeseries_type() {
        use crate::component::GridType;
        use std::collections::HashMap;

        let emissions = create_scalar_item("Emissions", vec![1.0, 2.0, 3.0]);
        let temperature = create_four_box_item("Temperature");
        let read_transforms = HashMap::from([
            (
                "Emissions".to_string(),
                ReadTransformInfo::default().with_timeseries_type(TimeseriesType::Average),
            ),
            (
                "Temperature".to_string(),
                ReadTransformInfo::new(GridType::FourBox, None)
                    .with_source(VariableSource::Upstream),
            ),
        ]);
        let state = InputState::build_with_transforms(
            vec![&emissions, &temperature],
            2001.0,
            TransformContext { read_transforms },
        );

        let emissions = state.get_scalar_window("Emissions");
        assert_eq!(emissions.timeseries_type(), TimeseriesType::Average);
        assert!(emissions.interval().is_err());

        let temperature = state.get_hemispheric_window("Temperature");
        assert_eq!(temperature.timeseries_type(), TimeseriesType::Point);
        assert_eq!(temperature.get_time(), 2002.0);
        assert_eq!(
            temperature.interval().unwrap(),
            (vec![15.5, 10.5], vec![16.5, 11.5])
        );
        assert_eq!(
            state
                .get_four_box_window("Temperature")
                .interval()
                .unwrap()
                .1,
            vec![17.0, 16.0, 12.0, 11.0]
        );
    }

    #[test]
    fn planned_state_reads_in_any_order() {
        let items = vec![
            create_scalar_item("Long name", vec![1.0, 2.0]),
            create_scalar_item("B", vec![3.0, 4.0]),
            create_scalar_item("Unread", vec![0.0, 0.0]),
            create_scalar_item("A", vec![5.0, 6.0]),
        ];
        let plan = InputPlan {
            indices: vec![0, 3, 1],
            order: vec![1, 2, 0],
            transforms: vec![None, None, None],
        };
        let state = InputState::planned(&items, &plan, 2000.0);
        for (name, expected) in [
            ("Long name", 2.0),
            ("A", 6.0),
            ("B", 4.0),
            ("B", 4.0),
            ("A", 6.0),
            ("Long name", 2.0),
        ] {
            assert_eq!(state.get_scalar_window(name).get(), expected, "{name}");
        }
        assert!(!state.has("Unread"));
        assert_eq!(
            state
                .iter()
                .map(|item| item.name.as_str())
                .collect::<Vec<_>>(),
            ["Long name", "A", "B"]
        );
    }
}

#[cfg(test)]
mod output_buffer_tests {
    use super::*;

    #[test]
    fn writes_by_name_in_any_order() {
        let mut buffer = OutputBuffer::new(vec!["Long name".into(), "B".into(), "A".into()]);
        let mut out = buffer.writer();
        out.set("A", 1.0);
        out.set("Long name", 2.0);
        out.set("B", 3.0);
        out.set("Undeclared", 4.0);
        out.set("B", 5.0);
        assert_eq!(buffer.take(1), Some(StateValue::Scalar(5.0)));
        assert_eq!(buffer.take(1), None);
        assert_eq!(
            buffer.take_undeclared(),
            [("Undeclared".to_string(), StateValue::Scalar(4.0))]
        );
        let state = buffer.into_output_state();
        assert_eq!(state.len(), 2);
        assert_eq!(state["A"], StateValue::Scalar(1.0));
        assert_eq!(state["Long name"], StateValue::Scalar(2.0));
    }

    #[test]
    fn collect_extends_from_an_output_state() {
        let state = OutputBuffer::collect(vec!["A".into()], |out| {
            out.extend(OutputState::from([
                ("A".to_string(), StateValue::Scalar(1.0)),
                ("B".to_string(), StateValue::Scalar(2.0)),
            ]));
            Ok(())
        })
        .unwrap();
        assert_eq!(state.len(), 2);
        assert_eq!(state["B"], StateValue::Scalar(2.0));
    }
}

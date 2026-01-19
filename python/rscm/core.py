"""
Core classes and functions for Rust Simple Climate Models (RSCMs).

This module re-exports the core types from the Rust extension module,
providing the fundamental building blocks for constructing and running
climate models.

Model Building
--------------
ModelBuilder
    Builder pattern for assembling climate models from components.
Model
    A coupled set of components solved on a common time axis.

Time Series
-----------
TimeAxis
    Defines the temporal grid for model execution and data.
Timeseries
    Time-indexed data with interpolation support.
TimeseriesCollection
    Container for multiple named timeseries.
InterpolationStrategy
    Controls how values are interpolated between time points.

Components
----------
PythonComponent
    Wrapper to use Python components in Rust models.
RequirementDefinition
    Specifies a component's input/output requirements.
RequirementType
    Enum: Input, Output, State, or EmptyLink.
GridType
    Enum: Scalar, FourBox, or Hemispheric.

Spatial Grids
-------------
FourBoxGrid, FourBoxRegion, FourBoxSlice
    Four-box grid (MAGICC standard): NO, NL, SO, SL regions.
HemisphericGrid, HemisphericRegion, HemisphericSlice
    Two-region hemispheric grid: Northern, Southern.
ScalarGrid, ScalarRegion
    Single global value (default).

State Values
------------
StateValue
    Wrapper for scalar or grid values returned from components.
TimeseriesWindow
    Provides access to current and historical values within solve().
FourBoxTimeseriesWindow, HemisphericTimeseriesWindow
    Grid-typed variants of TimeseriesWindow.

Variable Schema
---------------
VariableSchema
    Declares model variables and aggregation relationships.
    Enables automatic computation of derived values (sums, means, weighted sums).

Examples
--------
Building a simple model:

>>> from rscm.core import ModelBuilder, TimeAxis, Timeseries
>>> import numpy as np
>>> model = (
...     ModelBuilder()
...     .with_time_axis(TimeAxis.from_values(np.arange(2000, 2101)))
...     .with_rust_component(my_component)
...     .with_exogenous_variable("Emissions", emissions_ts)
... ).build()
>>> model.run()
>>> results = model.timeseries()

See Also
--------
rscm.component : Base class for Python components
"""

from rscm._lib.core import (
    GridType,
    InterpolationStrategy,
    Model,
    ModelBuilder,
    PythonComponent,
    RequirementDefinition,
    RequirementType,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
    VariableSchema,
    VariableType,
)
from rscm._lib.core.spatial import (
    FourBoxGrid,
    FourBoxRegion,
    HemisphericGrid,
    HemisphericRegion,
    ScalarGrid,
    ScalarRegion,
)
from rscm._lib.core.state import (
    FourBoxSlice,
    FourBoxTimeseriesWindow,
    HemisphericSlice,
    HemisphericTimeseriesWindow,
    StateValue,
    TimeseriesWindow,
)

__all__ = [
    # Core types
    "FourBoxGrid",
    "FourBoxRegion",
    "FourBoxSlice",
    "FourBoxTimeseriesWindow",
    "GridType",
    "HemisphericGrid",
    "HemisphericRegion",
    "HemisphericSlice",
    "HemisphericTimeseriesWindow",
    "InterpolationStrategy",
    "Model",
    "ModelBuilder",
    "PythonComponent",
    "RequirementDefinition",
    "RequirementType",
    "ScalarGrid",
    "ScalarRegion",
    "StateValue",
    "TimeAxis",
    "Timeseries",
    "TimeseriesCollection",
    "TimeseriesWindow",
    "VariableSchema",
    "VariableType",
]

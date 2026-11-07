"""
Core classes and functions for Rust Simple Climate Models (RSCMs).

This module re-exports the core types from the Rust extension module,
providing the fundamental building blocks for constructing and running
climate models.

**Model building**

- [ModelBuilder][] assembles a climate model from components.
- [Model][] is a coupled set of components solved on a common time axis.
- [build_models][] and [run_models][] build and run many models in parallel.

**Time series**

- [TimeAxis][] holds the ordered time bounds a model or timeseries is defined on.
- [Timeseries][] holds time-indexed data with interpolation support.
- [FourBoxTimeseries][] and [HemisphericTimeseries][] are its grid variants.
- [TimeseriesCollection][] holds multiple named timeseries.
- [InterpolationStrategy][] controls how values are interpolated between time points.

**Components**

- [PythonComponent][] wraps a Python component for `ModelBuilder.with_component`.
- [VariableDefinition][] specifies one of a component's inputs or outputs.
- [VariableRole][] is Input, Output or State.
- [GridType][] is Scalar, FourBox or Hemispheric.
- [TimeseriesType][] is Point or Average.

**Spatial grids**

- [FourBoxGrid][], [FourBoxRegion][] and [FourBoxSlice][] describe the MAGICC
  four-box grid: Northern Ocean, Northern Land, Southern Ocean, Southern Land.
- [HemisphericGrid][], [HemisphericRegion][] and [HemisphericSlice][] describe the
  Northern and Southern hemispheres.
- [ScalarGrid][] and [ScalarRegion][] describe a single global value (the default).

**State values**

- [StateValue][] wraps the scalar or grid values a component returns.
- [TimeseriesWindow][] gives current and historical values within `solve()`.
- [FourBoxTimeseriesWindow][] and [HemisphericTimeseriesWindow][] are its grid variants.
- [VariableSource][] is where a window's values come from,
  which decides what `get()` reads.

**Variable schema**

- [VariableSchema][] declares model variables and aggregation relationships,
  so derived values (sums, means, weighted sums) are computed automatically.

**Units**

- [Unit][] is a physical unit with parsing, normalisation and conversion.

Examples
--------
Building a simple model:

>>> from rscm.core import ModelBuilder, TimeAxis, Timeseries
>>> import numpy as np
>>> model = (
...     ModelBuilder()
...     .with_time_axis(TimeAxis.from_values(np.arange(2000, 2101)))
...     .with_component(my_component)
...     .with_prescribed_variable("Emissions|CO2", emissions_ts)
... ).build()
>>> model.run()
>>> results = model.timeseries()

See Also
--------
rscm.component : Base class for Python components
"""

from rscm._lib.core import (
    FourBoxTimeseries,
    GridType,
    HemisphericTimeseries,
    InterpolationStrategy,
    Model,
    ModelBuilder,
    PythonComponent,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
    TimeseriesType,
    Unit,
    VariableDefinition,
    VariableRole,
    VariableSchema,
    build_models,
    run_models,
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
    VariableSource,
)

__all__ = [
    # Core types
    "FourBoxGrid",
    "FourBoxRegion",
    "FourBoxSlice",
    "FourBoxTimeseries",
    "FourBoxTimeseriesWindow",
    "GridType",
    "HemisphericGrid",
    "HemisphericRegion",
    "HemisphericSlice",
    "HemisphericTimeseries",
    "HemisphericTimeseriesWindow",
    "InterpolationStrategy",
    "Model",
    "ModelBuilder",
    "PythonComponent",
    "ScalarGrid",
    "ScalarRegion",
    "StateValue",
    "TimeAxis",
    "Timeseries",
    "TimeseriesCollection",
    "TimeseriesType",
    "TimeseriesWindow",
    "Unit",
    "VariableDefinition",
    "VariableRole",
    "VariableSchema",
    "VariableSource",
    "build_models",
    "run_models",
]

"""
Core classes and functions for Rust Simple Climate Models (RSCMs)
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
    "VariableType",
]

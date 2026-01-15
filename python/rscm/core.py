"""
Core classes and functions for Rust Simple Climate Models (RSCMs)
"""

from rscm._lib.core import (
    FourBoxGrid,
    FourBoxRegion,
    FourBoxSlice,
    FourBoxTimeseriesWindow,
    GridType,
    HemisphericGrid,
    HemisphericRegion,
    HemisphericSlice,
    HemisphericTimeseriesWindow,
    InterpolationStrategy,
    Model,
    ModelBuilder,
    PythonComponent,
    RequirementDefinition,
    RequirementType,
    ScalarGrid,
    ScalarRegion,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
    TimeseriesWindow,
    VariableType,
)

# Typed component base class and descriptors
from rscm.component import Component, Input, Output, State

__all__ = [
    # Typed component support
    "Component",
    "Input",
    "Output",
    "State",
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
    "TimeAxis",
    "Timeseries",
    "TimeseriesCollection",
    "TimeseriesWindow",
    "VariableType",
]

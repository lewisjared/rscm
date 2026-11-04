"""Run FaIR in RSCM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rscm.core import InterpolationStrategy, ModelBuilder, TimeAxis, Timeseries
from rscm.fair.config import FairConfig, build_components, initial_values

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

    from rscm.core import Model


def _on_timebounds(
    name: str, values: ArrayLike, n_timebounds: int
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (n_timebounds - 1,):
        # No step starts at the last bound, so its value is never read.
        return np.append(array, array[-1])
    if array.shape != (n_timebounds,):
        msg = (
            f"{name} has shape {array.shape}, "
            f"expected ({n_timebounds - 1},) timepoints or ({n_timebounds},) timebounds"
        )
        raise ValueError(msg)
    return array


def build_model(
    config: FairConfig,
    inputs: Mapping[str, tuple[ArrayLike, str]],
    timebounds: ArrayLike,
) -> Model:
    """Build an unrun model of a FaIR configuration.

    `inputs` maps each exogenous variable to its values and unit.
    Emissions are FaIR timepoint values, one per step.
    Prescribed concentrations and forcing are FaIR timebound values.
    """
    bounds = np.asarray(timebounds, dtype=np.float64)
    axis = TimeAxis.from_values(bounds)
    builder = ModelBuilder().with_time_axis(axis)
    for component in build_components(config):
        builder.with_rust_component(component)
    for name, (values, unit) in inputs.items():
        builder.with_exogenous_variable(
            name,
            Timeseries(
                _on_timebounds(name, values, len(bounds)),
                axis,
                unit,
                InterpolationStrategy.Previous,
            ),
        )
    builder.with_initial_values(initial_values(config))
    return builder.build()


def run_fair(
    config: FairConfig,
    inputs: Mapping[str, tuple[ArrayLike, str]],
    timebounds: ArrayLike,
) -> Model:
    """Run a FaIR configuration and return the model, see `build_model`."""
    model = build_model(config, inputs, timebounds)
    model.run()
    return model

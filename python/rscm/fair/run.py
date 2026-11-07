"""Run FaIR in RSCM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rscm.core import InterpolationStrategy, ModelBuilder, Timeseries
from rscm.fair import naming
from rscm.fair.config import (
    FairConfig,
    components,
    forcing_agents,
    initial_values,
    with_lags,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

    from rscm.core import Model, TimeAxis


def _on_timebounds(
    name: str, values: ArrayLike, n_timebounds: int
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    timepoints = name.startswith("Emissions|")
    if timepoints and array.shape == (n_timebounds - 1,):
        # No step starts at the last bound, so its value is never read.
        return np.append(array, array[-1])
    if array.shape != (n_timebounds,):
        expected = f"({n_timebounds - 1},) timepoints or " if timepoints else ""
        msg = (
            f"{name} has shape {array.shape}, "
            f"expected {expected}({n_timebounds},) timebounds"
        )
        raise ValueError(msg)
    return array


def prescribed_forcing_scales(config: FairConfig) -> dict[str, float]:
    """Return the `forcing_scale` of each prescribed forcing variable."""
    return {
        naming.prescribed_forcing(s): float(s["forcing_scale"])
        for s in config["species"]
        if s["input_mode"] == "forcing"
    }


def build_model(
    config: FairConfig,
    inputs: Mapping[str, tuple[ArrayLike, str]],
    time_axis: TimeAxis,
    *,
    scale_prescribed_forcing: bool = False,
) -> Model:
    """Build an unrun model of a FaIR configuration.

    `inputs` maps each prescribed variable to its values and unit.
    Emissions are FaIR timepoint values, one per step.
    Prescribed concentrations and forcing are FaIR timebound values.
    `time_axis` has a step starting at each FaIR timebound,
    `TimeAxis.from_values(timebounds)`.

    FaIR ignores `forcing_scale` for prescribed forcing.
    `scale_prescribed_forcing` applies it to the inputs, as fair-calibrate does.
    """
    n_timebounds = len(time_axis)
    scales = prescribed_forcing_scales(config) if scale_prescribed_forcing else {}
    builder = ModelBuilder().with_time_axis(time_axis)
    for component in components(config):
        builder.with_component(component)
    with_lags(builder, config)
    first = {}
    for name, (values, unit) in inputs.items():
        array = _on_timebounds(name, values, n_timebounds)
        if name in scales:
            array = array * scales[name]
        first[name] = float(array[0])
        builder.with_prescribed_variable(
            name, Timeseries(array, time_axis, unit, InterpolationStrategy.Previous)
        )
    initial = initial_values(config)
    agents = [naming.forcing(s) for s in forcing_agents(config)]
    # FaIR's first forcing is the prescribed forcing, before any feedback.
    for s in forcing_agents(config):
        name = naming.prescribed_forcing(s)
        if s["input_mode"] == "forcing" and name in first:
            initial[naming.forcing(s)] = first[name]
    total = float(np.nansum([initial[agent] for agent in agents]))
    initial[naming.TOTAL_FORCING] = initial[naming.ENERGY_IMBALANCE] = total
    builder.with_initial_values(initial)
    return builder.build()


def run_fair(
    config: FairConfig,
    inputs: Mapping[str, tuple[ArrayLike, str]],
    time_axis: TimeAxis,
    *,
    scale_prescribed_forcing: bool = False,
) -> Model:
    """Run a FaIR configuration and return the model, see `build_model`."""
    model = build_model(
        config, inputs, time_axis, scale_prescribed_forcing=scale_prescribed_forcing
    )
    model.run()
    return model

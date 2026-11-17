"""Model builder that constructs RSCM models from configuration."""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import InputSpec, ModelConfig, TimeConfig
from .exceptions import ValidationError
from .registry import component_registry

if TYPE_CHECKING:
    from rscm._lib.core import Model, Timeseries

__all__ = ["build_model"]

_TWO_LAYER_COMPONENT = "TwoLayer"


def build_model(
    config: ModelConfig | dict[str, Any],
    prescribed: Mapping[str, Timeseries] | None = None,
) -> Model:
    """Build a model from configuration.

    Parameters
    ----------
    config
        Model configuration (ModelConfig instance or dict from TOML)
    prescribed
        Prescribed data by variable name, such as the forcing.
        It replaces any `[inputs]` entry for the same variable.

    Returns
    -------
    Model
        Configured RSCM model ready to run

    Raises
    ------
    ValidationError
        If the configuration has unknown keys, an invalid time axis or input,
        or a component of the wrong type
    ComponentNotFoundError
        If a component type is not registered
    ValueError
        If the model type is unknown, or the model cannot be built,
        such as when no forcing is given
    """
    # Handle dict config
    if isinstance(config, dict):
        model_type = config.get("model", {}).get("type", "")
    else:
        model_type = config.model_type

    if model_type == "two-layer":
        return _two_layer_model(config, prescribed or {})

    msg = f"Unknown model type: {model_type!r}"
    raise ValueError(msg)


def _two_layer_model(config: Any, prescribed: Mapping[str, Timeseries]) -> Model:
    from rscm._lib.core import ModelBuilder, TimeAxis  # noqa: PLC0415
    from rscm.config.models import two_layer  # noqa: F401, PLC0415
    from rscm.names import (  # noqa: PLC0415
        SURFACE_AIR_TEMPERATURE_CHANGE,
        ocean_layer_temperature,
    )

    if isinstance(config, dict):
        params = _climate_parameters(config.get("components", {}))
        time = _time_config(config.get("time"))
        initial_values = dict(config.get("initial_values", {}))
        inputs = _input_specs(config.get("inputs", {}))
    else:
        params = asdict(config.climate)
        if config.time is None:
            raise ValidationError(_NO_TIME)
        time = config.time
        initial_values = dict(config.initial_values)
        inputs = dict(config.inputs)

    component = component_registry.get(_TWO_LAYER_COMPONENT).from_parameters(params)
    time_axis = TimeAxis.from_values(np.arange(time.start, time.end + 1, dtype=float))
    builder = (
        ModelBuilder()
        .with_time_axis(time_axis)
        .with_component(component.build())
        .with_initial_values(
            {
                SURFACE_AIR_TEMPERATURE_CHANGE: 0.0,
                ocean_layer_temperature(2): 0.0,
                **initial_values,
            }
        )
    )
    for name, spec in inputs.items():
        if name not in prescribed:
            constant = _constant_input(name, spec, time_axis)
            builder.with_prescribed_variable(name, constant)
    for name, timeseries in prescribed.items():
        builder.with_prescribed_variable(name, timeseries)
    return builder.build()


_NO_TIME = (
    "The config has no [time] table, so the model has no time axis. "
    "Add one with start and end years."
)


def _check_keys(table: str, values: Mapping[str, Any], known: set[str]) -> None:
    for key in values:
        if key not in known:
            close = difflib.get_close_matches(key, sorted(known), n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            msg = f"Unknown key {key!r} in {table}.{hint}"
            raise ValidationError(msg)


def _climate_parameters(components: Mapping[str, Any]) -> dict[str, Any]:
    _check_keys("[components]", components, {"climate"})
    if "climate" not in components:
        msg = "The config has no [components.climate] table."
        raise ValidationError(msg)
    climate = components["climate"]
    _check_keys("[components.climate]", climate, {"type", "parameters"})
    component_type = climate.get("type", _TWO_LAYER_COMPONENT)
    component_registry.get(component_type)
    if component_type != _TWO_LAYER_COMPONENT:
        msg = (
            f"[components.climate] type must be {_TWO_LAYER_COMPONENT!r} "
            f"for a two-layer model, not {component_type!r}."
        )
        raise ValidationError(msg)
    return dict(climate.get("parameters", {}))


def _time_config(time: Mapping[str, Any] | None) -> TimeConfig:
    if time is None:
        raise ValidationError(_NO_TIME)
    _check_keys("[time]", time, {"start", "end"})
    for key in ("start", "end"):
        value = time.get(key)
        if value is None:
            msg = f"[time] has no {key!r} year."
            raise ValidationError(msg)
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"[time] {key} must be a whole year, not {value!r}."
            raise ValidationError(msg)
    try:
        return TimeConfig(start=time["start"], end=time["end"])
    except ValueError as error:
        msg = f"[time] {error}"
        raise ValidationError(msg) from error


def _input_specs(inputs: Mapping[str, Any]) -> dict[str, InputSpec]:
    known = {f.name for f in fields(InputSpec)}
    specs = {}
    for name, spec in inputs.items():
        _check_keys(f"[inputs.{name!r}]", spec, known)
        specs[name] = InputSpec(**spec)
    return specs


def _constant_input(name: str, spec: InputSpec, time_axis: Any) -> Timeseries:
    from rscm._lib.core import InterpolationStrategy, Timeseries  # noqa: PLC0415

    if spec.file is not None:
        msg = (
            f"Input {name!r} reads {spec.file!r}, "
            "but input files are not supported yet. "
            "Give a constant value, or pass the data to build_model as prescribed."
        )
        raise ValidationError(msg)
    if spec.value is None or spec.unit is None:
        msg = f"Input {name!r} needs a value and a unit."
        raise ValidationError(msg)
    return Timeseries(
        np.full(len(time_axis), float(spec.value)),
        time_axis,
        spec.unit,
        InterpolationStrategy.Previous,
    )

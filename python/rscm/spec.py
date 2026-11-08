"""
Describe a model as data before building it.

A [ModelSpec][rscm.spec.ModelSpec] holds a model's components with their parameters,
its lags, initial values and grid weights.
[bind][rscm.spec.ModelSpec.bind] adds prescribed data and a time axis
to give a [ModelSetup][rscm.spec.ModelSetup],
which lists the inputs the model needs, summarises itself and builds the model.

Model packages return a setup from their configuration,
such as [rscm.fair.setup_model][rscm.fair.run.setup_model]
and [rscm.magicc.emissions_driven.setup_model][].
A new model can write its spec by hand:

>>> from rscm.two_layer import TwoLayerBuilder
>>> spec = ModelSpec(
...     components={"TwoLayer": ComponentSpec(TwoLayerBuilder, parameters)},
...     initial_values={
...         "Surface Air Temperature Change": 0.0,
...         "Ocean Temperature|Layer 2": 0.0,
...     },
... )
>>> setup = spec.bind({"Effective Radiative Forcing": forcing}, time_axis)
>>> print(setup)
>>> model = setup.run()
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rscm.core import (
    FourBoxTimeseries,
    HemisphericTimeseries,
    ModelBuilder,
    Timeseries,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rscm.core import GridType, Model, PrescribedInput, TimeAxis, VariableSchema

    Prescribed = Timeseries | FourBoxTimeseries | HemisphericTimeseries

__all__ = ["ComponentSpec", "Lag", "ModelSetup", "ModelSpec", "diff"]


@dataclass(frozen=True)
class ComponentSpec:
    """A component as its builder and the parameters it is built from."""

    builder: Any
    """A component builder such as `ClimateUDEBBuilder`, or a Python component class."""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def configure(self) -> Any:
        """Return the builder configured with the parameters, before building."""
        return self.builder.from_parameters(dict(self.parameters))

    def build(self) -> Any:
        """Build the component.

        A builder with `from_parameters` is given the parameters as a dict.
        Any other class is called with them as keyword arguments.
        """
        if hasattr(self.builder, "from_parameters"):
            return self.configure().build()
        return self.builder(**self.parameters)


@dataclass(frozen=True)
class Lag:
    """A lag on ``variable``, into ``consumer`` alone or into every consumer."""

    variable: str
    consumer: str | None = None

    def add_to(self, builder: ModelBuilder) -> None:
        """Add the lag to a model builder."""
        if self.consumer is None:
            builder.with_lag(self.variable)
        else:
            builder.with_lag_for(self.variable, self.consumer)

    def __str__(self) -> str:
        return f"{self.variable} -> {self.consumer or 'every consumer'}"


@dataclass(frozen=True)
class ModelSpec:
    """The components, lags and initial state of a model, before it has inputs."""

    components: Mapping[str, ComponentSpec]
    """Each component by name, in the order they are added to the model."""
    lags: Sequence[Lag] = ()
    initial_values: Mapping[str, float] = field(default_factory=dict)
    initial_grid_values: Mapping[str, Sequence[float]] = field(default_factory=dict)
    grid_weights: Mapping[GridType, Sequence[float]] = field(default_factory=dict)
    schema: VariableSchema | None = None
    name: str = "Model"

    def bind(
        self, prescribed: Mapping[str, Prescribed], time_axis: TimeAxis
    ) -> ModelSetup:
        """Return the setup of this model with prescribed data on a time axis."""
        return ModelSetup(self, dict(prescribed), time_axis)


@dataclass(frozen=True)
class ModelSetup:
    """A model spec with its prescribed data and time axis, ready to build."""

    spec: ModelSpec
    prescribed: Mapping[str, Prescribed]
    time_axis: TimeAxis

    def builder(self) -> ModelBuilder:
        """Return a model builder holding everything in the setup."""
        spec = self.spec
        builder = ModelBuilder().with_time_axis(self.time_axis)
        if spec.schema is not None:
            builder.with_schema(spec.schema)
        for grid, weights in spec.grid_weights.items():
            builder.with_grid_weights(grid, list(weights))
        for component in spec.components.values():
            builder.with_component(component.build())
        for lag in spec.lags:
            lag.add_to(builder)
        for name, timeseries in self.prescribed.items():
            builder.with_prescribed_variable(name, timeseries)
        builder.with_initial_values(dict(spec.initial_values))
        builder.with_initial_grid_values(
            {name: list(values) for name, values in spec.initial_grid_values.items()}
        )
        return builder

    def prescribed_inputs(self) -> list[PrescribedInput]:
        """Return every input no component produces, and whether it is supplied."""
        return self.builder().prescribed_inputs()

    def build(self) -> Model:
        """Build the unrun model."""
        return self.builder().build()

    def run(self) -> Model:
        """Build and run the model and return it."""
        model = self.build()
        model.run()
        return model

    def describe(self) -> str:
        """Summarise the components, lags, inputs and initial state."""
        spec = self.spec
        bounds = np.asarray(self.time_axis.bounds())
        lines = [
            spec.name,
            f"  time axis  {bounds[0]:g} to {bounds[-1]:g}, "
            f"{len(self.time_axis)} steps",
            f"  components ({len(spec.components)}, in the order added)",
        ]
        lines += _table(
            (name, f"{len(c.parameters)} parameters")
            for name, c in spec.components.items()
        )
        lines.append(f"  lags ({len(spec.lags)})")
        lines += [f"    {lag}" for lag in spec.lags]
        groups = Counter(_group(name) for name in self.prescribed)
        lines.append(f"  prescribed data ({len(self.prescribed)})")
        lines += _table((group, str(n)) for group, n in sorted(groups.items()))
        missing = [
            i.definition.name for i in self.prescribed_inputs() if not i.supplied
        ]
        lines.append(f"  missing inputs: {', '.join(missing) or 'none'}")
        lines.append(
            f"  initial values: {len(spec.initial_values)} scalar, "
            f"{len(spec.initial_grid_values)} gridded"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def _table(rows: Any) -> list[str]:
    rows = list(rows)
    width = max((len(name) for name, _ in rows), default=0)
    return [f"    {name:<{width}}  {value}" for name, value in rows]


def _group(name: str) -> str:
    return name.split("|", 1)[0]


_IDENTITY_KEYS = ("name", "variable", "concentration")


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out |= _flatten(item, f"{prefix}.{key}" if prefix else str(key))
        return out
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        out = {}
        for index, item in enumerate(value):
            key = next((item[k] for k in _IDENTITY_KEYS if item.get(k)), index)
            out |= _flatten(item, f"{prefix}[{str(key).split('|')[-1]}]")
        return out
    return {prefix: value}


def _show(value: Any) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, list) and len(value) > 4:  # noqa: PLR2004
        return f"<{len(value)} values>"
    return repr(value)


def diff(a: ModelSetup, b: ModelSetup, *, limit: int = 6) -> str:
    """Describe what differs between two setups.

    Lists added and removed components, changed parameters, initial values,
    lags and the names of prescribed data.
    At most ``limit`` parameter changes are shown for each component.
    """
    lines = [f"{a.spec.name} -> {b.spec.name}"]
    ca, cb = a.spec.components, b.spec.components
    lines += [f"  - component {name}" for name in ca if name not in cb]
    lines += [f"  + component {name}" for name in cb if name not in ca]
    for name in (n for n in ca if n in cb):
        pa = _flatten(dict(ca[name].parameters), "")
        pb = _flatten(dict(cb[name].parameters), "")
        changed = sorted(k for k in pa.keys() | pb.keys() if pa.get(k) != pb.get(k))
        lines += [
            f"  ~ {name}.{k}: {_show(pa.get(k))} -> {_show(pb.get(k))}"
            for k in changed[:limit]
        ]
        if len(changed) > limit:
            lines.append(f"  ~ {name}: {len(changed) - limit} more parameters")
    ia, ib = a.spec.initial_values, b.spec.initial_values
    counts = {
        "changed": sum(ia[k] != ib[k] for k in ia.keys() & ib.keys()),
        "added": len(ib.keys() - ia.keys()),
        "removed": len(ia.keys() - ib.keys()),
    }
    if any(counts.values()):
        shown = ", ".join(f"{n} {kind}" for kind, n in counts.items() if n)
        lines.append(f"  ~ initial values: {shown}")
    la, lb = set(a.spec.lags), set(b.spec.lags)
    lines += [f"  - lag {lag}" for lag in sorted(la - lb, key=str)]
    lines += [f"  + lag {lag}" for lag in sorted(lb - la, key=str)]
    for sign, names in (
        ("-", a.prescribed.keys() - b.prescribed.keys()),
        ("+", b.prescribed.keys() - a.prescribed.keys()),
    ):
        groups = Counter(_group(n) for n in names)
        for group, count in sorted(groups.items()):
            members = sorted(n for n in names if _group(n) == group)
            if count == 1:
                lines.append(f"  {sign} prescribed {members[0]}")
                continue
            parts = [n.split("|", 1)[-1] for n in members]
            shown = ", ".join(parts[:4]) + (", ..." if count > 4 else "")  # noqa: PLR2004
            lines.append(f"  {sign} {count} prescribed {group}|{{{shown}}}")
    if len(lines) == 1:
        lines.append("  identical")
    return "\n".join(lines)

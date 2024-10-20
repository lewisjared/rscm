# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Components in Python
#
# We can also create new components in Python and pass them over to Rust.
# This enables the use of native Rust and Python-based components
# in the same coupled model.

# %%
import attrs
import matplotlib.pyplot as plt
import numpy as np

from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    PythonComponent,
    RequirementDefinition,
    RequirementType,
    TimeAxis,
    Timeseries,
)

# %% [markdown]
# A python-based component must conform with the
# `rscm._lib.core.CustomComponent` protocol.
# This protocol defines two required functions:
# * definitions
# * solve


# %%
@attrs.frozen
class ScaleComponent:
    """
    Scale the input by a factor after 2015

    Example of a custom Python component.

    This component must conform with the
    `rscm._lib.core.CustomComponent` protocol.
    """

    scale_factor: float
    scale_year: int

    def definitions(self) -> list[RequirementDefinition]:
        """
        Define the requirements for this component
        """
        # TODO: resolve later
        return [
            RequirementDefinition("input", "K", RequirementType.Input),
            RequirementDefinition("output", "K", RequirementType.Output),
        ]

    def solve(
        self, time_current: float, time_next: float, input_state: dict[str, float]
    ) -> dict[str, float]:
        """
        Solve the component for a given timestep

        This calculates the output state at the end (time_next) of the timestep

        Parameters
        ----------
        time_current
            Start of the timestep
        time_next
            End of the timestep

            This is the same as the start of the next timestep.
        input_state
            State at the start of the timestep

            Values are extracted from the model's state for the variables of interest.

        Returns
        -------
        State at the end of the timestep.

        This will be written to the model's state for the next timestep
        """
        if time_current > self.scale_year:
            return {"output": input_state.get("input") * self.scale_factor}
        else:
            return {"output": input_state.get("input")}


# %% [markdown]
# This Python class can be instantiated and called as normal:

# %%
scale_component = ScaleComponent(scale_factor=3, scale_year=2015)

# %%
res = scale_component.solve(2000, 1, {"input": 35.0})
assert res["output"] == 35.0

# %%
res = scale_component.solve(2050, 1, {"input": 35.0})
assert res["output"] == 35.0 * 3.0

# %% [markdown]
# But the magic happens when we build a new `PythonComponent`
# with this instance of the class.
# What happens here is that Rust creates a new struct which holds onto a `PyObject`
# of `scale_component`.
# This  `PythonComponent` struct implements the `Component` trait which in turn calls
# methods on `scale_component` within Rust.
# Since `PythonComponent` implements the `Component` trait it can be used within a
# model in the same way any other Rust-native `Component` is.
# This enables coupling between Rust and Python components.

# %%
component_in_rust = PythonComponent.build(scale_component)

# %% [markdown]
# It can be solved like any other Rust-native components

# %%
res = component_in_rust.solve(0, 1, {"input": 35.0})
assert res["output"] == 35.0

# %% [markdown]
# Below is an example using the component as part of a `Model`
# (which is implemented in Rust).

# %%
input_ts = Timeseries(
    np.asarray([1.0, 2.0, 3.0]),
    TimeAxis.from_values(np.asarray([1850.0, 2000.0, 2100.0])),
    "K",
    InterpolationStrategy.Previous,
)

time_axis = TimeAxis.from_bounds(np.arange(1750.0, 2100, 10.0))

model = (
    ModelBuilder()
    .with_py_component(component_in_rust)
    .with_time_axis(time_axis)
    .with_exogenous_variable("input", input_ts)
).build()


# %%
model.run()

# %%
result = model.timeseries()
result

# %%
plt.plot(time_axis.values(), result.get_timeseries_by_name("input").values())
plt.plot(time_axis.values(), result.get_timeseries_by_name("output").values())

# %% [markdown]
# The 1 time step delay is because the component propagates
# the input state from t0 to the output at the start of t1

# %%
assert result.get_timeseries_by_name("input").at_time(1800.0) == 1.0
assert result.get_timeseries_by_name("output").at_time(1800.0) == 1.0
assert result.get_timeseries_by_name("input").at_time(2050.0) == 2.0
assert result.get_timeseries_by_name("output").at_time(2050.0) == 2.0 * 3.0

# %%

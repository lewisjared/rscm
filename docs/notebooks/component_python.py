# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Components in Python
#
# This notebook demonstrates how to create climate model components in Python
# that integrate seamlessly with RSCM's Rust core.
#
# Python components can be coupled with Rust-native components in the same model,
# enabling rapid prototyping while maintaining performance for production code.
#
# ## Overview
#
# Create Python components by subclassing `rscm.component.Component` with
# declarative `Input`, `Output`, and `State` descriptors. This provides:
#
# - Auto-generated `definitions()` method
# - Type-safe input/output access
# - Support for scalar and grid-based (FourBox, Hemispheric) variables
#
# ## Related Resources
#
# - [Rust Components](../component_rust/): Creating components in Rust
# - [External Modules](../../external_modules/): Packaging components outside the RSCM repository
# - [Coupled Models](../coupled_model/): Combining multiple components
# - [Key Concepts](../../key_concepts/): Core RSCM architecture

# %%
import numpy as np

from rscm.component import Component, Input, Output, State
from rscm.core import (
    FourBoxSlice,
    InterpolationStrategy,
    ModelBuilder,
    PythonComponent,
    StateValue,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)

# %% [markdown]
# ## Creating a Typed Component
#
# To create a component, subclass `rscm.component.Component` and declare
# inputs/outputs using descriptors. The metaclass automatically generates:
#
# - `definitions()` method returning `list[RequirementDefinition]`
# - Inner `Inputs` class with typed fields for each input
# - Inner `Outputs` class for constructing return values
#
# `solve()` reads exogenous inputs and state variables with `at_start()`,
# the value at the start of the timestep.
# See [Timestep Semantics](../../key_concepts/#timestep-semantics) for when other accessors apply.
#
# ### Basic Example: Scalar Component


# %%
class ScaleComponent(Component):
    """Scale the input by a factor after a given year."""

    # The first argument is the variable name used in the model
    input_value = Input("input", unit="K")
    output_value = Output("output", unit="K")

    def __init__(self, scale_factor: float, scale_year: int):
        self.scale_factor = scale_factor
        self.scale_year = scale_year

    def solve(self, t_current: float, t_next: float, inputs: "ScaleComponent.Inputs"):
        """Scale the input once `t_current` passes `scale_year`."""
        current_input = inputs.input_value.at_start()

        if t_current > self.scale_year:
            result = current_input * self.scale_factor
        else:
            result = current_input

        return self.Outputs(output_value=result)


# %% [markdown]
# ### Component Instantiation
#
# Create an instance with your desired parameters:

# %%
scale_component = ScaleComponent(scale_factor=3, scale_year=2015)

# %% [markdown]
# ### Auto-generated Definitions
#
# The `definitions()` method is automatically generated from the descriptors.
# This tells the model what inputs the component requires and what outputs
# it produces:

# %%
for defn in scale_component.definitions():
    print(f"  {defn.requirement_type}: {defn.name} [{defn.unit}]")

# %% [markdown]
# ## Wrapping for Rust Integration
#
# To use a Python component in an RSCM model, wrap it with `PythonComponent.build()`.
# This creates a Rust struct that:
#
# - Holds a reference to your Python object
# - Implements the Rust `Component` trait
# - Handles data conversion between Python and Rust
#
# This enables seamless coupling between Rust and Python components.

# %%
wrapped = PythonComponent.build(scale_component)

# %% [markdown]
# ### Direct Component Invocation
#
# You can call `solve()` directly on the wrapped component for testing.
# `solve()` returns `dict[str, StateValue]`, and `.to_scalar()` extracts the float value:

# %%
collection = TimeseriesCollection()
collection.add_timeseries(
    "input",
    Timeseries(
        np.asarray([35.0, 35.0]),
        TimeAxis.from_bounds(np.asarray([2000.0, 2001.0, 2002.0])),
        "K",
        InterpolationStrategy.Previous,
    ),
)

result = wrapped.solve(2000, 2001, collection)

# The result is a dict of StateValue objects
print(f"Result: {result}")
print(f"Output value: {result['output'].to_scalar()}")

# %% [markdown]
# ## Using Components in a Model
#
# Components are typically used within a `Model` that handles:
#
# - Time stepping
# - Dependency resolution between components
# - State management across timesteps

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
    .with_py_component(wrapped)
    .with_time_axis(time_axis)
    .with_exogenous_variable("input", input_ts)
).build()
model.run()

timeseries = model.timeseries()

# %% tags=["remove_input"]
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for name in ["input", "output"]:
    plt.plot(
        time_axis.values(), timeseries.get_timeseries_by_name(name).values(), label=name
    )
plt.axvline(x=2015, color="gray", linestyle="--", alpha=0.5, label="scale_year")
plt.xlabel("Year")
plt.ylabel("Temperature (K)")
plt.legend()
plt.title("ScaleComponent: Input vs Output")
plt.show()

# %% [markdown]
# The output lags the input by one timestep, because the component reads its input at the start
# of a step and writes its output at the end.
# The model steps every 10 years, so the first step that starts after 2015 runs from 2020 to 2030.
# The tripled output therefore first appears in 2030.

# %% [markdown]
# ## State Variables
#
# A `State` is read at the start of each timestep and written at the end.
# It appears in both `Inputs` and `Outputs`.
# The model needs an initial value for every state variable.


# %%
class Accumulator(Component):
    """Accumulate emissions over time."""

    emissions = Input("Emissions|CO2", unit="GtC / yr")
    cumulative = State("Cumulative Emissions|CO2", unit="GtC")

    def solve(self, t_current: float, t_next: float, inputs):
        """Add this timestep's emissions to the running total."""
        dt = t_next - t_current
        total = inputs.cumulative.at_start() + inputs.emissions.at_start() * dt
        return self.Outputs(cumulative=total)


emissions_ts = Timeseries(
    np.asarray([10.0, 10.0]),
    TimeAxis.from_values(np.asarray([2000.0, 2010.0])),
    "GtC / yr",
    InterpolationStrategy.Previous,
)

state_model = (
    ModelBuilder()
    .with_py_component(PythonComponent.build(Accumulator()))
    .with_time_axis(TimeAxis.from_bounds(np.arange(2000.0, 2006.0)))
    .with_exogenous_variable("Emissions|CO2", emissions_ts)
    .with_initial_values({"Cumulative Emissions|CO2": 0.0})
).build()
state_model.run()

print(
    state_model.timeseries().get_timeseries_by_name("Cumulative Emissions|CO2").values()
)

# %% [markdown]
# ## Grid-Based Components
#
# RSCM supports spatially-resolved variables using two grid types:
#
# - **FourBox**: Northern Ocean, Northern Land, Southern Ocean, Southern Land
# - **Hemispheric**: Northern, Southern
#
# Specify the grid type in your `Input` or `Output` descriptor:


# %%
class RegionalComponent(Component):
    """Regional temperature response to a global forcing."""

    forcing = Input("Effective Radiative Forcing", unit="W/m^2")
    regional_temp = Output("Regional Temperature", unit="K", grid="FourBox")

    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity

    def solve(self, t_current: float, t_next: float, inputs):
        """Scale the forcing by a fixed factor per region."""
        erf = inputs.forcing.at_start()
        return self.Outputs(
            regional_temp=FourBoxSlice(
                northern_ocean=erf * self.sensitivity * 0.8,
                northern_land=erf * self.sensitivity * 1.2,
                southern_ocean=erf * self.sensitivity * 0.7,
                southern_land=erf * self.sensitivity * 1.1,
            )
        )


# %%
regional = RegionalComponent(sensitivity=0.5)

for defn in regional.definitions():
    grid = defn.grid_type
    print(f"  {defn.requirement_type.name}: {defn.name} [{defn.unit}] ({grid})")

# %% [markdown]
# Solving it once returns a FourBox `StateValue` for the regional output:

# %%
forcing = TimeseriesCollection()
forcing.add_timeseries(
    "Effective Radiative Forcing",
    Timeseries(
        np.asarray([2.0, 2.0]),
        TimeAxis.from_bounds(np.asarray([2000.0, 2001.0, 2002.0])),
        "W/m^2",
        InterpolationStrategy.Previous,
    ),
)
PythonComponent.build(regional).solve(2000, 2001, forcing)["Regional Temperature"]

# %% [markdown]
# ## StateValue: Understanding Component Outputs
#
# Component `solve()` methods return `dict[str, StateValue]`. The `StateValue`
# class wraps scalar or grid values:
#
# - `StateValue.scalar(float)`: single global value
# - `StateValue.four_box(FourBoxSlice)`: four regional values
# - `StateValue.hemispheric(HemisphericSlice)`: two hemispheric values
#
# Use accessor methods to extract values:

# %%
# Create different StateValue types
scalar_val = StateValue.scalar(15.0)
grid_val = StateValue.four_box(FourBoxSlice.uniform(10.0))

print(f"Scalar: {scalar_val}")
print(f"  is_scalar: {scalar_val.is_scalar()}")
print(f"  as_scalar: {scalar_val.as_scalar()}")
print()
print(f"Grid: {grid_val}")
print(f"  is_four_box: {grid_val.is_four_box()}")
print(f"  to_scalar (aggregated): {grid_val.to_scalar()}")

# %% [markdown]
# ## Input Access Patterns
#
# The typed `Inputs` class provides several ways to access input data:
#
# | Method | Description |
# |--------|-------------|
# | `inputs.field.at_start()` | Value at start of timestep (index N) |
# | `inputs.field.at_end()` | Value at end of timestep (index N+1) |
# | `inputs.field.previous` | Previous timestep value (index N-1) |
# | `inputs.field.at_offset(n)` | Value at relative offset |
# | `inputs.field.last_n(n)` | NumPy array of last n values |
#
# For grid inputs, use `at_start_all()` to get a `FourBoxSlice` or `HemisphericSlice`,
# or `at_start(region)` for an individual region.
#
# Read an upstream component's output for this timestep with `at_end()` or `at_end_all()`.
# Inputs arrive in the declared unit and grid when the model has a schema.
# See [Timestep Semantics](../../key_concepts/#timestep-semantics) for `at_start()` vs `at_end()` usage.

# %% [markdown]
# ## Saving Models
#
# `Model.to_toml()` saves a Python component with its class path and the output of `to_json()`.
# By default `to_json()` writes the instance attributes as JSON,
# and `from_json()` restores them without calling `__init__`.
# Override both when an attribute is not JSON-serialisable, such as a NumPy array.
# Define the class at module level so `Model.from_toml()` can import it.
# Loading imports the module the file names, so only load model files you trust.

# %% [markdown]
# ## Summary
#
# Key points for creating Python components:
#
# 1. Subclass `rscm.component.Component`
# 2. Declare variables with `Input()`, `Output()`, or `State()` descriptors
# 3. Implement `solve(t_current, t_next, inputs)` returning `self.Outputs(...)`
# 4. Wrap with `PythonComponent.build()` for model integration
# 5. Use `grid="FourBox"` or `grid="Hemispheric"` for spatial variables
# 6. Component outputs are `StateValue` objects, so use `.to_scalar()` for floats

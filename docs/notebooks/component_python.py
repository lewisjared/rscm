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
# This notebook shows how to write climate model components in Python.
# Python components couple with Rust components in the same model,
# so a component can be prototyped in Python before it is ported to Rust.
#
# ## Overview
#
# Create Python components by subclassing `rscm.component.Component`
# and declaring variables with `Input`, `Output` and `State` descriptors.
# This provides:
#
# - A generated `definitions()` method.
# - Typed input and output access.
# - Scalar and grid-based (four-box, hemispheric) variables.
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
# The metaclass of `rscm.component.Component` generates:
#
# - A `definitions()` method returning `list[VariableDefinition]`.
# - An inner `Inputs` class with a typed field for each input.
# - An inner `Outputs` class for constructing return values.
#
# `solve()` reads each input with `get()`,
# the value the model hands the component for the step.
# See [Step Semantics](../../key_concepts/#step-semantics) for how the model wiring sets it.
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
        current_input = inputs.input_value.get()

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
# `definitions()` is generated from the descriptors.
# It tells the model what the component reads and writes:

# %%
for defn in scale_component.definitions():
    print(f"  {defn.role}: {defn.name} [{defn.unit}]")

# %% [markdown]
# ## Wrapping for Rust Integration
#
# `ModelBuilder.with_component()` takes a Python component instance directly.
# It wraps the instance in a `PythonComponent`, a Rust struct that:
#
# - Holds a reference to your Python object
# - Implements the Rust `Component` trait
# - Handles data conversion between Python and Rust
#
# `PythonComponent()` applies the same wrapper by hand,
# which lets you call the component outside a model.

# %%
wrapped = PythonComponent(scale_component)

# %% [markdown]
# ### Direct Component Invocation
#
# Call `solve()` on the wrapped component to test it.
# Outside a model, each point input needs a value at `t_next`.
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
# A `Model` handles:
#
# - Stepping through the time axis.
# - Ordering components by their dependencies.
# - Carrying state between steps.

# %%
input_ts = Timeseries(
    np.asarray([1.0, 2.0, 3.0]),
    TimeAxis.from_values(np.asarray([1750.0, 2000.0, 2100.0])),
    "K",
    InterpolationStrategy.Previous,
)

time_axis = TimeAxis.from_bounds(np.arange(1750.0, 2100, 10.0))

model = (
    ModelBuilder()
    .with_component(wrapped)
    .with_time_axis(time_axis)
    .with_prescribed_variable("input", input_ts)
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
# The output follows the input with no lag.
# The component writes its output at the end of each step,
# and `get()` reads a prescribed point at the end of the step too.
# The model steps every 10 years, so the first step that starts after 2015 runs from 2020 to 2030.
# The tripled output therefore first appears in 2030.

# %% [markdown]
# ## State Variables
#
# A `State` is read at the start of each step and written at the end.
# It appears in both `Inputs` and `Outputs`.
# The model needs an initial value for every state variable.


# %%
class Accumulator(Component):
    """Accumulate emissions over time."""

    emissions = Input("Emissions|CO2", unit="GtC/yr")
    cumulative = State("Cumulative Emissions|CO2", unit="GtC")

    def solve(self, t_current: float, t_next: float, inputs):
        """Add this step's emissions to the running total."""
        dt = t_next - t_current
        total = inputs.cumulative.get() + inputs.emissions.get() * dt
        return self.Outputs(cumulative=total)


emissions_ts = Timeseries(
    np.asarray([10.0, 10.0]),
    TimeAxis.from_values(np.asarray([2000.0, 2010.0])),
    "GtC/yr",
    InterpolationStrategy.Previous,
)

state_model = (
    ModelBuilder()
    .with_component(Accumulator())
    .with_time_axis(TimeAxis.from_bounds(np.arange(2000.0, 2006.0)))
    .with_prescribed_variable("Emissions|CO2", emissions_ts)
    .with_initial_values({"Cumulative Emissions|CO2": 0.0})
).build()
state_model.run()

print(
    state_model.timeseries().get_timeseries_by_name("Cumulative Emissions|CO2").values()
)

# %% [markdown]
# ## Grid-Based Components
#
# RSCM supports spatially-resolved variables using two grids:
#
# - **Four-box**: Northern Ocean, Northern Land, Southern Ocean, Southern Land
# - **Hemispheric**: Northern, Southern
#
# Specify the grid in your `Input` or `Output` descriptor:


# %%
class RegionalComponent(Component):
    """Regional temperature response to a global forcing."""

    forcing = Input("Effective Radiative Forcing", unit="W/m^2")
    regional_temp = Output("Surface Air Temperature Change", unit="K", grid="four_box")

    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity

    def solve(self, t_current: float, t_next: float, inputs):
        """Scale the forcing by a fixed factor per region."""
        erf = inputs.forcing.get()
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
    grid = defn.grid
    print(f"  {defn.role.name}: {defn.name} [{defn.unit}] ({grid})")

# %% [markdown]
# Solving it once returns a four-box `StateValue` for the regional output:

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
PythonComponent(regional).solve(2000, 2001, forcing)["Surface Air Temperature Change"]

# %% [markdown]
# ## StateValue: Understanding Component Outputs
#
# A wrapped component's `solve()` returns `dict[str, StateValue]`.
# `StateValue` wraps a scalar or grid value:
#
# - `StateValue.scalar(float)`: one global value.
# - `StateValue.four_box(FourBoxSlice)`: four regional values.
# - `StateValue.hemispheric(HemisphericSlice)`: two hemispheric values.
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
# Each field of the typed `Inputs` class is a window over the input:
#
# | Method | Description |
# |--------|-------------|
# | `inputs.field.get()` | The value this component should use for the step |
# | `inputs.field.interval()` | A point's values at the start and end of the step |
# | `inputs.field.history(n)` | The value `n` steps before `get()`, or `None` |
# | `inputs.field.history_len()` | Number of values `history()` can read |
# | `inputs.field.get_time()` | Time of the value `get()` returns |
# | `inputs.field.interpolate(t)` | Value at time `t`, such as a fixed reference year |
#
# For a point, `get()` reads the end of the step, whether the input is prescribed or upstream,
# and the start of the step for own state and a lagged input.
# For grid inputs, `get()` and `interval()` return a `FourBoxSlice` or `HemisphericSlice`,
# and `region(i)` returns a scalar window for one region.
#
# Inputs arrive in the unit and grid the component declares,
# converted from the schema's where they differ.
# See [Step Semantics](../../key_concepts/#step-semantics) for how the model picks the step.

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
# To create a Python component:
#
# 1. Subclass `rscm.component.Component`.
# 2. Declare variables with `Input()`, `Output()` or `State()` descriptors.
# 3. Implement `solve(t_current, t_next, inputs)` returning `self.Outputs(...)`.
# 4. Pass an instance to `ModelBuilder.with_component()`.
# 5. Use `grid="four_box"` or `grid="hemispheric"` for spatial variables.
# 6. A wrapped component's `solve()` returns `StateValue` objects, so use `.to_scalar()` for floats.

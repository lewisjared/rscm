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
# - [Rust Components](component_rust.md): Creating components in Rust
# - [Coupled Models](coupled_model.py): Combining multiple components
# - [Key Concepts](../key_concepts.md): Core RSCM architecture

# %%
import matplotlib.pyplot as plt
import numpy as np

from rscm.component import Component, Input, Output
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
# ### Basic Example: Scalar Component


# %%
class ScaleComponent(Component):
    """
    Scale the input by a factor after a given year.

    This example demonstrates a simple scalar component that:
    - Reads a single input value
    - Applies conditional logic based on time
    - Returns a single output value
    """

    # Declare inputs and outputs using descriptors
    # The first argument is the variable name used in the model
    input_value = Input("input", unit="K")
    output_value = Output("output", unit="K")

    def __init__(self, scale_factor: float, scale_year: int):
        """
        Initialise the component.

        Parameters
        ----------
        scale_factor
            Multiplier applied after scale_year
        scale_year
            Year after which scaling is applied
        """
        self.scale_factor = scale_factor
        self.scale_year = scale_year

    def solve(self, t_current: float, t_next: float, inputs: "ScaleComponent.Inputs"):
        """
        Solve the component for a single timestep.

        Parameters
        ----------
        t_current
            Start time of the timestep
        t_next
            End time of the timestep
        inputs
            Typed inputs providing access to current and historical values.
            Access values via `inputs.<field_name>.current` or `.previous`.

        Returns
        -------
        ScaleComponent.Outputs
            Typed outputs for this timestep
        """
        # Access the current value using the typed interface
        current_input = inputs.input_value.current

        if t_current > self.scale_year:
            result = current_input * self.scale_factor
        else:
            result = current_input

        # Return using the auto-generated Outputs class
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
component_in_rust = PythonComponent.build(scale_component)

# %% [markdown]
# ### Direct Component Invocation
#
# You can call `solve()` directly on the wrapped component for testing.
# Note that `solve()` returns `dict[str, StateValue]` - use `.to_scalar()`
# to extract the float value:

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

result = component_in_rust.solve(2000, 2001, collection)

# The result is a dict of StateValue objects
print(f"Result: {result}")
print(f"Output value: {result['output'].to_scalar()}")

# %% [markdown]
# ## Using Components in a Model
#
# Components are typically used within a `Model` that handles:
# - Time stepping
# - Dependency resolution between components
# - State management across timesteps

# %%
# Create exogenous input data
input_ts = Timeseries(
    np.asarray([1.0, 2.0, 3.0]),
    TimeAxis.from_values(np.asarray([1850.0, 2000.0, 2100.0])),
    "K",
    InterpolationStrategy.Previous,
)

# Define the model time axis
time_axis = TimeAxis.from_bounds(np.arange(1750.0, 2100, 10.0))

# Build the model
model = (
    ModelBuilder()
    .with_py_component(component_in_rust)
    .with_time_axis(time_axis)
    .with_exogenous_variable("input", input_ts)
).build()

# %%
# Run the simulation
model.run()

# %%
# Access results
timeseries = model.timeseries()

plt.figure(figsize=(10, 5))
plt.plot(
    time_axis.values(),
    timeseries.get_timeseries_by_name("input").values(),
    label="input",
)
plt.plot(
    time_axis.values(),
    timeseries.get_timeseries_by_name("output").values(),
    label="output",
)
plt.xlabel("Year")
plt.ylabel("Temperature (K)")
plt.legend()
plt.title("ScaleComponent: Input vs Output")
plt.axvline(x=2015, color="gray", linestyle="--", alpha=0.5, label="scale_year")
plt.show()

# %% [markdown]
# Note the 1-timestep delay: the component reads input at t0 and writes
# output at t1. After 2015, the output is 3x the input.

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
    """
    Example component with FourBox grid output.

    Demonstrates how to work with spatially-resolved variables.
    """

    forcing = Input("Effective Radiative Forcing", unit="W/m^2")
    regional_temp = Output("Regional Temperature", unit="K", grid="FourBox")

    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity

    def solve(self, t_current: float, t_next: float, inputs):
        """Compute regional temperature response from forcing."""
        erf = inputs.forcing.current

        # Return FourBox output with different values per region
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

# Check the auto-generated definitions
for defn in regional.definitions():
    grid = defn.grid_type
    print(f"  {defn.requirement_type.name}: {defn.name} [{defn.unit}] ({grid})")

# %% [markdown]
# ## StateValue: Understanding Component Outputs
#
# Component `solve()` methods return `dict[str, StateValue]`. The `StateValue`
# class wraps scalar or grid values:
#
# - `StateValue.scalar(float)` - single global value
# - `StateValue.four_box(FourBoxSlice)` - four regional values
# - `StateValue.hemispheric(HemisphericSlice)` - two hemispheric values
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
# | `inputs.field.current` | Current timestep value |
# | `inputs.field.previous` | Previous timestep value |
# | `inputs.field.at_offset(n)` | Value at relative offset |
# | `inputs.field.last_n(n)` | NumPy array of last n values |
#
# For grid inputs, `current` returns a `FourBoxSlice` or `HemisphericSlice`.

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
# 6. Component outputs are `StateValue` objects - use `.to_scalar()` for floats

# %%

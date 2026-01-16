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
# # Model Debugging and Inspection
#
# This notebook demonstrates how to debug and inspect RSCM models.
# You will learn to inspect dependency graphs, trace data flow between
# components, and diagnose common integration issues.
#
# ## Overview
#
# When developing climate models, you often need to:
#
# - Understand how components are connected
# - Trace how data flows through the model
# - Identify why a model produces unexpected results
# - Validate that components are correctly coupled
#
# RSCM provides several inspection tools to help with these tasks.
#
# ## Related Resources
#
# - [Coupled Models](coupled_model.py): Building multi-component models
# - [State Serialisation](state_serialisation.py): Saving model snapshots
# - [Key Concepts](../key_concepts.md): Core RSCM architecture

# %%
import matplotlib.pyplot as plt
import numpy as np
import pydot
from IPython.display import Image, display

from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
)

# %% [markdown]
# ## Setting Up a Test Model
#
# First, we'll create a simple model to demonstrate inspection techniques.
# This model couples a carbon cycle component with a CO2 ERF (Effective
# Radiative Forcing) component.

# %%
# Time axis from 1750 to 2100
t_initial = 1750.0
t_final = 2100.0
time_axis = TimeAxis.from_values(np.arange(t_initial, t_final + 1, 10.0))

# Component parameters
co2_erf = CO2ERFBuilder.from_parameters(dict(erf_2xco2=4.0, conc_pi=280.0)).build()
carbon_cycle = CarbonCycleBuilder.from_parameters(
    dict(tau=20.3, conc_pi=280.0, alpha_temperature=0.0)
).build()

# Exogenous data
emissions = Timeseries(
    np.asarray([0.0, 5.0, 10.0, 8.0]),
    TimeAxis.from_bounds(np.asarray([t_initial, 1850.0, 1950.0, 2050.0, t_final])),
    "GtC / yr",
    InterpolationStrategy.Previous,
)

surface_temp = Timeseries(
    np.asarray([0.0]),
    TimeAxis.from_bounds(np.asarray([t_initial, t_final])),
    "K",
    InterpolationStrategy.Previous,
)

# Build the model
model = (
    ModelBuilder()
    .with_rust_component(carbon_cycle)
    .with_rust_component(co2_erf)
    .with_time_axis(time_axis)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_exogenous_variable("Surface Temperature", surface_temp)
    .with_initial_values(
        {
            "Cumulative Land Uptake": 0.0,
            "Cumulative Emissions|CO2": 0.0,
            "Atmospheric Concentration|CO2": 280.0,
        }
    )
).build()

# %% [markdown]
# ## Inspecting the Dependency Graph
#
# The dependency graph shows how components are connected. Each node
# represents a component, and edges show data dependencies (which
# variable flows from one component to another).
#
# Use `model.as_dot()` to get the graph in DOT format, which can be
# visualised using Graphviz.


# %%
def view_graph(model):
    """Display the model's dependency graph."""
    graph = pydot.graph_from_dot_data(model.as_dot())[0]
    display(Image(graph.create_png()))


view_graph(model)

# %% [markdown]
# ### Reading the Dependency Graph
#
# In the graph above:
#
# - **Nodes** are components (showing their type and key parameters)
# - **Edges** are labelled with the variable name that flows between components
# - The **root node (0)** is the starting point for breadth-first traversal
#
# This helps you understand:
#
# - Which component produces each variable
# - Which component consumes each variable
# - The execution order (BFS from root)

# %%
# You can also print the raw DOT format for programmatic analysis
print(model.as_dot())

# %% [markdown]
# ## Inspecting Model State
#
# The `TimeseriesCollection` contains all variables in the model.
# You can inspect what variables exist and their current values.

# %%
# Run the model partway through
model.step()
model.step()
model.step()

# Get the timeseries collection
collection = model.timeseries()

# List all variable names
print("Variables in model:")
for name in collection.names():
    print(f"  - {name}")

# %% [markdown]
# ### Accessing Variable Data
#
# Use `get_timeseries_by_name()` to access a specific variable's data.

# %%
# Get a specific variable
co2_conc = collection.get_timeseries_by_name("Atmospheric Concentration|CO2")

print(f"Units: {co2_conc.units}")
print(f"Latest value: {co2_conc.latest_value():.2f}")
print(f"Values shape: {co2_conc.values().shape}")

# %%
# Plot the variable
plt.figure(figsize=(10, 4))
values = co2_conc.values()
# The timeseries may have NaN for future timesteps
valid_indices = ~np.isnan(values)
times = time_axis.values()[valid_indices.flatten()]
valid_values = values[valid_indices]
plt.plot(times, valid_values)
plt.xlabel("Year")
plt.ylabel(f"CO2 Concentration ({co2_conc.units})")
plt.title("Atmospheric CO2 Concentration")
plt.grid(True)
plt.show()

# %% [markdown]
# ## Tracing Data Flow
#
# To understand how values propagate through the model, you can inspect
# multiple related variables together.

# %%
# Compare related variables
variables_to_trace = [
    "Emissions|CO2|Anthropogenic",
    "Cumulative Emissions|CO2",
    "Atmospheric Concentration|CO2",
    "Effective Radiative Forcing|CO2",
]

# Run model to completion first
model.run()
collection = model.timeseries()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, var_name in zip(axes, variables_to_trace):
    ts = collection.get_timeseries_by_name(var_name)
    if ts is not None:
        values = ts.values()
        valid = ~np.isnan(values)
        ax.plot(time_axis.values()[valid.flatten()], values[valid])
        ax.set_title(var_name)
        ax.set_xlabel("Year")
        ax.set_ylabel(ts.units)
        ax.grid(True)
    else:
        ax.text(
            0.5,
            0.5,
            f"Variable not found:\n{var_name}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step-by-Step Debugging
#
# For detailed debugging, step through the model one timestep at a time
# and inspect state after each step.

# %%
# Rebuild model for step-by-step debugging
model = (
    ModelBuilder()
    .with_rust_component(carbon_cycle)
    .with_rust_component(co2_erf)
    .with_time_axis(time_axis)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_exogenous_variable("Surface Temperature", surface_temp)
    .with_initial_values(
        {
            "Cumulative Land Uptake": 0.0,
            "Cumulative Emissions|CO2": 0.0,
            "Atmospheric Concentration|CO2": 280.0,
        }
    )
).build()

# Step through and record values
debug_log = []

print("Step-by-step execution:")
print("-" * 60)

for i in range(5):  # First 5 steps
    # Get current time bounds
    t_start, t_end = model.current_time_bounds()
    print(f"\nStep {i + 1}: {t_start:.0f} -> {t_end:.0f}")

    # Execute one step
    model.step()

    # Inspect key variables after step
    collection = model.timeseries()
    co2 = collection.get_timeseries_by_name("Atmospheric Concentration|CO2")
    erf = collection.get_timeseries_by_name("Effective Radiative Forcing|CO2")

    co2_val = co2.latest_value() if co2 else float("nan")
    erf_val = erf.latest_value() if erf else float("nan")

    print(f"  CO2 concentration: {co2_val:.2f} ppm")
    print(f"  CO2 ERF: {erf_val:.4f} W/m^2")

    debug_log.append(
        {
            "step": i + 1,
            "time": t_end,
            "co2": co2_val,
            "erf": erf_val,
        }
    )

# %% [markdown]
# ## Using Serialisation for Debugging
#
# Model state can be serialised to TOML format for inspection or to
# create checkpoints. This is useful for:
#
# - Sharing problematic model states with collaborators
# - Creating reproducible test cases
# - Inspecting the exact state at a specific point

# %%
# Serialise current state
toml_str = model.to_toml()

# Show first 2000 characters of the serialised state
print("Model state (TOML format, truncated):")
print("-" * 60)
print(toml_str[:2000])
print("...")

# %% [markdown]
# ### Restoring from Checkpoint
#
# You can restore a model from its serialised state and continue execution
# from that point.

# %%
from rscm.core import Model

# Restore model from TOML
restored_model = Model.from_toml(toml_str)

# Verify state matches
print(f"Original model time: {model.current_time()}")
print(f"Restored model time: {restored_model.current_time()}")

# Continue execution from checkpoint
restored_model.run()

# Compare final values
var_name = "Atmospheric Concentration|CO2"
orig_co2 = model.timeseries().get_timeseries_by_name(var_name)
rest_co2 = restored_model.timeseries().get_timeseries_by_name(var_name)

# The original model didn't run to completion, so run it now
model.run()
orig_co2 = model.timeseries().get_timeseries_by_name(var_name)

print(f"\nFinal CO2 (original): {orig_co2.latest_value():.2f}")
print(f"Final CO2 (restored): {rest_co2.latest_value():.2f}")

# %% [markdown]
# ## Diagnosing Common Issues
#
# ### Missing Variable Errors
#
# If a component requires a variable that isn't provided, the build step
# will fail with a descriptive error message listing available variables.

# %%
# Example: Missing exogenous variable
try:
    incomplete_model = (
        ModelBuilder()
        .with_rust_component(carbon_cycle)
        .with_time_axis(time_axis)
        # Missing: "Emissions|CO2|Anthropogenic"
        # Missing: "Surface Temperature"
        .with_initial_values(
            {
                "Cumulative Land Uptake": 0.0,
                "Cumulative Emissions|CO2": 0.0,
                "Atmospheric Concentration|CO2": 280.0,
            }
        )
    ).build()
except Exception as e:
    print("Build error (expected):")
    print(f"  {e}")

# %% [markdown]
# ### Missing Initial Values
#
# State variables require initial values. The error message will indicate
# which variable is missing.

# %%
# Example: Missing initial value
try:
    incomplete_model = (
        ModelBuilder()
        .with_rust_component(carbon_cycle)
        .with_time_axis(time_axis)
        .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
        .with_exogenous_variable("Surface Temperature", surface_temp)
        # Missing initial values
    ).build()
except Exception as e:
    print("Build error (expected):")
    print(f"  {e}")

# %% [markdown]
# ### Debugging Tips Summary
#
# 1. **Check the dependency graph** - Use `model.as_dot()` to visualise
#    component connections and verify the expected data flow
#
# 2. **List all variables** - Use `collection.names()` to see what
#    variables exist in the model
#
# 3. **Inspect specific values** - Use `get_timeseries_by_name()` to
#    access individual variable timeseries
#
# 4. **Step through execution** - Use `model.step()` for fine-grained
#    control and inspect state after each timestep
#
# 5. **Use serialisation** - Save model state with `to_toml()` to create
#    checkpoints or share problematic states
#
# 6. **Read error messages** - Build errors include details about missing
#    variables and available alternatives

# %% [markdown]
# ## Summary
#
# This notebook demonstrated how to:
#
# 1. Visualise the dependency graph with `as_dot()`
# 2. Access variables via `TimeseriesCollection`
# 3. Trace data flow between components
# 4. Step through execution for detailed debugging
# 5. Use serialisation for checkpoints and state inspection
# 6. Understand common error messages
#
# For more advanced debugging of Rust components, see the
# [Rust Development Tips](../developers/rust_tips.md) guide.

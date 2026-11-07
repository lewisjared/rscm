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
# This notebook shows how to debug and inspect RSCM models.
# You will learn to inspect dependency graphs, trace data flow between components,
# and diagnose common integration issues.
#
# ## Overview
#
# When developing climate models, you often need to:
#
# - Understand how components are connected.
# - Trace how data flows through the model.
# - Identify why a model produces unexpected results.
# - Validate that components are correctly coupled.
#
# ## Related Resources
#
# - [Coupled Models](../coupled_model/): Building multi-component models
# - [State Serialisation](../state_serialisation/): Saving model snapshots
# - [Key Concepts](../../key_concepts/): Core RSCM architecture

# %%
import json

import numpy as np

from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import (
    InterpolationStrategy,
    Model,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    VariableDefinition,
    VariableRole,
)

# %% [markdown]
# ## Setting Up a Test Model
#
# We reuse the carbon cycle and CO2 ERF model from [Coupled Models](../coupled_model/),
# with 10-year steps from 1750 to 2100 and CO2 emissions that change in 1850, 1950 and 2050.
# The setup defines the components `carbon_cycle` and `co2_erf`, the `time_axis`,
# the prescribed `emissions` and `surface_temp`, and a `builder` holding all of them.

# %% tags=["remove_input"]
t_initial = 1750.0
t_final = 2100.0
time_axis = TimeAxis.from_values(np.arange(t_initial, t_final + 1, 10.0))

co2_erf = CO2ERFBuilder.from_parameters(dict(erf_2xco2=4.0, conc_pi=280.0)).build()
carbon_cycle = CarbonCycleBuilder.from_parameters(
    dict(tau=20.3, conc_pi=280.0, alpha_temperature=0.0)
).build()

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

builder = (
    ModelBuilder()
    .with_component(carbon_cycle)
    .with_component(co2_erf)
    .with_time_axis(time_axis)
    .with_prescribed_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_prescribed_variable("Surface Temperature", surface_temp)
    .with_initial_values(
        {
            "Cumulative Land Uptake": 0.0,
            "Cumulative Emissions|CO2": 0.0,
            "Atmospheric Concentration|CO2": 280.0,
        }
    )
)

# %%
model = builder.build()

# %% tags=["remove_input"]
import matplotlib.pyplot as plt
import pydot
from IPython.display import SVG, display


def show_graph(model):
    """Render the component graph, labelling each node with its component name."""
    graph = pydot.graph_from_dot_data(model.as_dot())[0]
    for node in graph.get_nodes():
        label = node.get_label()
        if label:
            node.set_label(label.strip('"').split(" {")[0])
    display(SVG(graph.create_svg()))


# %% [markdown]
# ## Inspecting the Execution Graph
#
# The most useful debugging tool is `coupling()`,
# which shows what the builder decided about the model's coupling.
# It lists components in the order the runtime steps them.
#
# For each component, you can see:
#
# - **`<-` (green)**: input variables, tagged with their source:
#   - `(prescribed)`: supplied to the model rather than produced by a component.
#   - `(upstream from X)`: produced by component `X` earlier in the same step.
#   - `(lagged from X)`: component `X`'s value from the previous step.
#   - `(own_state)`: this component's own state from the previous step.
# - **`<>` (magenta)**: state variables (read the previous value, write a new value).
# - **`->` (blue)**: output variables produced by this component.
# - **Grid tags** (yellow): non-scalar grid types (for example `[FourBox]`).
# - **`point` or `average`**: whether the value holds at an instant or over a step.
#
# A "Lagged Edges" section then lists every edge that reads the previous step.
#
# The `format` parameter controls the output style:
#
# - `"rich"` (default): coloured terminal output.
# - `"plain"`: plain text with no ANSI codes.
# - `"json"`: structured JSON for programmatic use.

# %%
print(model.coupling())

# %% [markdown]
# The JSON format suits programmatic checks:

# %%
coupling = json.loads(model.coupling("json"))
for comp in coupling["components"]:
    n_in = len(comp.get("inputs", []))
    n_out = len(comp.get("outputs", []))
    n_state = len(comp.get("states", []))
    print(
        f"[{comp['order']}] {comp['name']}: {n_in} inputs, {n_out} outputs, {n_state} states"
    )

# %% [markdown]
# ### Structural Dependency Graph
#
# `model.as_dot()` returns the component graph in Graphviz DOT format.
# Each node is a component, and each edge is a variable that flows from one component to another.
# Rendered with Graphviz, it looks like this:

# %% tags=["remove_input"]
show_graph(model)

# %% [markdown]
# ## Inspecting Model State
#
# The `TimeseriesCollection` returned by `model.timeseries()` holds every variable in the model.
# After three steps, we list the variables and read one of them.

# %%
model.step()
model.step()
model.step()

collection = model.timeseries()
print(collection.names())

# %% [markdown]
# `get_timeseries_by_name()` returns one variable's `Timeseries`.
# Time points the model has not reached yet are NaN.

# %%
co2_conc = collection.get_timeseries_by_name("Atmospheric Concentration|CO2")

print(f"Units: {co2_conc.unit}")
print(f"Latest value: {co2_conc.latest_value():.2f}")
print(f"First values: {co2_conc.values()[:5]}")

# %% [markdown]
# ## Tracing Data Flow
#
# To see how values propagate through the model, run it to completion
# and plot related variables together.
# Here emissions accumulate, raise the CO2 concentration, and the concentration drives the forcing.

# %%
model.run()
collection = model.timeseries()

# %% tags=["remove_input"]
variables_to_trace = [
    "Emissions|CO2|Anthropogenic",
    "Cumulative Emissions|CO2",
    "Atmospheric Concentration|CO2",
    "Effective Radiative Forcing|CO2",
]

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, var_name in zip(axes.flat, variables_to_trace):
    ts = collection.get_timeseries_by_name(var_name)
    drawstyle = "steps-post" if var_name.startswith("Emissions") else "default"
    ax.plot(time_axis.values(), ts.values(), drawstyle=drawstyle)
    ax.set_title(var_name)
    ax.set_ylabel(ts.unit)
    ax.grid(True, alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("Year")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Step-by-Step Debugging
#
# For detailed debugging, step through a fresh model one step at a time
# and inspect state after each step.
# Here we advance to just before the emissions start in 1850 and print five steps.

# %%
model = builder.build()
while model.current_time() < 1840.0:
    model.step()

for _ in range(5):
    t_start, t_end = model.current_time_bounds()
    model.step()
    collection = model.timeseries()
    co2 = collection.get_timeseries_by_name("Atmospheric Concentration|CO2")
    erf = collection.get_timeseries_by_name("Effective Radiative Forcing|CO2")
    print(
        f"{t_start:.0f} -> {t_end:.0f}  CO2 {co2.latest_value():.2f} ppm"
        f"  ERF {erf.latest_value():.3f} W/m^2"
    )

# %% [markdown]
# ## Using Serialisation for Debugging
#
# `to_toml()` captures the full model state as a checkpoint.
# This is useful for sharing a problematic state with collaborators,
# creating reproducible test cases, or inspecting the exact state at one point.
# [State Serialisation](../state_serialisation/) describes the format.
#
# A model restored with `Model.from_toml()` continues from the same point
# and gives the same results as the original.

# %%
restored_model = Model.from_toml(model.to_toml())
assert restored_model.current_time() == model.current_time()

restored_model.run()
model.run()

var_name = "Atmospheric Concentration|CO2"
np.testing.assert_allclose(
    restored_model.timeseries().get_timeseries_by_name(var_name).values(),
    model.timeseries().get_timeseries_by_name(var_name).values(),
)

# %% [markdown]
# ## Diagnosing Common Issues
#
# ### Missing Prescribed Variables
#
# An input that no component produces needs prescribed data or an initial value.
# Without it the build fails, and the error names each missing variable and the components that read it.
# This model leaves out both `Emissions|CO2|Anthropogenic` and `Surface Temperature`.

# %%
incomplete_builder = (
    ModelBuilder()
    .with_component(carbon_cycle)
    .with_time_axis(time_axis)
    .with_initial_values(
        {
            "Cumulative Land Uptake": 0.0,
            "Cumulative Emissions|CO2": 0.0,
            "Atmospheric Concentration|CO2": 280.0,
        }
    )
)

try:
    incomplete_builder.build()
except ValueError as e:
    print(f"Build failed: {e}")

# %% [markdown]
# `with_missing_prescribed_allowed()` builds the model anyway.
# The missing inputs are NaN, and the NaN propagates into the outputs.

# %%
incomplete_model = incomplete_builder.with_missing_prescribed_allowed().build()
incomplete_model.run()

co2 = incomplete_model.timeseries().get_timeseries_by_name(
    "Atmospheric Concentration|CO2"
)
print(f"CO2 concentration: {co2.values()[:4]}")

# %% [markdown]
# ### Missing Initial Values
#
# State variables need initial values.
# Without them the build fails, and the error names the missing variable.

# %%
try:
    (
        ModelBuilder()
        .with_component(carbon_cycle)
        .with_time_axis(time_axis)
        .with_prescribed_variable("Emissions|CO2|Anthropogenic", emissions)
        .with_prescribed_variable("Surface Temperature", surface_temp)
    ).build()
except Exception as e:
    print(f"Build failed: {e}")

# %% [markdown]
# ### Runtime Errors
#
# If a component fails while solving, `step()` and `run()` raise a `RuntimeError`.
# The message names the component and the time interval that failed.
#
# The component below returns an output it never declared, which the model rejects.


# %%
class MisnamedOutput:
    """Declares one output but returns a misspelt name."""

    def definitions(self):
        """Declare the output."""
        return [VariableDefinition("Ocean Heat Uptake", "W/m^2", VariableRole.Output)]

    def solve(self, t_current, t_next, inputs):
        """Return a value under the wrong name."""
        return {"Ocean Heat Uptak": 0.5}


failing_model = (
    ModelBuilder().with_time_axis(time_axis).with_component(MisnamedOutput())
).build()

try:
    failing_model.run()
except RuntimeError as e:
    print(f"Run failed: {e}")

print(f"Model time after failure: {failing_model.current_time()}")

# %% [markdown]
# The time index does not advance past the failed step.
# Components that already ran in that step may have updated their internal state,
# so rebuild the model rather than retrying `step()` or `run()` on it.
#
# Stepping a finished model also raises a `RuntimeError`.

# %%
try:
    model.step()
except RuntimeError as e:
    print(f"Step failed: {e}")

# %% [markdown]
# ## Debugging Tips
#
# - Use `coupling()` to see execution order, where each input comes from, lags and grid types.
# - Check inputs marked `(prescribed)` if a model built with `with_missing_prescribed_allowed()`
#   gives NaN.
# - Use `step()` to stop at a point of interest and read values with `get_timeseries_by_name()`.
# - Save a problem state with `to_toml()` to reproduce or share it.
# - Read the errors: build errors explain missing initial values and schema or grid mismatches,
#   and runtime errors name the failing component and time interval.
#
# For debugging Rust components, see the [Rust Development Tips](../../developers/rust_tips/) guide.

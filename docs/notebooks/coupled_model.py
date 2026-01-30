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
# # Coupled Models
#
# This notebook demonstrates how to build coupled climate models by combining
# multiple components that exchange state information.
#
# ## Overview
#
# When modelling a complex system, you typically need multiple components
# coupled together. Each component models a particular aspect of the Earth System
# using a set of equations with known inputs and outputs. These inputs can come
# from other components or be prescribed as exogenous variables.
#
# ## Related Resources
#
# - [Components in Python](component_python.py): Creating Python components
# - [Components in Rust](component_rust.md): Creating Rust components
# - [Key Concepts](../key_concepts.md): Core RSCM architecture
# - [State Serialisation](state_serialisation.py): Saving and loading model state


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
from IPython.display import Image, display

from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)
from rscm.two_layer import TwoLayerBuilder

# %% [markdown]
# ## Building a Model
#
# The `ModelBuilder` class constructs coupled models from components.
# It captures the model's requirements (components, time axis, exogenous variables)
# before creating a concrete model. This pattern makes it easy to share and
# extend model configurations.

# %%
model_builder = ModelBuilder()

# %% [markdown]
# ### Time axis
#
# The time axis is a critical aspect of a model as it defines the timesteps
# on which the model is solved.
# Components only exchange information at the end of a time step.
#
# The size of the timesteps don't need to be the same across the time axis.
# This enables the use of longer timesteps during preindustrial spinup
# and then decreasing as the level of anthropogenic emissions increases.

# %%
t_initial = 1750.0
t_final = 2100.0

# %%
time_axis = TimeAxis.from_values(
    np.concat([np.arange(t_initial, 2015.0, 5), np.arange(2015.0, t_final + 1, 1.0)])
)
time_axis

# %%
model_builder.with_time_axis(time_axis)

# %% [markdown]
# ### Components
#
# A model consists of one or more components.
#
# Components can be implemented in either Rust (for performance)
# or Python (for rapid prototyping).
# See [Components in Rust](component_rust.md) and
# [Components in Python](component_python.py) for implementation details.
#
# Below we create a model with a basic carbon cycle and CO2 ERF calculation.
# The CO2 ERF component calculates the radiative forcing based on CO2
# concentrations from the Carbon Cycle component.

# %%
# Component parameters
tau = 20.3
conc_pi = 280.0
erf_2xco2 = 4.0
alpha_temperature = 0.0  # No temperature feedback

# %%
co2_erf_component = CO2ERFBuilder.from_parameters(
    dict(erf_2xco2=erf_2xco2, conc_pi=conc_pi)
).build()
carbon_cycle_component = CarbonCycleBuilder.from_parameters(
    dict(tau=tau, conc_pi=conc_pi, alpha_temperature=alpha_temperature)
).build()

# %%
model_builder.with_rust_component(carbon_cycle_component).with_rust_component(
    co2_erf_component
)

# %% [markdown]
# ## Exogenous Variables
#
# Variables that are not produced by any component must be provided as
# exogenous (external) data.
# Each exogenous variable is a `Timeseries` with values, time axis, units,
# and interpolation strategy.

# %%
# model the CO2 emissions as a heaviside function at step_year
step_year = 1850.0
step_size = 10.0

emissions = Timeseries(
    np.asarray([0.0, 0.0, step_size, step_size]),
    TimeAxis.from_bounds(
        np.asarray(
            [
                t_initial,
                (t_initial + step_year) / 2.0,
                step_year,
                step_year + 50.0,
                t_final,
            ]
        )
    ),
    "GtC / yr",
    InterpolationStrategy.Previous,
)

# %%

# %%
# Raw values
plt.plot(emissions.time_axis.values(), emissions.values())

# %%
# Interpolation onto the model time_axis happens automatically during solve()

# %%
surface_temp = Timeseries(
    np.asarray([0.42]),
    TimeAxis.from_bounds(np.asarray([t_initial, t_final])),
    "K",
    InterpolationStrategy.Previous,
)

# %%
model_builder.with_exogenous_variable(
    "Surface Temperature", surface_temp
).with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)

# %% [markdown]
# ## Initial Values
#
# Components that track state (using `#[states(...)]` in Rust or `State()`
# in Python) require initial values.
# These values are used at the first timestep;
# subsequent timesteps use the state from the previous step.

# %%
initial_state = {
    "Cumulative Land Uptake": 0.0,
    "Cumulative Emissions|CO2": 0.0,
    "Atmospheric Concentration|CO2": 300.0,
}

model_builder.with_initial_values(initial_state)

# %% [markdown]
# ## Build the Model
#
# Building the model generates a directed graph of component relationships.
# The model uses this graph to determine the solve order via breadth-first search.
# If any required information is missing, the build step will fail with an
# exception explaining what is needed.
#

# %%
model = model_builder.build()


# %% [markdown]
# These steps can also be chained together as shown below:

# %%
model = (
    ModelBuilder()
    .with_rust_component(carbon_cycle_component)
    .with_rust_component(co2_erf_component)
    .with_time_axis(time_axis)
    .with_exogenous_variable("Surface Temperature", surface_temp)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_initial_values(initial_state)
).build()


# %% [markdown]
# The graph can be visualised with each node representing a component and the edges
# describe the flow of state through the model.
# This graph is solved using a breadth first search starting from the "0" node.


# %%
# This requires graphviz to be installed
def view_pydot(pdot):
    """
    Show a dot graph inside a notebook.

    Parameters
    ----------
    pdot
        Pydot graph object to display.
    """
    plt = Image(pdot.create_png())
    display(plt)


graph = pydot.graph_from_dot_data(model.as_dot())[0]
view_pydot(graph)

# %% [markdown]
# ## Run the model
#
# Once we have a concrete model we can solve it.
# You can either step through the model step by step or run for all timesteps at once

# %%
model.current_time_bounds()

# %%
model.step()
model.current_time_bounds()

# %% [markdown]
# The results from the run can be extracted using `timeseries` and then converted to a
# pandas DataFrame for easier manipulation and plotting.


# %%
def as_dataframe(
    timeseries_collection: TimeseriesCollection, time_axis: TimeAxis
) -> pd.DataFrame:
    """
    Convert a collection of timeseries to a pandas DataFrame with multi-index.

    Parameters
    ----------
    timeseries_collection
        RSCM timeseries collection
    time_axis
        Time axis for column labels

    Returns
    -------
        DataFrame with (variable, unit) multi-index and time columns
    """
    data = []
    index_tuples = []
    for name in timeseries_collection.names():
        ts = timeseries_collection.get_timeseries_by_name(name)

        index_tuples.append((name, ts.units))
        data.append(ts.values())

    return pd.DataFrame(
        data,
        columns=time_axis.values(),
        index=pd.MultiIndex.from_tuples(index_tuples, names=["variable", "unit"]),
    )


results = as_dataframe(model.timeseries(), time_axis)
results

# %%
# Filter out cumulative variables and plot
filtered = results.loc[
    ~results.index.get_level_values("variable").str.startswith("Cumulative")
]
filtered.T.plot(figsize=(10, 5))
plt.xlabel("Year")
plt.ylabel("Value")
plt.legend(filtered.index.get_level_values("variable"))
plt.show()

# %%
model.run()

# %%
results = as_dataframe(model.timeseries(), time_axis)
filtered = results.loc[
    ~results.index.get_level_values("variable").str.startswith("Cumulative")
]
filtered.T.plot(figsize=(10, 5))
plt.xlabel("Year")
plt.ylabel("Value")
plt.legend(filtered.index.get_level_values("variable"))
plt.show()

# %% [markdown]
# ## Complex Coupled Model with Feedback
#
# The example above uses a prescribed surface temperature.
# In reality, temperature responds to radiative forcing, and temperature changes
# can affect carbon cycle processes (e.g., soil respiration, ocean solubility).
#
# Below we build a more complex coupled model that demonstrates these feedbacks:
#
# ```
# Emissions → Carbon Cycle → CO2 Concentration → CO2 ERF → Two-Layer → Temperature
#                  ↑                                                          │
#                  └──────────────────── feedback ────────────────────────────┘
# ```
#
# This creates a feedback loop where:
#
# 1. **Carbon Cycle**: Converts emissions to atmospheric CO2 concentration
# 2. **CO2 ERF**: Calculates radiative forcing from CO2 concentration
# 3. **Two-Layer Model**: Computes surface temperature from radiative forcing
# 4. **Temperature Feedback**: Surface temperature affects carbon cycle lifetime
#    (warmer temperatures reduce CO2 uptake efficiency)

# %%
# Create components for the feedback-coupled model
# Enable temperature feedback in the carbon cycle
carbon_cycle_feedback = CarbonCycleBuilder.from_parameters(
    dict(
        tau=25.0,  # Baseline atmospheric lifetime (years)
        conc_pi=278.0,  # Pre-industrial CO2 (ppm)
        alpha_temperature=0.05,  # Temperature sensitivity (1/K)
    )
).build()

co2_erf = CO2ERFBuilder.from_parameters(
    dict(
        erf_2xco2=3.7,  # Forcing from CO2 doubling (W/m²)
        conc_pi=278.0,  # Pre-industrial CO2 (ppm)
    )
).build()

two_layer = TwoLayerBuilder.from_parameters(
    dict(
        lambda0=1.1,  # Climate feedback parameter (W/m²/K)
        a=0.0,  # Nonlinear feedback coefficient
        efficacy=1.3,  # Ocean heat uptake efficacy
        eta=0.7,  # Heat exchange coefficient (W/m²/K)
        heat_capacity_surface=8.0,  # Surface heat capacity (W yr/m²/K)
        heat_capacity_deep=100.0,  # Deep ocean heat capacity (W yr/m²/K)
    )
).build()

# %% [markdown]
# ### Emissions Scenario
#
# We create a more realistic emissions scenario with:
#
# - Pre-industrial period (1750-1850): near-zero emissions
# - Historical growth (1850-2020): exponential increase
# - Future scenario (2020-2100): peak and decline (SSP1-like pathway)

# %%
# Create emissions scenario
years = np.array([1750, 1850, 1950, 2000, 2020, 2050, 2100])
emission_values = np.array([0.0, 0.5, 3.0, 7.0, 10.0, 5.0, 1.0])  # GtC/yr

emissions_scenario = Timeseries(
    emission_values,
    TimeAxis.from_bounds(
        np.concatenate([years, [2101]])  # bounds need n+1 values
    ),
    "GtC / yr",
    InterpolationStrategy.Linear,
)

# %%
# Plot the emissions scenario
plt.figure(figsize=(10, 4))
# Interpolate to yearly values for plotting
plot_years = np.arange(1750, 2101)
plt.plot(years, emission_values, "o", markersize=8, label="Scenario points")
plt.fill_between(years, emission_values, alpha=0.3, label="Emissions pathway")
plt.xlabel("Year")
plt.ylabel("Emissions (GtC/yr)")
plt.title("CO2 Emissions Scenario")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### Build the Feedback-Coupled Model
#
# Note that we don't need to provide `Surface Temperature` as an exogenous variable
# because the Two-Layer model produces it. The model builder automatically resolves
# the dependency graph.

# %%
feedback_model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    # Components
    .with_rust_component(carbon_cycle_feedback)
    .with_rust_component(co2_erf)
    .with_rust_component(two_layer)
    # Exogenous input - only emissions needed
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions_scenario)
    # Initial state
    .with_initial_values(
        {
            "Cumulative Land Uptake": 0.0,
            "Cumulative Emissions|CO2": 0.0,
            "Atmospheric Concentration|CO2": 278.0,
            "Surface Temperature": 0.0,  # Anomaly from pre-industrial
        }
    )
).build()

# %% [markdown]
# ### Visualise the Dependency Graph
#
# The graph now shows the feedback structure with three components and their
# data flow connections.

# %%
graph = pydot.graph_from_dot_data(feedback_model.as_dot())[0]
view_pydot(graph)

# %% [markdown]
# The dependency graph shows:
#
# - **Node 0**: Root/entry point
# - **CarbonCycle**: Takes emissions and temperature, outputs CO2 concentration
# - **CO2ERF**: Takes CO2 concentration, outputs radiative forcing
# - **TwoLayer**: Takes radiative forcing, outputs surface temperature
#
# The edges show which variables flow between components.

# %%
# Run the feedback-coupled model
feedback_model.run()

# %%
# Extract and plot results
feedback_results = as_dataframe(feedback_model.timeseries(), time_axis)

# Select key variables for plotting
key_vars = [
    "Emissions|CO2|Anthropogenic",
    "Atmospheric Concentration|CO2",
    "Effective Radiative Forcing|CO2",
    "Surface Temperature",
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, var in zip(axes, key_vars):
    if var in feedback_results.index.get_level_values("variable"):
        data = feedback_results.loc[var]
        unit = data.index.get_level_values("unit")[0]
        ax.plot(data.columns, data.values.flatten())
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{var}\n({unit})")
        ax.set_title(var)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Comparing With and Without Temperature Feedback
#
# To see the effect of temperature feedback on the carbon cycle, let's run
# the model with feedback disabled and compare.

# %%
# Model without temperature feedback
carbon_cycle_no_feedback = CarbonCycleBuilder.from_parameters(
    dict(
        tau=25.0,
        conc_pi=278.0,
        alpha_temperature=0.0,  # No feedback
    )
).build()

no_feedback_model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_rust_component(carbon_cycle_no_feedback)
    .with_rust_component(co2_erf)
    .with_rust_component(two_layer)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions_scenario)
    .with_initial_values(
        {
            "Cumulative Land Uptake": 0.0,
            "Cumulative Emissions|CO2": 0.0,
            "Atmospheric Concentration|CO2": 278.0,
            "Surface Temperature": 0.0,
        }
    )
).build()

no_feedback_model.run()
no_feedback_results = as_dataframe(no_feedback_model.timeseries(), time_axis)

# %%
# Compare temperature response
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# CO2 concentration comparison
ax = axes[0]
ax.plot(
    feedback_results.columns,
    feedback_results.loc["Atmospheric Concentration|CO2"].values.flatten(),
    label="With feedback",
)
ax.plot(
    no_feedback_results.columns,
    no_feedback_results.loc["Atmospheric Concentration|CO2"].values.flatten(),
    "--",
    label="Without feedback",
)
ax.set_xlabel("Year")
ax.set_ylabel("CO2 Concentration (ppm)")
ax.set_title("CO2 Concentration")
ax.legend()
ax.grid(True, alpha=0.3)

# Temperature comparison
ax = axes[1]
ax.plot(
    feedback_results.columns,
    feedback_results.loc["Surface Temperature"].values.flatten(),
    label="With feedback",
)
ax.plot(
    no_feedback_results.columns,
    no_feedback_results.loc["Surface Temperature"].values.flatten(),
    "--",
    label="Without feedback",
)
ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (K)")
ax.set_title("Surface Temperature")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# The temperature feedback amplifies warming by reducing the efficiency of
# natural carbon sinks at higher temperatures. This positive feedback loop
# results in:
#
# - Higher CO2 concentrations
# - Greater radiative forcing
# - Higher surface temperatures
#
# This demonstrates why coupled climate-carbon models are important for
# projecting future climate change.

# %% [markdown]
# ## Summary
#
# This notebook demonstrated how to:
#
# 1. Create a `ModelBuilder` with a time axis
# 2. Add Rust components with parameters
# 3. Provide exogenous variables as `Timeseries` objects
# 4. Set initial values for state variables
# 5. Build and run the model
# 6. Extract results as `TimeseriesCollection`
# 7. Build complex feedback-coupled models with multiple components
# 8. Visualise dependency graphs showing component relationships
# 9. Compare model behaviour with and without feedback processes
#
# For more details on creating custom components, see the
# [Components in Python](component_python.py) and
# [Components in Rust](component_rust.md) guides.

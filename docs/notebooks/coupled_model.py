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
#
# For more details on creating custom components, see the
# [Components in Python](component_python.py) and
# [Components in Rust](component_rust.md) guides.

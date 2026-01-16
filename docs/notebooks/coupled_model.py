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
# # Coupled models
#
# When modelling a complex system
# often you need more than one component coupled together.
# Each component will model a particular aspect of the Earth System
# using a set of equations with some known inputs and outputs.
# These inputs can come from other components or prescribed as
# exogenous variables if not components produce that variable as output.
#
# Typically, a model consists of more than one component.
# These components will interact with each other by passing state amongst them.


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
import scmdata
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
# ## Building the model
#
# The `ModelBuilder` class can be used to build up a coupled model.
# This builder class captures the requirements for the model including components,
# time axis and exogenous variables before creating a concrete model.
# This makes it easier to share model setups and extend them if needed.

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
# Below we create a model that consists of a basic carbon cycle
# and a CO2 ERF calculation.
# The CO2 ERF component calculates the radiative forcing of CO2
# modified CO2 concentrations from the Carbon Cycle component.

# %%
# Some parameters for the components
# We don't have a cononical way of dealing with parameters yet.
# Ideally they should be decoupled from describing how the components interact
tau = 20.3
conc_pi = 280.0
erf_2xco2 = 4.0
alpha_temperature = 0.0  # Have no temperature feedback

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
# ## Exogenous variables
#

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
# Post interpolation onto the time_axis of the model
# TODO

# %%
surface_temp = Timeseries(
    np.asarray([0.42]),
    TimeAxis.from_bounds(np.asarray([t_initial, t_final])),
    "K",
    InterpolationStrategy.Previous,
)

# %%
# TODO: Figure out how to make a nested timeseries.
#  i.e. Emissions|CO2 = Emissions|CO2|Anthropogenic + Emissions|CO2|LULUCF
model_builder.with_exogenous_variable(
    "Surface Temperature", surface_temp
).with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)

# %% [markdown]
# ## Initial Values
#
# The Carbon Cycle component is a coupled set of ordinary differential equations
# that are solved as an initial value problem.
# In order to solve these equations the state ot t=0 must be known.
# For later time steps the state from the previous time step is used
#
# This step might be refactored to be included in the component builder.

# %%
initial_state = {
    "Cumulative Land Uptake": 0.0,
    "Cumulative Emissions|CO2": 0.0,
    "Atmospheric Concentration|CO2": 300.0,
}

model_builder.with_initial_values(initial_state)

# %% [markdown]
# ## Build the model
#
# We can now build the model.
# This step generates a directed graph of the relationships between the components.
# This graph is then used by the model to determine the order in which
# it should solve the components.
# If any required information is missing then this step will fail
# with an exceptions which explains what it required.
#
# Often additional exogenous data is required.
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
# scmdata object for easier plotting


# %%
class RSCMRun(scmdata.run.BaseScmRun):  # type: ignore
    """ScmRun object with minimal required metadata"""

    required_cols = ("variable", "unit")


def as_scmrun(timeseries_collection: TimeseriesCollection) -> RSCMRun:
    """Convert a collection of timeseries to a scmdata object"""
    data = []
    columns = []
    for name in timeseries_collection.names():
        ts = timeseries_collection.get_timeseries_by_name(name)

        columns.append({"variable": name, "unit": ts.units})
        data.append(ts.values())

    as_dataframe = pd.DataFrame(
        data,
        columns=time_axis.values(),
        index=pd.MultiIndex.from_frame(pd.DataFrame(columns)),
    )
    return RSCMRun(as_dataframe)


results = as_scmrun(model.timeseries())
results

# %%
results.filter(variable="Cumulative *", keep=False).line_plot(hue="variable")

# %%
model.run()

# %%
as_scmrun(model.timeseries()).filter(variable="Cumulative *", keep=False).line_plot(
    hue="variable"
)

# %% [markdown]
# ## Features to add
#
# * Hierachy of timeseries / n box timeseries
# * Better hinting of required parameters
# * Parameter handling

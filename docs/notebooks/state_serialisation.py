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
# # Model state serialisation
#
# It can be useful to capture the current state of a model and serialise it for later use.
#
# `Model.to_toml()` produces a human-readable TOML description of a model,
# including its components, parameters, timeseries and internal state.
# `Model.from_toml()` recreates the model from that string.
#
# ## Related Resources
#
# - [Coupled Models](coupled_model.py): Building multi-component models
# - [Scenario Pipeline](scenario_pipeline.py): Branching scenarios from a historical run
# - [Model Debugging and Inspection](debugging_inspection.py): Using checkpoints to debug

# %%
import tempfile
from pathlib import Path

import numpy as np

from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import (
    InterpolationStrategy,
    Model,
    ModelBuilder,
    TimeAxis,
    Timeseries,
)

# %% [markdown]
# ## Create a simple coupled model
#
# We first create a simple coupled model with a carbon cycle and CO2 ERF component.

# %%
t_initial = 1750
t_final = 2100

time_axis = TimeAxis.from_values(
    np.concatenate(
        [
            np.arange(t_initial, 2015.0, 5),
            np.arange(2015.0, t_final + 1, 1.0),
        ]
    )
)

# Carbon cycle parameters
tau = 20.3
conc_pi = 280.0
erf_2xco2 = 4.0
alpha_temperature = 0.0  # No temperature feedback

co2_erf_component = CO2ERFBuilder.from_parameters(
    dict(erf_2xco2=erf_2xco2, conc_pi=conc_pi)
).build()
carbon_cycle_component = CarbonCycleBuilder.from_parameters(
    dict(tau=tau, conc_pi=conc_pi, alpha_temperature=alpha_temperature)
).build()

initial_state = {
    "Cumulative Land Uptake": 0.0,
    "Cumulative Emissions|CO2": 0.0,
    "Atmospheric Concentration|CO2": 300.0,
}

# Create step emissions scenario
step_year = 1850.0
step_size = 10.0

emissions_bounds = np.asarray(
    [t_initial, (t_initial + step_year) / 2.0, step_year, step_year + 50.0, t_final]
)
emissions = Timeseries(
    np.asarray([0.0, 0.0, step_size, step_size]),
    TimeAxis.from_bounds(emissions_bounds),
    "GtC / yr",
    InterpolationStrategy.Previous,
)

surface_temp = Timeseries(
    np.asarray([0.42]),
    TimeAxis.from_bounds(np.asarray([float(t_initial), float(t_final)])),
    "K",
    InterpolationStrategy.Previous,
)

model = (
    ModelBuilder()
    .with_rust_component(carbon_cycle_component)
    .with_rust_component(co2_erf_component)
    .with_time_axis(time_axis)
    .with_exogenous_variable("Surface Temperature", surface_temp)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_initial_values(initial_state)
).build()

# %%
# We can then step the model forward in time
model.step()
model.step()

# %% [markdown]
# We can log the state of a model using TOML.
#
# This format is human-readable and can be used to store the state of a model to disk.
# It contains these sections:
#
# - `time_index`: the current position on the time axis.
# - `components`: the component graph, with each component's type and parameters.
# - `collection`: every timeseries, with values solved so far and NaN for later time points.
# - `time_axis`: the model's time bounds.
# - `component_states`: internal state that components carry between steps.

# %%
serialised_model = model.to_toml()
print("\n".join(serialised_model.splitlines()[:40]))
print("...")

# %% [markdown]
# ## Recreating the model state
#
# A new model can be initialised using the TOML representation.
# The new model will have the same state and components as the original model.

# %%
new_model = Model.from_toml(serialised_model)
print(new_model.as_dot())

# %% [markdown]
# It will also be on the same timestep as the original model.

# %%
assert new_model.current_time() == model.current_time()

# %% [markdown]
# The newly created model can then be run as normal.
# It produces the same results as the original model run to completion.

# %%
new_model.run()
model.run()

variable = "Atmospheric Concentration|CO2"
np.testing.assert_allclose(
    new_model.timeseries().get_timeseries_by_name(variable).values(),
    model.timeseries().get_timeseries_by_name(variable).values(),
)
print("Restored and original runs match")

# %% [markdown]
# ## Checkpoints on disk
#
# Write the TOML string to a file to keep a checkpoint between sessions or share it with collaborators.

# %%
checkpoint_model = (
    ModelBuilder()
    .with_rust_component(carbon_cycle_component)
    .with_rust_component(co2_erf_component)
    .with_time_axis(time_axis)
    .with_exogenous_variable("Surface Temperature", surface_temp)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_initial_values(initial_state)
).build()

while checkpoint_model.current_time() < 1900.0:
    checkpoint_model.step()

with tempfile.TemporaryDirectory() as tmp:
    checkpoint = Path(tmp) / "model_1900.toml"
    checkpoint.write_text(checkpoint_model.to_toml())

    resumed = Model.from_toml(checkpoint.read_text())

print(f"Resumed at {resumed.current_time()}")
resumed.run()
print(f"Finished at {resumed.current_time()}")

# %% [markdown]
# ## Limitations
#
# - A restored model keeps its original time axis and exogenous data.
#   There is no API yet to change these on a restored model.
#   To run new scenarios from a branch point,
#   see the [Scenario Pipeline](scenario_pipeline.py) tutorial.
# - Models containing Python components cannot be serialised yet.
# - The TOML records no format version.
#   Treat checkpoints as tied to the RSCM version that wrote them.

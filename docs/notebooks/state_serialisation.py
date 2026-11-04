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
# - [Coupled Models](../coupled_model/): Building multi-component models
# - [Scenario Pipeline](../scenario_pipeline/): Branching scenarios from a historical run
# - [Model Debugging and Inspection](../debugging_inspection/): Using checkpoints to debug

# %%
import tempfile
import tomllib
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
# We reuse the carbon cycle and CO2 ERF model from [Coupled Models](../coupled_model/),
# with a step in CO2 emissions in 1850.
# Its `ModelBuilder` is in `builder`.

# %% tags=["remove_input"]
t_initial = 1750.0
t_final = 2100.0

time_axis = TimeAxis.from_values(
    np.concat([np.arange(t_initial, 2015.0, 5), np.arange(2015.0, t_final + 1, 1.0)])
)

emissions = Timeseries(
    np.asarray([0.0, 0.0, 10.0, 10.0]),
    TimeAxis.from_bounds(np.asarray([t_initial, 1800.0, 1850.0, 1900.0, t_final])),
    "GtC / yr",
    InterpolationStrategy.Previous,
)
surface_temp = Timeseries(
    np.asarray([0.42]),
    TimeAxis.from_bounds(np.asarray([t_initial, t_final])),
    "K",
    InterpolationStrategy.Previous,
)

builder = (
    ModelBuilder()
    .with_rust_component(
        CarbonCycleBuilder.from_parameters(
            dict(tau=20.3, conc_pi=280.0, alpha_temperature=0.0)
        ).build()
    )
    .with_rust_component(
        CO2ERFBuilder.from_parameters(dict(erf_2xco2=4.0, conc_pi=280.0)).build()
    )
    .with_time_axis(time_axis)
    .with_exogenous_variable("Surface Temperature", surface_temp)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_initial_values(
        {
            "Cumulative Land Uptake": 0.0,
            "Cumulative Emissions|CO2": 0.0,
            "Atmospheric Concentration|CO2": 300.0,
        }
    )
)

# %% [markdown]
# We build the model and step it forward twice.

# %%
model = builder.build()
model.step()
model.step()

# %% [markdown]
# ## Serialising to TOML
#
# The TOML is human-readable and can be stored on disk.
# It contains these sections:
#
# - `time_index`: the current position on the time axis.
# - `components`: the component graph, with each component's type and parameters.
# - `collection`: every timeseries, with values solved so far and NaN for later time points.
# - `time_axis`: the model's time bounds.
# - `component_states`: internal state that components carry between steps.
#
# Most of the string is timeseries values.
# Parsing it with `tomllib` picks out the parts above.

# %%
serialised_model = model.to_toml()
state = tomllib.loads(serialised_model)
print(f"{len(serialised_model.splitlines())} lines of TOML")

co2 = state["collection"]["timeseries"][0]
print("time_index:", state["time_index"])
print("components:", [node["type"] for node in state["components"]["nodes"]])
print("CarbonCycle parameters:", state["components"]["nodes"][1]["parameters"])
print(f"{co2['name']}:", co2["data"]["Scalar"]["values"]["data"][:4], "...")

# %% [markdown]
# ## Recreating the model state
#
# A new model can be initialised from the TOML representation.
# It has the same components and state as the original model,
# and is on the same timestep.

# %%
new_model = Model.from_toml(serialised_model)
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
# `builder.build()` can be called again to get a fresh model from the same configuration.

# %%
checkpoint_model = builder.build()
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
#   see the [Scenario Pipeline](../scenario_pipeline/) tutorial.
# - A Python component is saved through its `to_json()` and `from_json()` methods,
#   so its class must be importable when the model is restored.
# - The TOML records no format version.
#   Treat checkpoints as tied to the RSCM version that wrote them.

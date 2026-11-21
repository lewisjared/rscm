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
# `Model.checkpoint()` returns a `Record` of the model at its current step.
# A record holds the model's builder, the step index, every timeseries and each component's internal state.
# It serialises as JSON, and `Model.restore()` rebuilds the model from it.
#
# ## Related Resources
#
# - [Coupled Models](../coupled_model/): Building multi-component models
# - [Scenario Pipeline](../scenario_pipeline/): Branching scenarios from a historical run
# - [Model Debugging and Inspection](../debugging_inspection/): Using checkpoints to debug

# %%
import json
import tempfile
from pathlib import Path

import numpy as np

from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import (
    InterpolationStrategy,
    Model,
    ModelBuilder,
    Record,
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
    "GtC/yr",
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
    .with_component(
        CarbonCycleBuilder.from_parameters(
            dict(tau=20.3, co2_pi=280.0, alpha_temperature=0.0)
        ).build()
    )
    .with_component(
        CO2ERFBuilder.from_parameters(dict(erf_2xco2=4.0, co2_pi=280.0)).build()
    )
    .with_time_axis(time_axis)
    .with_prescribed_variable("Surface Air Temperature Change", surface_temp)
    .with_prescribed_variable("Emissions|CO2", emissions)
    .with_initial_values(
        {
            "Cumulative Carbon Uptake|Land": 0.0,
            "Cumulative Emissions|CO2": 0.0,
            "Atmospheric Concentrations|CO2": 300.0,
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
# ## Recording the model
#
# A record serialises as JSON with these fields:
#
# - `format`: the record's layout version. A record of another format cannot be read.
# - `rscm`: the RSCM version that wrote it. Restoring a record from another version warns.
# - `builder`: the `ModelBuilder`, with its components, parameters, prescribed data and initial values.
# - `time_index`: the current position on the time axis.
# - `values`: every timeseries, with values solved so far and NaN (`null`) for later time points.
# - `timeseries_types`: whether each variable is a point or an average.
# - `states`: internal state that each component carries between steps, keyed by its label.

# %%
record = model.checkpoint()
saved = json.loads(record.to_json())

print("format:", saved["format"], "rscm:", saved["rscm"])
print("time_index:", saved["time_index"])
print("components:", [c["type"] for c in saved["builder"]["components"]])
print("labels:", saved["builder"]["labels"])
co2 = record.timeseries().get_timeseries_by_name("Atmospheric Concentrations|CO2")
print("Atmospheric Concentrations|CO2:", co2.values()[:4], "...")

# %% [markdown]
# ## Restoring the model
#
# `Model.restore()` builds the model again from the record's builder,
# then loads the saved values and component states.
# It fails if the rebuilt model stores a variable in another unit, grid or timeseries type,
# which happens when a component's declarations change between versions.
# The restored model is on the same step as the original.

# %%
new_model = Model.restore(record)
assert new_model.current_time() == model.current_time()

# %% [markdown]
# The restored model can then be run as normal.
# It produces the same results as the original model run to completion.

# %%
new_model.run()
model.run()

variable = "Atmospheric Concentrations|CO2"
np.testing.assert_allclose(
    new_model.timeseries().get_timeseries_by_name(variable).values(),
    model.timeseries().get_timeseries_by_name(variable).values(),
)
print("Restored and original runs match")

# %% [markdown]
# `model.output()` returns the same kind of record once a run finishes,
# so a run's results and its spec are saved together.
# Pickling and `copy.deepcopy` of a model also go through a record.

# %%
output = model.output()
print(f"Output record at step {output.time_index}")

# %% [markdown]
# ## Checkpoints on disk
#
# Write the JSON to a file to keep a checkpoint between sessions or share it with collaborators.
# `builder.build()` can be called again to get a fresh model from the same configuration.

# %%
checkpoint_model = builder.build()
while checkpoint_model.current_time() < 1900.0:
    checkpoint_model.step()

with tempfile.TemporaryDirectory() as tmp:
    checkpoint = Path(tmp) / "model_1900.json"
    checkpoint.write_text(checkpoint_model.checkpoint().to_json())

    resumed = Model.restore(Record.from_json(checkpoint.read_text()))

print(f"Resumed at {resumed.current_time()}")
resumed.run()
print(f"Finished at {resumed.current_time()}")

# %% [markdown]
# ## Limitations
#
# - A restored model keeps its original time axis.
#   `Model.branch()` changes future prescribed data and component parameters,
#   as the [Scenario Pipeline](../scenario_pipeline/) tutorial shows.
# - A Python component is saved through its `to_json()` and `from_json()` methods,
#   so its class must be importable when the model is restored.
#   Restoring imports the module the file names, so only load checkpoints you trust.
# - A checkpoint from another RSCM version restores with a warning,
#   and its results may differ from the run that wrote it.

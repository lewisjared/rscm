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
# It can be useful to be able to be able to capture the current state of a model
# and serialise it for later use.
#
# We provide an TOML representation of a model's state that provides a human-readable
# description of the model, including its components and state.

# %%
from docs.notebooks.helpers.models import example_model_builder

from rscm.core import Model

# %%
# Create a simple example coupled model with some exogenous data
model = example_model_builder(1750, 2100).build()

# %%
# We can then step the model forward in time
model.step()
model.step()

# %% [markdown]
# We can log the state of a model using TOML.
#
# This format is human-readable and can be used to store the state of a model to disk.
# A binary-format could be used to store the state of a model in a more compact way.
#

# %%
serialised_model = model.to_toml()
print(serialised_model)

# %% [markdown]
# ## Recreating the model state
#
# A new model can be initialised using the TOML representation.
# The new model will have the same state and components as the original model

# %%
new_model = Model.from_toml(serialised_model)
print(new_model.as_dot())

# %% [markdown]
# It will also on the same timestep as the original model.

# %%
assert new_model.current_time() == model.current_time()

# %% [markdown]
# The newly created model can then be run as normal.

# %%
new_model.run()

# %% [markdown]
# Currently we don't provide many hooks to modify the state of the model post-creation.
# A `Model` configuration is effectively immutable after creation.
#
# A `ModelBuilder` could be created using the contents of the model state to enable
# modification or to provide the initial conditions for a later run.
# The common use case that we are aiming to support is to be able to spin a model up
# over a historical period with a given set of parameters,
# and then run multiple different future scenarios with the same parameters.
#
# The historical period does not need to be run for every scenario.
# Instead a model can be created from the state of the historical model and run with
# the new scenario data for the future period.

# %% [markdown]
# Some psuedo-code that illustrates the desired operation is shown below:
#
# ```py
# historical_time_axis = TimeAxis.from_values(np.arange(1850, 2015 + 1))
# scenario_time_axis = TimeAxis.from_values(np.arange(2015, 2100))
# historical_exogenous_data: TimeseriesCollection = get_historical_data()
#
#
# model_builder = get_builder_with_components()
#
# historical_model = (
#     get_builder_with_components()
#     .with_time_axis(historical_time_axis)
#     .with_exogenous_collection(historical_exogenous_data)
# ).build()
#
# historical_model.run()
#
# historical_model_state = historical_model.to_toml()
#
# # This state can be archived for later use
#
# for scenario in get_list_of_scenarios():
#     model = (
#         ModelBuilder
#         .from_model_state(historical_model_state)
#         # Some components may require more than one previous timestep
#         # so we should merge time axes rather than dropping the historical data
#         .with_time_axis(scenario_time_axis.merge(historical_time_axis))
#         .with_current_time_step(2015.0)
#         .with_exogenous_collection(scenario)
#     ).build()
#
#     model.run()
#
#     # Save results
# ```
#

# %%

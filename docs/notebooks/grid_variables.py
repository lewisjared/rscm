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
# # Working with Grid Variables
#
# This notebook demonstrates how to use spatially-resolved variables in RSCM.
# You will learn to work with four-box and hemispheric grids, couple scalar
# components with gridded components, and understand grid validation patterns.
#
# ## Overview
#
# Climate variables often have spatial structure:
#
# - Temperatures differ between land and ocean
# - Ocean heat uptake varies by region
# - Aerosol forcing is spatially heterogeneous
#
# RSCM provides three grids to capture this:
#
# | Grid            | Regions              | Use Case                    |
# |-----------------|----------------------|-----------------------------|
# | **Scalar**      | 1 (global)           | Well-mixed, global averages |
# | **Hemispheric** | 2 (Northern/Southern)| Latitudinal gradients       |
# | **Four-box**    | 4 (NO/NL/SO/SL)      | MAGICC-style ocean-land     |
#
# ## Related Resources
#
# - [Spatial Grids Reference](../../grids/): Complete grid API documentation
# - [Coupled Models](../coupled_model/): Building multi-component models
# - [Debugging](../debugging_inspection/): Inspecting model state

# %%
import numpy as np

from rscm.component import Component, Input, Output
from rscm.components import FourBoxOceanHeatUptakeBuilder
from rscm.core import (
    FourBoxGrid,
    FourBoxRegion,
    FourBoxSlice,
    GridType,
    HemisphericGrid,
    HemisphericSlice,
    InterpolationStrategy,
    Model,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    VariableSchema,
)

# %% [markdown]
# ## Understanding Grids
#
# ### Four-box grid
#
# The four-box grid divides the globe into four regions based on hemisphere
# and surface type:
#
# ```
#           |   Ocean   |   Land   |
# ----------|-----------|----------|
# Northern  |    NO     |    NL    |
# Southern  |    SO     |    SL    |
# ```
#
# This structure captures key physical differences:
#
# - **Oceans** have higher heat capacity than land
# - **Northern Hemisphere** has more land area than southern
# - Different regions respond differently to forcing

# %%
# magicc_standard() weights the four regions equally
fb_grid = FourBoxGrid.magicc_standard()

print("Four-box grid properties:")
print(f"  Name: {fb_grid.grid_name()}")
print(f"  Size: {fb_grid.size()} regions")
print(f"  Regions: {fb_grid.region_names()}")
print(f"  Weights: {fb_grid.weights()}")

# %% [markdown]
# ### Hemispheric Grid
#
# The Hemispheric grid divides the globe into Northern and Southern hemispheres:
#
# ```
# Northern Hemisphere (0-90N)
# --------------------------
# Southern Hemisphere (0-90S)
# ```

# %%
hemi_grid = HemisphericGrid.equal_weights()

print("Hemispheric Grid Properties:")
print(f"  Name: {hemi_grid.grid_name()}")
print(f"  Size: {hemi_grid.size()} regions")
print(f"  Regions: {hemi_grid.region_names()}")
print(f"  Weights: {hemi_grid.weights()}")

# %% [markdown]
# ### Region Constants
#
# The region constants give the index of each region in a grid's values:
#
# | Constant | Index |
# |----------|-------|
# | `FourBoxRegion.NORTHERN_OCEAN` | 0 |
# | `FourBoxRegion.NORTHERN_LAND` | 1 |
# | `FourBoxRegion.SOUTHERN_OCEAN` | 2 |
# | `FourBoxRegion.SOUTHERN_LAND` | 3 |
# | `HemisphericRegion.NORTHERN` | 0 |
# | `HemisphericRegion.SOUTHERN` | 1 |

# %% [markdown]
# ## Grid Aggregation
#
# Grid values can be aggregated to a global scalar using area-weighted averaging.
# This is essential when connecting gridded components to scalar components.

# %%
# Warming above pre-industrial in K, ordered [NO, NL, SO, SL]
regional_temps = [1.5, 1.8, 0.9, 1.0]
print(f"Northern Land: {regional_temps[FourBoxRegion.NORTHERN_LAND]} K")

global_temp = fb_grid.aggregate_global(regional_temps)
print(f"Global mean (equal weights): {global_temp:.2f} K")

# %% [markdown]
# Equal weights ignore that the Northern Hemisphere has more land than the Southern.
# `FourBoxGrid.with_weights()` takes other grid weights instead.
# The weights below come from MAGICC's land fractions (42% of the NH, 21% of the SH)
# and are used throughout this notebook.

# %%
magicc_weights = [0.29, 0.21, 0.395, 0.105]
magicc_grid = FourBoxGrid.with_weights(magicc_weights)
print(
    f"Global mean (MAGICC weights): {magicc_grid.aggregate_global(regional_temps):.2f} K"
)

# %% [markdown]
# These grid objects only affect standalone calculations.
# To change the weights a model uses for its own aggregation,
# pass them to `ModelBuilder.with_grid_weights()` as shown [below](#aggregating-inside-a-model).

# %% [markdown]
# ### Aggregation Order Matters
#
# For nonlinear processes, the order of aggregation can affect results:
#
# 1. **Aggregate-then-compute**: Get global mean, apply function once
# 2. **Compute-then-aggregate**: Apply function regionally, then aggregate
#
# These give different results for nonlinear functions:


# %%
def nonlinear_response(temperature):
    """Calculate simple nonlinear climate response (quadratic)."""
    return temperature**2


# Approach 1: Aggregate inputs, then compute
global_temp = fb_grid.aggregate_global(regional_temps)
response_from_global = nonlinear_response(global_temp)
print(f"Approach 1 (aggregate-then-compute): {response_from_global:.3f}")

# Approach 2: Compute regionally, then aggregate
regional_responses = [nonlinear_response(t) for t in regional_temps]
response_aggregated = fb_grid.aggregate_global(regional_responses)
print(f"Approach 2 (compute-then-aggregate): {response_aggregated:.3f}")

# %% [markdown]
# The two differ because the function is nonlinear.
#
# **Best practice**: Use compute-then-aggregate when regional differences in
# the nonlinear response are physically meaningful.

# %% [markdown]
# ## Using Grid Components in a Model
#
# Now let's use a real gridded component in a model. The
# `FourBoxOceanHeatUptake` takes scalar forcing as input and
# disaggregates it to regional ocean heat uptake values.

# %%
heat_uptake_component = FourBoxOceanHeatUptakeBuilder.from_parameters(
    {
        "northern_ocean_ratio": 1.2,
        "northern_land_ratio": 0.6,
        "southern_ocean_ratio": 1.6,
        "southern_land_ratio": 0.6,
    }
).build()

print(f"Inputs:  {heat_uptake_component.input_names()}")
print(f"Outputs: {heat_uptake_component.output_names()}")

# %% [markdown]
# ### Setting Up the Model
#
# We need to provide:
#
# 1. A time axis for model execution
# 2. Prescribed forcing data (scalar input)
# 3. The gridded component

# %%
t_initial = 2000.0
t_final = 2100.0
time_axis = TimeAxis.from_values(np.arange(t_initial, t_final + 1, 10.0))

erf_times = np.array([t_initial, 2025.0, 2050.0, 2075.0, t_final])
erf_values = np.array([2.0, 3.0, 4.5, 5.5, 6.0])

forcing = Timeseries(
    erf_values,
    TimeAxis.from_values(erf_times),
    "W/m^2",
    InterpolationStrategy.Linear,
)

model = (
    ModelBuilder()
    .with_component(heat_uptake_component)
    .with_time_axis(time_axis)
    .with_prescribed_variable("Effective Radiative Forcing", forcing)
).build()

# %% [markdown]
# ### Running the Model and Extracting Grid Results
#
# Gridded outputs are read with the grid-specific getter.
# Their values have shape `(n_times, n_regions)`.

# %%
model.run()
collection = model.timeseries()

heat_uptake_ts = collection.get_four_box_timeseries_by_name("Heat Uptake")
values = heat_uptake_ts.values()
print(f"Unit: {heat_uptake_ts.unit}, shape: {values.shape}")

# %% [markdown]
# ### Visualising Regional Results
#
# The plot shows the uptake in each region against the global forcing that drives it.

# %% tags=["remove_input"]
import matplotlib.pyplot as plt

times = time_axis.values()
erf_ts = collection.get_timeseries_by_name("Effective Radiative Forcing")
region_names = ["Northern Ocean", "Northern Land", "Southern Ocean", "Southern Land"]

fig, ax = plt.subplots(figsize=(10, 5))
for idx, name in enumerate(region_names):
    ax.plot(times, values[:, idx], label=name, linewidth=2)
ax.plot(times, erf_ts.values(), "k--", label="Input ERF", linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("W/m²")
ax.set_title("Regional heat uptake vs global forcing")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### Aggregating Grid Output Back to Scalar
#
# To verify conservation, we can aggregate the regional values back to a
# global mean. With these ratios averaging to 1.0 and equal grid
# weights, the aggregated value should match the input forcing.
#
# A component reads a prescribed point at the end of the step, where it also writes its outputs.
# The uptake stored at a time point therefore comes from the forcing at the same point.

# %%
global_uptake = np.array([fb_grid.aggregate_global(row.tolist()) for row in values[1:]])
erf_at_end = erf_ts.values()[1:]
np.testing.assert_allclose(global_uptake, erf_at_end)

print("Aggregated uptake vs forcing at the end of each step:")
for label, i in [("First step", 0), ("Last step", -1)]:
    print(f"  {label}: ERF = {erf_at_end[i]:.2f}, uptake = {global_uptake[i]:.2f}")

# %% [markdown]
# ## Working with Hemispheric Grids
#
# Hemispheric grids are useful when the ocean-land distinction is less
# important than the North-South gradient.

# %%
# Warming in K, ordered [Northern, Southern]
hemi_temps = [1.6, 0.9]
print(f"Global mean: {hemi_grid.aggregate_global(hemi_temps)} K")

# %% [markdown]
# A model aggregates four-box values to Hemispheric by averaging ocean and land in each hemisphere.
# The [example below](#aggregating-inside-a-model) shows this.

# %% [markdown]
# ## Writing Gridded Python Components
#
# Declare a gridded variable by passing `grid="four_box"` or `grid="hemispheric"`
# to `Input`, `Output` or `State`.
# Return a `FourBoxSlice` or `HemisphericSlice` for gridded outputs.
#
# The component below scales a global forcing into regional temperatures.
# A second component reads those regional values and computes the land-ocean contrast.


# %%
class RegionalWarming(Component):
    """Scale global forcing into regional temperatures."""

    forcing = Input("Effective Radiative Forcing", unit="W/m^2")
    temperature = Output("Surface Air Temperature Change", unit="K", grid="four_box")

    def __init__(self, sensitivity: list[float]):
        # Regional sensitivities in K per W/m^2, ordered [NO, NL, SO, SL]
        self.sensitivity = sensitivity

    def solve(self, t_current, t_next, inputs):
        """Compute regional temperatures from the forcing at the end of the step."""
        erf = inputs.forcing.get()
        return self.Outputs(
            temperature=FourBoxSlice.from_array([s * erf for s in self.sensitivity])
        )


class LandOceanContrast(Component):
    """Difference between mean land and mean ocean temperature."""

    temperature = Input("Surface Air Temperature Change", unit="K", grid="four_box")
    contrast = Output("Land-Ocean Contrast", unit="K")

    def solve(self, t_current, t_next, inputs):
        """Read regional values from the four-box window."""
        regional, _ = inputs.temperature.interval()
        land = (regional.northern_land + regional.southern_land) / 2
        ocean = (
            regional.get(FourBoxRegion.NORTHERN_OCEAN)
            + regional.get(FourBoxRegion.SOUTHERN_OCEAN)
        ) / 2
        return self.Outputs(contrast=land - ocean)


# %% [markdown]
# A grid window gives access to the regional values:
#
# - `get()` returns a `FourBoxSlice` (or `HemisphericSlice`) of the values for the step.
# - `interval()` returns the slices at the start and end of the step.
# - A slice's `get(region)` returns a single regional value.
# - `region(region)` returns a scalar `TimeseriesWindow` for one region's history.
#
# `LandOceanContrast` reads the start of the step,
# so it sees `RegionalWarming`'s output one step after it is written.
# `get()` would read this step's output instead.

# %%
sensitivity = [0.7, 1.1, 0.5, 0.9]
grid_time_axis = TimeAxis.from_values(np.arange(2000.0, 2011.0))
grid_forcing = Timeseries(
    np.array([1.0, 2.0]),
    TimeAxis.from_values(np.array([2000.0, 2010.0])),
    "W/m^2",
    InterpolationStrategy.Linear,
)

schema = VariableSchema()
schema.add_variable("Effective Radiative Forcing", "W/m^2")
schema.add_variable("Surface Air Temperature Change", "K", GridType.FourBox)
schema.add_variable("Land-Ocean Contrast", "K")

regional_model = (
    ModelBuilder()
    .with_time_axis(grid_time_axis)
    .with_schema(schema)
    .with_component(RegionalWarming(sensitivity))
    .with_component(LandOceanContrast())
    .with_prescribed_variable("Effective Radiative Forcing", grid_forcing)
).build()
regional_model.run()

regional_results = regional_model.timeseries()
temps = regional_results.get_four_box_timeseries_by_name(
    "Surface Air Temperature Change"
).values()
contrast = regional_results.get_timeseries_by_name("Land-Ocean Contrast").values()
for year, t, c in zip(grid_time_axis.values()[1:4], temps[1:4], contrast[1:4]):
    print(
        f"{year:.0f}  [NO, NL, SO, SL] = {np.round(t, 2)}  land-ocean contrast = {c:.2f} K"
    )

# %% [markdown]
# ## Aggregating Inside a Model
#
# The schema decides the grid a variable is stored on.
# If a component writes a finer grid than the schema declares,
# the model aggregates the output when writing it.
# Here the schema stores `Surface Air Temperature Change` as a scalar,
# so the four-box output of `RegionalWarming` becomes a global mean.
#
# By default the model aggregates with equal weights.
# `ModelBuilder.with_grid_weights()` sets the grid weights for a grid.
# The weights must sum to 1.0, ordered `[NO, NL, SO, SL]` for the four-box grid and `[N, S]` for the hemispheric grid.
#
# The helper below builds the `RegionalWarming` model with `Surface Air Temperature Change` stored on a chosen grid.


# %%
def build_regional_model(
    temperature_grid: GridType, weights: list[float] | None = None
) -> Model:
    """Build a model whose schema stores Surface Air Temperature Change on the given grid."""
    schema = VariableSchema()
    schema.add_variable("Effective Radiative Forcing", "W/m^2")
    schema.add_variable("Surface Air Temperature Change", "K", temperature_grid)

    builder = (
        ModelBuilder()
        .with_time_axis(grid_time_axis)
        .with_schema(schema)
        .with_component(RegionalWarming(sensitivity))
        .with_prescribed_variable("Effective Radiative Forcing", grid_forcing)
    )
    if weights is not None:
        builder = builder.with_grid_weights(GridType.FourBox, weights)
    return builder.build()


equal_model = build_regional_model(GridType.Scalar)
weighted_model = build_regional_model(GridType.Scalar, magicc_weights)
equal_model.run()
weighted_model.run()

for label, m in [("Equal weights", equal_model), ("MAGICC weights", weighted_model)]:
    temps = (
        m.timeseries().get_timeseries_by_name("Surface Air Temperature Change").values()
    )
    print(f"{label}: {np.round(temps[1:4], 3)}")

# %% [markdown]
# `coupling()` lists the grid transformations the model applies.

# %%
print(weighted_model.coupling("plain"))

# %% [markdown]
# Declaring the schema variable as `Hemispheric` instead averages ocean and land within each hemisphere.

# %%
hemi_model = build_regional_model(GridType.Hemispheric)
hemi_model.run()
hemi_values = (
    hemi_model.timeseries()
    .get_hemispheric_timeseries_by_name("Surface Air Temperature Change")
    .values()
)
print(f"[N, S] for 2001 to 2003:\n{np.round(hemi_values[1:4], 3)}")

# %% [markdown]
# Rust components also support read-side aggregation,
# where a component reads a coarser grid than the schema stores.
# Python components receive the window at the schema's resolution.
# See [Schema-Driven Auto-Aggregation](../../grids/#schema-driven-auto-aggregation) for details.

# %% [markdown]
# ## Grid Coupling Rules
#
# The model only aggregates from finer to coarser grids:
#
# | From → To             | Transformation         | Automatic? |
# |-----------------------|------------------------|------------|
# | FourBox → Scalar      | Weighted average       | Yes        |
# | FourBox → Hemispheric | Average per hemisphere | Yes        |
# | Hemispheric → Scalar  | Weighted average       | Yes        |
# | Scalar → FourBox      | Broadcast              | No         |
# | Scalar → Hemispheric  | Broadcast              | No         |
# | Hemispheric → FourBox | Disaggregation         | No         |
#
# Automatic aggregation needs a schema that declares the coarser grid.
# The builder validates grids when the model is built.
# Without a schema, a four-box producer cannot feed a Scalar consumer:


# %%
class ScalarReader(Component):
    """Reads Surface Air Temperature Change as a global value."""

    temperature = Input("Surface Air Temperature Change", unit="K")
    response = Output("Response", unit="K")

    def solve(self, t_current, t_next, inputs):
        """Pass the global temperature through."""
        return self.Outputs(response=inputs.temperature.get())


try:
    (
        ModelBuilder()
        .with_time_axis(grid_time_axis)
        .with_component(RegionalWarming(sensitivity))
        .with_component(ScalarReader())
        .with_prescribed_variable("Effective Radiative Forcing", grid_forcing)
    ).build()
except ValueError as e:
    print(e)


# %% [markdown]
# Disaggregation is rejected even with a schema,
# because there is no physical basis to split coarse values into finer regions without extra assumptions.


# %%
class HemisphericProducer(Component):
    """Produces hemispheric temperatures."""

    temperature = Output("Surface Air Temperature Change", unit="K", grid="hemispheric")

    def solve(self, t_current, t_next, inputs):
        """Return fixed hemispheric values."""
        return self.Outputs(temperature=HemisphericSlice(northern=1.2, southern=0.8))


four_box_schema = VariableSchema()
four_box_schema.add_variable("Surface Air Temperature Change", "K", GridType.FourBox)

try:
    (
        ModelBuilder()
        .with_time_axis(grid_time_axis)
        .with_schema(four_box_schema)
        .with_component(HemisphericProducer())
    ).build()
except ValueError as e:
    print(e)

# %% [markdown]
# ### Disaggregation Best Practice
#
# To go from a coarse grid to a finer one, write a component that states the assumption.
# `HemisphericToFourBox` assumes the ocean is warmer than the land by a fixed ratio
# and keeps each hemisphere's mean unchanged.


# %%
class HemisphericToFourBox(Component):
    """Split hemispheric temperatures into ocean and land with a fixed ocean/land ratio."""

    hemispheric = Input("Surface Air Temperature Change", unit="K", grid="hemispheric")
    four_box = Output("Regional Surface Temperature", unit="K", grid="four_box")

    def __init__(self, ocean_land_ratio: float):
        self.ocean_factor = 2 * ocean_land_ratio / (1 + ocean_land_ratio)
        self.land_factor = 2 / (1 + ocean_land_ratio)

    def solve(self, t_current, t_next, inputs):
        """Scale each hemisphere so that its ocean and land values average to it."""
        hemi, _ = inputs.hemispheric.interval()
        return self.Outputs(
            four_box=FourBoxSlice(
                northern_ocean=hemi.northern * self.ocean_factor,
                northern_land=hemi.northern * self.land_factor,
                southern_ocean=hemi.southern * self.ocean_factor,
                southern_land=hemi.southern * self.land_factor,
            )
        )


split_model = (
    ModelBuilder()
    .with_time_axis(grid_time_axis)
    .with_component(HemisphericProducer())
    .with_component(HemisphericToFourBox(ocean_land_ratio=1.2))
).build()
split_model.run()

no, nl, so, sl = (
    split_model.timeseries()
    .get_four_box_timeseries_by_name("Regional Surface Temperature")
    .values()[-1]
)
print(f"NO={no:.3f}, NL={nl:.3f}, SO={so:.3f}, SL={sl:.3f}")
np.testing.assert_allclose([(no + nl) / 2, (so + sl) / 2], [1.2, 0.8])

# %% [markdown]
# The final check confirms that ocean and land average back to the hemispheric input.
#
# ## Summary
#
# - Choose the grid from the physics: well-mixed quantities are Scalar, regional ones gridded.
# - Aggregation order matters for nonlinear processes.
# - The schema sets the stored grid, and the model aggregates finer outputs on write.
# - `with_grid_weights()` sets the grid weights the model uses for aggregation.
# - Disaggregation needs an explicit assumption, written as a component.
#
# ### Next Steps
#
# - [Spatial Grids Reference](../../grids/): Full API documentation
# - [Debugging](../debugging_inspection/): Inspecting grid variables in models
# - [Rust Components](../component_rust/): Creating gridded Rust components

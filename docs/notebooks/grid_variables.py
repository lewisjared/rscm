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
# You will learn to work with FourBox and Hemispheric grids, couple scalar
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
# RSCM provides three grid types to capture this:
#
# | Grid Type       | Regions              | Use Case                    |
# |-----------------|----------------------|-----------------------------|
# | **Scalar**      | 1 (global)           | Well-mixed, global averages |
# | **Hemispheric** | 2 (Northern/Southern)| Latitudinal gradients       |
# | **FourBox**     | 4 (NO/NL/SO/SL)      | MAGICC-style ocean-land     |
#
# ## Related Resources
#
# - [Spatial Grids Reference](../grids.md): Complete grid API documentation
# - [Coupled Models](coupled_model.py): Building multi-component models
# - [Debugging](debugging_inspection.py): Inspecting model state

# %%
import matplotlib.pyplot as plt
import numpy as np

from rscm.components import FourBoxOceanHeatUptakeBuilder
from rscm.core import (
    FourBoxGrid,
    FourBoxRegion,
    HemisphericGrid,
    HemisphericRegion,
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
)

# %% [markdown]
# ## Understanding Grid Types
#
# ### FourBox Grid
#
# The FourBox grid divides the globe into four regions based on hemisphere
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
# Create a FourBox grid with equal weights
fb_grid = FourBoxGrid.magicc_standard()

print("FourBox Grid Properties:")
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
# Create a Hemispheric grid
hemi_grid = HemisphericGrid.equal_weights()

print("Hemispheric Grid Properties:")
print(f"  Name: {hemi_grid.grid_name()}")
print(f"  Size: {hemi_grid.size()} regions")
print(f"  Regions: {hemi_grid.region_names()}")
print(f"  Weights: {hemi_grid.weights()}")

# %% [markdown]
# ### Region Constants
#
# Access individual regions using the region constants:

# %%
print("FourBox Region Indices:")
print(f"  Northern Ocean (NO): {FourBoxRegion.NORTHERN_OCEAN}")
print(f"  Northern Land (NL): {FourBoxRegion.NORTHERN_LAND}")
print(f"  Southern Ocean (SO): {FourBoxRegion.SOUTHERN_OCEAN}")
print(f"  Southern Land (SL): {FourBoxRegion.SOUTHERN_LAND}")

print("\nHemispheric Region Indices:")
print(f"  Northern: {HemisphericRegion.NORTHERN}")
print(f"  Southern: {HemisphericRegion.SOUTHERN}")

# %% [markdown]
# ## Grid Aggregation
#
# Grid values can be aggregated to a global scalar using area-weighted averaging.
# This is essential when connecting gridded components to scalar components.

# %%
# Regional temperature values (example: K above pre-industrial)
regional_temps = [1.5, 1.8, 0.9, 1.0]  # [NO, NL, SO, SL]
print(f"Regional temperatures: {regional_temps}")
print(f"  Northern Ocean: {regional_temps[FourBoxRegion.NORTHERN_OCEAN]:.1f} K")
print(f"  Northern Land:  {regional_temps[FourBoxRegion.NORTHERN_LAND]:.1f} K")
print(f"  Southern Ocean: {regional_temps[FourBoxRegion.SOUTHERN_OCEAN]:.1f} K")
print(f"  Southern Land:  {regional_temps[FourBoxRegion.SOUTHERN_LAND]:.1f} K")

# Aggregate to global mean with equal weights
global_temp = fb_grid.aggregate_global(regional_temps)
print(f"\nGlobal mean (equal weights): {global_temp:.2f} K")

# %%
# With custom weights (e.g., realistic area fractions)
# Northern Hemisphere has more land, Southern has more ocean
custom_weights = [0.25, 0.16, 0.42, 0.17]
custom_grid = FourBoxGrid.with_weights(custom_weights)

global_temp_weighted = custom_grid.aggregate_global(regional_temps)
print(f"Custom weights: {custom_weights}")
print(f"Global mean (weighted): {global_temp_weighted:.2f} K")

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

print(f"\nDifference: {abs(response_from_global - response_aggregated):.3f}")
print("(Non-zero because function is nonlinear)")

# %% [markdown]
# **Best practice**: Use compute-then-aggregate when regional differences in
# the nonlinear response are physically meaningful.

# %% [markdown]
# ## Using Grid Components in a Model
#
# Now let's use a real gridded component in a model. The
# `FourBoxOceanHeatUptake` takes scalar forcing as input and
# disaggregates it to regional ocean heat uptake values.

# %%
# Create a FourBox ocean heat uptake component
# This demonstrates scalar-to-grid disaggregation
heat_uptake_component = FourBoxOceanHeatUptakeBuilder.from_parameters(
    {
        "northern_ocean_ratio": 1.2,  # Oceans absorb more
        "northern_land_ratio": 0.6,  # Land absorbs less
        "southern_ocean_ratio": 1.6,  # Southern Ocean most efficient
        "southern_land_ratio": 0.6,
    }
).build()

print("FourBoxOceanHeatUptake Component:")
print(f"  Inputs: {heat_uptake_component.input_names()}")
print(f"  Outputs: {heat_uptake_component.output_names()}")

# %% [markdown]
# ### Setting Up the Model
#
# We need to provide:
#
# 1. A time axis for model execution
# 2. Exogenous forcing data (scalar input)
# 3. The gridded component

# %%
# Time axis: 2000 to 2100
t_initial = 2000.0
t_final = 2100.0
time_axis = TimeAxis.from_values(np.arange(t_initial, t_final + 1, 10.0))

# Exogenous forcing data - scalar effective radiative forcing (W/m^2)
# Simulate increasing forcing over time
erf_times = np.array([t_initial, 2025.0, 2050.0, 2075.0, t_final])
erf_values = np.array([2.0, 3.0, 4.5, 5.5, 6.0])

forcing = Timeseries(
    erf_values,
    TimeAxis.from_values(erf_times),
    "W/m^2",
    InterpolationStrategy.Linear,
)

# Build the model
model = (
    ModelBuilder()
    .with_rust_component(heat_uptake_component)
    .with_time_axis(time_axis)
    .with_exogenous_variable("Effective Radiative Forcing|Aggregated", forcing)
).build()

print("Model built successfully")
print(f"Time axis: {t_initial} to {t_final}")

# %% [markdown]
# ### Running the Model and Extracting Grid Results

# %%
# Run the model
model.run()

# Get the results
collection = model.timeseries()

print("Variables in model:")
for name in collection.names():
    print(f"  - {name}")

# %%
# Get the gridded output
heat_uptake_ts = collection.get_fourbox_timeseries_by_name("Ocean Heat Uptake|FourBox")

print("Heat Uptake Variable:")
print(f"  Units: {heat_uptake_ts.units}")
print("  Grid type: FourBox (4 regions)")

# Get values - shape is (n_times, n_regions)
values = heat_uptake_ts.values()
print(f"  Values shape: {values.shape}")

# %% [markdown]
# ### Visualising Regional Results

# %%
# Extract regional timeseries
times = time_axis.values()
valid_mask = ~np.isnan(values[:, 0])
plot_times = times[valid_mask]

# Create a plot showing all four regions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
region_names = ["Northern Ocean", "Northern Land", "Southern Ocean", "Southern Land"]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

for idx, (ax, name, color) in enumerate(zip(axes.flatten(), region_names, colors)):
    region_values = values[valid_mask, idx]
    ax.plot(plot_times, region_values, color=color, linewidth=2)
    ax.set_title(name)
    ax.set_xlabel("Year")
    ax.set_ylabel("Heat Uptake (W/m²)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_initial, t_final)

plt.suptitle("Regional Ocean Heat Uptake (FourBox)", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Compare all regions on one plot
plt.figure(figsize=(10, 6))

for idx, (name, color) in enumerate(zip(region_names, colors)):
    region_values = values[valid_mask, idx]
    plt.plot(plot_times, region_values, label=name, color=color, linewidth=2)

# Also plot the input forcing (scaled for comparison)
erf_ts = collection.get_timeseries_by_name("Effective Radiative Forcing|Aggregated")
erf_values_full = erf_ts.values()[valid_mask]
plt.plot(
    plot_times,
    erf_values_full,
    label="Input ERF",
    color="black",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Year")
plt.ylabel("W/m²")
plt.title("Regional Heat Uptake vs Global Forcing")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### Aggregating Grid Output Back to Scalar
#
# To verify conservation, we can aggregate the regional values back to a
# global mean. With the default ratios averaging to 1.0 and equal grid
# weights, the aggregated value should match the input forcing.

# %%
# Aggregate the regional heat uptake back to global
grid = FourBoxGrid.magicc_standard()

global_uptake = []
for i in range(len(plot_times)):
    timestep_values = values[valid_mask][i, :]
    global_val = grid.aggregate_global(timestep_values.tolist())
    global_uptake.append(global_val)

global_uptake = np.array(global_uptake)

# Compare with input forcing
print("Conservation check (global uptake ≈ input ERF):")
print(
    f"  Year {plot_times[0]:.0f}: Input ERF = {erf_values_full[0]:.2f}, "
    f"Aggregated uptake = {global_uptake[0]:.2f}"
)
print(
    f"  Year {plot_times[-1]:.0f}: Input ERF = {erf_values_full[-1]:.2f}, "
    f"Aggregated uptake = {global_uptake[-1]:.2f}"
)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(plot_times, erf_values_full, label="Input ERF", linewidth=2)
plt.plot(
    plot_times,
    global_uptake,
    label="Aggregated Heat Uptake",
    linestyle="--",
    linewidth=2,
)
plt.xlabel("Year")
plt.ylabel("W/m²")
plt.title("Conservation: Input Forcing vs Aggregated Output")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Creating Custom Grid Weights
#
# For more realistic simulations, you can use custom area weights that
# reflect the actual proportions of ocean and land in each hemisphere.

# %%
# Realistic area fractions (approximate)
# Northern: ~61% ocean, ~39% land
# Southern: ~81% ocean, ~19% land
# Global: Northern ~50%, Southern ~50%

realistic_weights = [
    0.5 * 0.61,  # Northern Ocean: ~30.5%
    0.5 * 0.39,  # Northern Land: ~19.5%
    0.5 * 0.81,  # Southern Ocean: ~40.5%
    0.5 * 0.19,  # Southern Land: ~9.5%
]
print(f"Realistic area weights: {realistic_weights}")
print(f"Sum: {sum(realistic_weights):.2f}")

realistic_grid = FourBoxGrid.with_weights(realistic_weights)

# Aggregate with realistic weights
example_values = [1.5, 1.8, 0.9, 1.0]
equal_mean = fb_grid.aggregate_global(example_values)
realistic_mean = realistic_grid.aggregate_global(example_values)

print(f"\nRegional temperatures: {example_values}")
print(f"Global mean (equal weights):     {equal_mean:.3f} K")
print(f"Global mean (realistic weights): {realistic_mean:.3f} K")

# %% [markdown]
# ## Working with Hemispheric Grids
#
# Hemispheric grids are useful when the ocean-land distinction is less
# important than the North-South gradient.

# %%
# Hemispheric temperature example
hemi_temps = [1.6, 0.9]  # [Northern, Southern]
print(
    f"Hemispheric temperatures: Northern = {hemi_temps[0]} K, "
    f"Southern = {hemi_temps[1]} K"
)

# Aggregate to global
global_hemi = hemi_grid.aggregate_global(hemi_temps)
print(f"Global mean: {global_hemi} K")

# %%
# Hemispheric vs FourBox: relationship
# FourBox can be aggregated to Hemispheric by averaging ocean+land per hemisphere

fourbox_temps = [1.5, 1.8, 0.9, 1.0]  # [NO, NL, SO, SL]

# Manual aggregation to hemispheric (with equal area weights within hemisphere)
northern_mean = (fourbox_temps[0] + fourbox_temps[1]) / 2  # (NO + NL) / 2
southern_mean = (fourbox_temps[2] + fourbox_temps[3]) / 2  # (SO + SL) / 2

print("FourBox to Hemispheric aggregation:")
print(
    f"  FourBox values: NO={fourbox_temps[0]}, NL={fourbox_temps[1]}, "
    f"SO={fourbox_temps[2]}, SL={fourbox_temps[3]}"
)
print(
    f"  Northern mean: ({fourbox_temps[0]} + {fourbox_temps[1]}) / 2 = {northern_mean}"
)
print(
    f"  Southern mean: ({fourbox_temps[2]} + {fourbox_temps[3]}) / 2 = {southern_mean}"
)

# %% [markdown]
# ## Grid Coupling Rules
#
# RSCM validates grid compatibility when building models:
#
# | From → To     | Transformation                  | Automatic? |
# |---------------|--------------------------------|------------|
# | FourBox → Scalar | Weighted average              | Yes        |
# | FourBox → Hemispheric | Average per hemisphere   | Yes        |
# | Hemispheric → Scalar | Weighted average          | Yes        |
# | Scalar → FourBox | Broadcast (same to all)       | Yes        |
# | Scalar → Hemispheric | Broadcast                 | Yes        |
# | Hemispheric → FourBox | **Not supported**        | No         |
#
# The last case (Hemispheric → FourBox) requires explicit disaggregation
# because there's no physical basis to split hemisphere values into
# ocean/land without additional assumptions.

# %% [markdown]
# ### Disaggregation Best Practice
#
# When disaggregating from coarse to fine grids, make assumptions explicit:


# %%
def hemispheric_to_fourbox(hemi_values, ocean_ratio=1.1):
    """
    Disaggregate hemispheric values to FourBox.

    Assumes oceans are warmer than land by a fixed ratio.

    Parameters
    ----------
    hemi_values : list
        [Northern, Southern] values
    ocean_ratio : float
        Ocean/land ratio (> 1 means oceans warmer)

    Returns
    -------
    list
        [NO, NL, SO, SL] values
    """
    north, south = hemi_values

    # Split each hemisphere into ocean/land with conservation
    # If O/L = r, and (O + L)/2 = mean, then:
    # O = mean * 2r / (1 + r)
    # L = mean * 2 / (1 + r)
    factor_ocean = 2 * ocean_ratio / (1 + ocean_ratio)
    factor_land = 2 / (1 + ocean_ratio)

    return [
        north * factor_ocean,  # Northern Ocean
        north * factor_land,  # Northern Land
        south * factor_ocean,  # Southern Ocean
        south * factor_land,  # Southern Land
    ]


# Example
hemi_input = [1.5, 1.0]  # Northern warmer than Southern
fourbox_output = hemispheric_to_fourbox(hemi_input, ocean_ratio=1.2)

print(f"Hemispheric input: N={hemi_input[0]}, S={hemi_input[1]}")
print("FourBox output (ocean_ratio=1.2):")
print(f"  NO={fourbox_output[0]:.3f}, NL={fourbox_output[1]:.3f}")
print(f"  SO={fourbox_output[2]:.3f}, SL={fourbox_output[3]:.3f}")

# Verify conservation (aggregate back should match input)
north_check = (fourbox_output[0] + fourbox_output[1]) / 2
south_check = (fourbox_output[2] + fourbox_output[3]) / 2
print("\nConservation check:")
print(f"  Northern: input={hemi_input[0]}, aggregated={north_check:.3f}")
print(f"  Southern: input={hemi_input[1]}, aggregated={south_check:.3f}")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated how to:
#
# 1. **Understand grid types** - Scalar, Hemispheric, and FourBox
# 2. **Access regional values** - Using region constants
# 3. **Aggregate grid values** - Area-weighted global means
# 4. **Use gridded components** - FourBoxOceanHeatUptake example
# 5. **Visualise regional outputs** - Plotting four-region timeseries
# 6. **Work with custom weights** - Realistic area fractions
# 7. **Handle grid coupling** - Understanding transformation rules
#
# ### Key Takeaways
#
# - Choose grid type based on physical requirements (well-mixed vs regional)
# - Aggregation order matters for nonlinear processes
# - Custom area weights give more realistic global means
# - Disaggregation requires explicit physical assumptions
# - FourBox → Hemispheric → Scalar transformations are automatic
# - Hemispheric → FourBox requires custom components
#
# ### Next Steps
#
# - [Spatial Grids Reference](../grids.md): Full API documentation
# - [Debugging](debugging_inspection.py): Inspecting grid variables in models
# - [Rust Components](component_rust.md): Creating gridded Rust components

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
# # RSCM Units Module
#
# This notebook provides a comprehensive guide to the RSCM units module, which handles parsing, normalisation, and conversion of physical units commonly used in climate modelling.
#
# ## Features
#
# - **Flexible parsing**: Handles various notations for the same unit (`W/m^2`, `W / m ^ 2`, `W m^-2` are all equivalent)
# - **Dimensional analysis**: Validates that conversions are physically meaningful
# - **Climate-specific units**: Carbon (C, CO₂), concentrations (ppm, ppb), radiative forcing (W/m²), and emissions rates (GtC/yr)
# - **Automatic conversion factors**: Calculates multipliers between compatible units including the CO₂-C molecular weight ratio

# %%
# Import the Unit class from RSCM
from rscm.core import Unit

# %% [markdown]
# ## 1. Parsing Units
#
# The `Unit` class parses unit strings with flexible syntax. Whitespace is normalised automatically, and multiple notations are supported.

# %%
# All of these are equivalent
u1 = Unit("W/m^2")
u2 = Unit("W / m ^ 2")
u3 = Unit("W m^-2")
u4 = Unit("W per m^2")

print(f"u1: {u1}")
print(f"u2: {u2}")
print(f"u3: {u3}")
print(f"u4: {u4}")

# They all compare as equal
assert u1 == u2 == u3 == u4
print("\nAll units are equal after normalisation!")

# %% [markdown]
# ### Supported Syntax
#
# The parser accepts several equivalent notations:
#
# | Notation | Meaning |
# |----------|---------|
# | `m^2`, `m**2`, `m2` | Square metres |
# | `W/m^2`, `W m^-2`, `W per m^2` | Watts per square metre |
# | `kg m`, `kg*m`, `kg·m` | Kilogram-metres |
# | `GtC / yr` | Gigatonnes of carbon per year |

# %%
# Exponent notations
print("Exponent notations:")
for notation in ["m^2", "m**2", "m2"]:
    u = Unit(notation)
    print(f"  {notation!r:10} -> {u.normalized()}")

print("\nDivision notations:")
for notation in ["W/m^2", "W m^-2", "W per m^2"]:
    u = Unit(notation)
    print(f"  {notation!r:15} -> {u.normalized()}")

# %% [markdown]
# ### Original vs Normalised
#
# The `Unit` class preserves the original input string while also providing a normalised form for comparison.

# %%
u = Unit("  GtC  /  yr  ")
print(f"Original:   {u.original!r}")
print(f"Normalised: {u.normalized()!r}")

# %% [markdown]
# ## 2. Climate-Specific Units
#
# The module includes units commonly used in climate modelling:
#
# | Category | Units |
# |----------|-------|
# | Carbon | `C`, `CO2` with automatic molecular weight conversion (44/12) |
# | Mass prefixes | `Gt` (giga-tonne), `Mt` (mega-tonne), `kt`, `t`, etc. |
# | Concentrations | `ppm`, `ppb`, `ppt` |
# | Time | `yr` (365.25 days), `day`, `h`, `min`, `s` |
# | Energy/Power | `W`, `J` with SI prefixes |
# | Temperature | `K`, `degC` (for differences) |

# %%
# Common climate units
climate_units = [
    "GtC/yr",  # Carbon emissions
    "GtCO2/yr",  # CO2 emissions
    "MtCO2/yr",  # CO2 emissions (megatonnes)
    "ppm",  # Concentration
    "ppb",  # Concentration
    "W/m^2",  # Radiative forcing
    "K",  # Temperature
]

print("Climate units:")
for unit_str in climate_units:
    u = Unit(unit_str)
    print(f"  {unit_str:12} -> normalised: {u.normalized()}")

# %% [markdown]
# ## 3. Checking Compatibility
#
# Units are compatible if they have the same physical dimension. For example, `GtC/yr` and `MtCO2/yr` are both mass per time, so they are compatible for conversion.

# %%
gtc_yr = Unit("GtC/yr")
mtco2_yr = Unit("MtCO2/yr")
flux = Unit("W/m^2")

print(f"GtC/yr compatible with MtCO2/yr? {gtc_yr.is_compatible(mtco2_yr)}")
print(f"GtC/yr compatible with W/m^2?    {gtc_yr.is_compatible(flux)}")

# %% [markdown]
# ### Dimensionless Units
#
# Some units like concentrations (ppm, ppb) are dimensionless.

# %%
ppm = Unit("ppm")
ppb = Unit("ppb")
watt = Unit("W")

print(f"ppm is dimensionless: {ppm.is_dimensionless()}")
print(f"ppb is dimensionless: {ppb.is_dimensionless()}")
print(f"W is dimensionless:   {watt.is_dimensionless()}")
print(f"\nppm compatible with ppb: {ppm.is_compatible(ppb)}")

# %% [markdown]
# ## 4. Unit Conversion
#
# The main use case is converting between compatible units. The `conversion_factor()` method returns the multiplier to convert from one unit to another.

# %%
gtc = Unit("GtC/yr")
mtco2 = Unit("MtCO2/yr")

factor = gtc.conversion_factor(mtco2)
print(f"Conversion factor from GtC/yr to MtCO2/yr: {factor:.2f}")
print("\nBreakdown:")
print("  - Gt to Mt: 1000x")
print(f"  - C to CO2: 44/12 = {44 / 12:.4f}x (molecular weight ratio)")
print(f"  - Total: 1000 * 44/12 = {1000 * 44 / 12:.2f}")

# %% [markdown]
# ### Converting Values
#
# Use the `convert()` method to convert actual values.

# %%
# Convert 10 GtC/yr to MtCO2/yr
value_gtc = 10.0
value_mtco2 = gtc.convert(value_gtc, mtco2)

print(f"{value_gtc} GtC/yr = {value_mtco2:.1f} MtCO2/yr")

# Convert back
value_back = mtco2.convert(value_mtco2, gtc)
print(f"{value_mtco2:.1f} MtCO2/yr = {value_back} GtC/yr")

# %% [markdown]
# ## 5. Common Conversion Examples
#
# Here are some commonly needed conversions in climate modelling:

# %%
conversions = [
    ("GtC", "MtCO2"),  # Carbon to CO2 with prefix
    ("GtCO2", "GtC"),  # CO2 to carbon
    ("GtC/yr", "GtC/s"),  # Different time units
    ("km", "m"),  # Distance
    ("ppm", "ppb"),  # Concentration
    ("GW", "MW"),  # Power
    ("GW", "W"),  # Power to base unit
]

print(f"{'From':<12} {'To':<12} {'Factor':>15}")
print("-" * 40)
for from_str, to_str in conversions:
    from_unit = Unit(from_str)
    to_unit = Unit(to_str)
    factor = from_unit.conversion_factor(to_unit)
    print(f"{from_str:<12} {to_str:<12} {factor:>15.6g}")

# %% [markdown]
# ## 6. Carbon-CO₂ Conversions
#
# A critical conversion in climate modelling is between carbon mass and CO₂ mass. The molecular weight ratio is 44/12 ≈ 3.667.
#
# - **C → CO₂**: Multiply by 44/12
# - **CO₂ → C**: Multiply by 12/44

# %%
# Carbon to CO2
gtc = Unit("GtC")
gtco2 = Unit("GtCO2")

c_to_co2 = gtc.conversion_factor(gtco2)
co2_to_c = gtco2.conversion_factor(gtc)

print("Carbon ↔ CO₂ conversion factors:")
print(f"  C  → CO₂: {c_to_co2:.6f} (≈ 44/12 = {44 / 12:.6f})")
print(f"  CO₂ → C:  {co2_to_c:.6f} (≈ 12/44 = {12 / 44:.6f})")

# Verify round-trip
round_trip = c_to_co2 * co2_to_c
print(f"\nRound-trip verification: {round_trip:.10f} (should be 1.0)")

# %% [markdown]
# ### Practical Example: Emissions Pathway Conversion
#
# Climate scenarios often provide emissions in GtCO₂/yr, but carbon cycle models may use GtC/yr internally.

# %%
# Sample emissions pathway in GtCO2/yr
years = [2020, 2030, 2040, 2050]
emissions_gtco2 = [40.0, 35.0, 25.0, 10.0]  # GtCO2/yr

gtco2_yr = Unit("GtCO2/yr")
gtc_yr = Unit("GtC/yr")

# Convert to GtC/yr for carbon cycle model
emissions_gtc = [gtco2_yr.convert(e, gtc_yr) for e in emissions_gtco2]

print(f"{'Year':<6} {'GtCO2/yr':>10} {'GtC/yr':>10}")
print("-" * 28)
for year, e_co2, e_c in zip(years, emissions_gtco2, emissions_gtc):
    print(f"{year:<6} {e_co2:>10.1f} {e_c:>10.2f}")

# %% [markdown]
# ## 7. SI Prefixes
#
# The unit system supports all standard SI prefixes from yocto (10⁻²⁴) to yotta (10²⁴).

# %%
# Power units with different prefixes
power_units = ["TW", "GW", "MW", "kW", "W", "mW"]
w = Unit("W")

print("Power unit conversions to Watts:")
for unit_str in power_units:
    u = Unit(unit_str)
    factor = u.conversion_factor(w)
    print(f"  1 {unit_str:3} = {factor:>15.0e} W")

# %% [markdown]
# ## 8. Error Handling
#
# Attempting to convert between incompatible units raises a `ValueError`.

# %%
gtc = Unit("GtC/yr")
flux = Unit("W/m^2")

print(f"Attempting to convert {gtc} to {flux}...")
print(f"Compatible: {gtc.is_compatible(flux)}")

try:
    factor = gtc.conversion_factor(flux)
except ValueError as e:
    print(f"\nError: {e}")

# %% [markdown]
# ## 9. Integration with Model Building
#
# In RSCM, units are validated at model build time. When a component declares that it expects inputs in a certain unit, the model builder automatically:
#
# 1. **Validates compatibility** - Ensures the schema unit and component unit have the same dimension
# 2. **Calculates conversion factors** - Computes the multiplier needed at runtime
# 3. **Applies conversion at runtime** - Automatically converts values when components read inputs
#
# This means you can have:
# - Schema storing data in `GtC/yr`
# - Component requesting data in `MtCO2/yr`
#
# And the framework handles the conversion transparently.

# %% [markdown]
# ## 10. Summary
#
# The RSCM units module provides:
#
# - **Flexible parsing** of unit strings with various notations
# - **Automatic normalisation** for consistent comparison
# - **Dimensional analysis** to prevent invalid conversions
# - **Conversion factor calculation** including molecular weight ratios
# - **Climate-specific units** ready to use
#
# Key methods:
#
# | Method | Description |
# |--------|-------------|
# | `Unit(unit_str)` | Parse a unit string |
# | `unit.normalized()` | Get the normalised string representation |
# | `unit.is_dimensionless()` | Check if unit is dimensionless |
# | `unit.is_compatible(other)` | Check if conversion is possible |
# | `unit.conversion_factor(other)` | Get the conversion multiplier |
# | `unit.convert(value, other)` | Convert a value to another unit |

# %%
# Final example: complete workflow
print("Complete workflow example:")
print("=" * 50)

# 1. Parse units
emissions_unit = Unit("GtCO2/yr")
model_unit = Unit("GtC/yr")

print(f"\n1. Input unit:  {emissions_unit}")
print(f"   Model unit:  {model_unit}")

# 2. Check compatibility
print(f"\n2. Compatible: {emissions_unit.is_compatible(model_unit)}")

# 3. Get conversion factor
factor = emissions_unit.conversion_factor(model_unit)
print(f"\n3. Conversion factor: {factor:.6f}")

# 4. Convert a value
input_value = 40.0  # 40 GtCO2/yr
converted = emissions_unit.convert(input_value, model_unit)
print(f"\n4. {input_value} GtCO2/yr = {converted:.2f} GtC/yr")

print("\n" + "=" * 50)
print("Done!")

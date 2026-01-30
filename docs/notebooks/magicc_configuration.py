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
# # MAGICC Component Configuration
#
# This notebook demonstrates the MAGICC (Model for the Assessment of Greenhouse gas
# Induced Climate Change) components available in RSCM.
#
# ## Overview
#
# RSCM includes Rust implementations of MAGICC's core physical processes, organised
# into four domains:
#
# - **Climate**: Temperature response to radiative forcing
# - **Chemistry**: Atmospheric lifetimes and concentrations of greenhouse gases
# - **Carbon**: Terrestrial and ocean carbon cycle
# - **Forcing**: Radiative forcing from aerosols and ozone
#
# ## Related Resources
#
# - [Coupled Models](coupled_model.py): Building models with multiple components
# - [Key Concepts](../key_concepts.md): Core RSCM architecture
# - [Components in Rust](component_rust.md): How components are implemented

# %%
from rscm.magicc import (
    AerosolDirectBuilder,
    AerosolIndirectBuilder,
    CH4ChemistryBuilder,
    ClimateUDEBBuilder,
    CO2BudgetBuilder,
    HalocarbonChemistryBuilder,
    N2OChemistryBuilder,
    OceanCarbonBuilder,
    OzoneForcingBuilder,
    TerrestrialCarbonBuilder,
)

# %% [markdown]
# ## Climate Components
#
# ### ClimateUDEB
#
# The 4-box Upwelling-Diffusion Energy Balance (UDEB) model couples a 4-region
# atmosphere (Northern Ocean, Northern Land, Southern Ocean, Southern Land) to a
# multi-layer ocean with vertical diffusion and upwelling.

# %%
climate_params = {
    # Ocean structure
    "n_layers": 50,
    "mixed_layer_depth": 60.0,  # m
    "layer_thickness": 100.0,  # m
    # Diffusivity
    "kappa": 0.75,  # cm^2/s
    "kappa_min": 0.1,
    "kappa_dkdt": -0.191,
    # Upwelling
    "w_initial": 3.5,  # m/yr
    "w_variable_fraction": 0.7,
    "w_threshold_temp_nh": 8.0,  # K
    "w_threshold_temp_sh": 8.0,
    # Climate sensitivity
    "ecs": 3.0,  # K (equilibrium climate sensitivity)
    "rf_2xco2": 3.71,  # W/m^2
    "rlo": 1.317,  # land-ocean warming ratio
    # Heat exchange
    "k_lo": 1.44,  # W/m^2/K
    "k_ns": 0.31,  # W/m^2/K
    "amplify_ocean_to_land": 1.02,
    # Area fractions
    "nh_land_fraction": 0.42,
    "sh_land_fraction": 0.21,
    # Temperature adjustment
    "temp_adjust_alpha": 1.04,
    "temp_adjust_gamma": -0.002,
    # Polar sinking
    "polar_sinking_ratio": 0.2,
    # Integration
    "steps_per_year": 12,
    "max_temperature": 25.0,  # K
}

climate_component = ClimateUDEBBuilder.from_parameters(climate_params).build()
print("ClimateUDEB component created")
print(f"  Inputs: {climate_component.input_names()}")
print(f"  Outputs: {climate_component.output_names()}")

# %% [markdown]
# ## Chemistry Components
#
# ### CH4 Chemistry
#
# Methane atmospheric chemistry using the Prather iteration method with concentration-
# dependent lifetime, OH feedback, and temperature sensitivity.

# %%
ch4_params = {
    "ch4_pi": 722.0,  # ppb, pre-industrial concentration
    "natural_emissions": 209.0,  # Tg CH4/yr
    "tau_oh": 9.3,  # years, tropospheric OH sink lifetime
    "tau_soil": 150.0,  # years, soil uptake lifetime
    "tau_strat": 120.0,  # years, stratospheric sink lifetime
    "tau_trop_cl": 200.0,  # years, tropospheric Cl sink lifetime
    "ch4_self_feedback": -0.32,  # CH4 self-feedback coefficient
    "oh_sensitivity_scale": 0.72,  # OH sensitivity scaling
    "oh_nox_sensitivity": 0.0042,  # (Tg N/yr)^-1
    "oh_co_sensitivity": -0.000105,  # (Tg CO/yr)^-1
    "oh_nmvoc_sensitivity": -0.000315,  # (Tg NMVOC/yr)^-1
    "temp_sensitivity": 0.0316,  # K^-1
    "include_temp_feedback": True,
    "include_emissions_feedback": True,
    "ppb_to_tg": 2.75,  # conversion factor
    "nox_reference": 0.0,  # Tg N/yr
    "co_reference": 0.0,  # Tg CO/yr
    "nmvoc_reference": 0.0,  # Tg NMVOC/yr
}

ch4_component = CH4ChemistryBuilder.from_parameters(ch4_params).build()
print("CH4Chemistry component created")
print(f"  Inputs: {ch4_component.input_names()}")
print(f"  Outputs: {ch4_component.output_names()}")

# %% [markdown]
# ### N2O Chemistry
#
# Nitrous oxide atmospheric chemistry with concentration-dependent lifetime feedback
# and stratospheric transport delay.

# %%
n2o_params = {
    "n2o_pi": 270.0,  # ppb, pre-industrial concentration
    "natural_emissions": 11.0,  # Tg N/yr
    "tau_n2o": 139.275,  # years, base atmospheric lifetime
    "lifetime_feedback": -0.04,  # lifetime feedback exponent
    "strat_delay": 1,  # years, stratospheric mixing delay
    "ppb_to_tg": 4.79,  # conversion factor
}

n2o_component = N2OChemistryBuilder.from_parameters(n2o_params).build()
print("N2OChemistry component created")
print(f"  Inputs: {n2o_component.input_names()}")
print(f"  Outputs: {n2o_component.output_names()}")

# %% [markdown]
# ### Halocarbon Chemistry
#
# F-gases and Montreal Protocol gases with exponential decay.
# Includes CFCs, HCFCs, HFCs, and other halogenated compounds.
#
# The full MAGICC implementation tracks 41 halocarbon species (23 F-gases and
# 18 Montreal Protocol gases) with detailed lifetimes, radiative efficiencies,
# and ozone depletion properties.
#
# For this demonstration, we show a simplified example with just a few key species.
# Each species requires: name, lifetime, radiative efficiency, pre-industrial
# concentration, molecular weight, chlorine atoms, bromine atoms, and fractional
# release factor.

# %%
halocarbon_params = {
    # Simplified F-gas list (2 species as example)
    "fgases": [
        {
            "name": "HFC-134a",
            "lifetime": 14.0,  # years
            "radiative_efficiency": 0.16,  # W/m^2/ppb
            "concentration_pi": 0.0,  # ppt
            "molecular_weight": 102.0,  # g/mol
            "n_cl": 0,  # no chlorine
            "n_br": 0,  # no bromine
            "fractional_release": 0.0,  # no stratospheric release
        },
        {
            "name": "SF6",
            "lifetime": 850.0,
            "radiative_efficiency": 0.57,
            "concentration_pi": 0.0,
            "molecular_weight": 146.0,
            "n_cl": 0,
            "n_br": 0,
            "fractional_release": 0.0,
        },
    ],
    # Simplified Montreal gas list (2 species as example)
    "montreal_gases": [
        {
            "name": "CFC-11",
            "lifetime": 52.0,
            "radiative_efficiency": 0.295,
            "concentration_pi": 0.0,
            "molecular_weight": 137.4,
            "n_cl": 3,  # 3 chlorine atoms
            "n_br": 0,
            "fractional_release": 0.47,  # stratospheric release factor
        },
        {
            "name": "CFC-12",
            "lifetime": 102.0,
            "radiative_efficiency": 0.364,
            "concentration_pi": 0.0,
            "molecular_weight": 120.9,
            "n_cl": 2,
            "n_br": 0,
            "fractional_release": 0.23,
        },
    ],
    "br_multiplier": 60.0,  # bromine efficiency for EESC
    "cfc11_release_normalisation": 0.47,  # CFC-11 fractional release reference
    "eesc_delay": 3.0,  # years, stratospheric mixing delay
    "air_molar_mass": 28.97,  # g/mol
    "atmospheric_mass_tg": 5.133e9,  # Tg
    "mixing_box_fraction": 0.949,  # effective mixing box fraction
}

halocarbon_component = HalocarbonChemistryBuilder.from_parameters(
    halocarbon_params
).build()
print("HalocarbonChemistry component created")
print(f"  Species tracked: {len(halocarbon_component.input_names())} emissions inputs")
print(f"  Example inputs: {halocarbon_component.input_names()[:4]}")
print(f"  Outputs: {halocarbon_component.output_names()}")

# %% [markdown]
# ## Carbon Cycle Components
#
# ### Terrestrial Carbon
#
# 4-pool terrestrial carbon cycle with CO2 fertilisation and temperature feedbacks.
# Pools: Plant, Detritus, Soil, Humus.

# %%
terrestrial_params = {
    "npp_pi": 66.27,  # GtC/yr, pre-industrial Net Primary Production
    "co2_pi": 278.0,  # ppm, pre-industrial CO2
    "beta": 0.6486,  # CO2 fertilisation factor
    "npp_temp_sensitivity": 0.0107,  # K^-1
    "resp_temp_sensitivity": 0.0685,  # K^-1
    "detritus_temp_sensitivity": 0.1358,  # K^-1
    "soil_temp_sensitivity": 0.1541,  # K^-1
    "humus_temp_sensitivity": 0.05,  # K^-1
    # Initial pool sizes (GtC)
    "plant_pool_pi": 884.86,
    "detritus_pool_pi": 92.77,
    "soil_pool_pi": 1681.53,
    "humus_pool_pi": 836.0,
    "respiration_pi": 12.26,  # GtC/yr
    # Transfer fractions
    "frac_npp_to_plant": 0.4483,
    "frac_npp_to_detritus": 0.3998,
    "frac_plant_to_detritus": 0.9989,
    "frac_detritus_to_soil": 0.3,
    "frac_soil_to_humus": 0.1,
    # Feedbacks
    "enable_fertilization": True,
    "enable_temp_feedback": True,
}

terrestrial_component = TerrestrialCarbonBuilder.from_parameters(
    terrestrial_params
).build()
print("TerrestrialCarbon component created")
print(f"  Inputs: {terrestrial_component.input_names()}")
print(f"  Outputs: {terrestrial_component.output_names()}")

# %% [markdown]
# ### Ocean Carbon
#
# IRF-based ocean carbon uptake using the 2D-BERN impulse response function
# with air-sea exchange and temperature feedback.
#
# The ocean component uses sophisticated impulse response functions with
# 6-term exponential decay to represent ocean transport. The IRF switches
# from "early" to "late" coefficients after 9.9 years.
#
# For simplicity, we use default IRF coefficients and only customize the
# high-level physical parameters.

# %%
ocean_params = {
    "co2_pi": 278.0,  # ppm, pre-industrial atmospheric CO2
    "pco2_pi": 278.0,  # ppm, pre-industrial ocean surface pCO2
    "gas_exchange_scale": 1.833492,  # gas exchange rate scaling
    "gas_exchange_tau": 7.46,  # years, gas exchange timescale
    "temp_sensitivity": 0.0423,  # K^-1, pCO2 temperature sensitivity
    "irf_scale": 0.9492864,  # IRF scaling factor
    "mixed_layer_depth": 50.0,  # m
    "ocean_surface_area": 3.5375e14,  # m^2
    "sst_pi": 18.2997,  # degC, pre-industrial sea surface temperature
    "steps_per_year": 12,  # sub-steps per year
    "max_history_months": 6000,  # maximum flux history length
    "irf_switch_time": 9.9,  # years, switch between early/late IRF
    # IRF exponential coefficients (2D-BERN model defaults)
    "irf_early_coefficients": [
        0.058648,
        0.07515,
        0.079338,
        0.41413,
        0.24845,
        0.12429,
    ],
    "irf_early_timescales": [1.0e10, 9.6218, 9.2364, 0.7603, 0.16294, 0.0032825],
    "irf_late_coefficients": [
        0.01369,
        0.012456,
        0.026933,
        0.026994,
        0.036608,
        0.06738,
    ],
    "irf_late_timescales": [1.0e10, 331.54, 107.57, 38.946, 11.677, 10.515],
    # Joos A24 polynomial for pCO2-DIC relationship
    "delta_ospp_offsets": [1.5568, 7.4706, 1.2748, 2.4491, 1.5468],
    "delta_ospp_coefficients": [-0.013993, -0.20207, -0.12015, -0.12639, -0.15326],
    "enable_temp_feedback": True,
}

ocean_component = OceanCarbonBuilder.from_parameters(ocean_params).build()
print("OceanCarbon component created")
print(f"  Inputs: {ocean_component.input_names()}")
print(f"  Outputs: {ocean_component.output_names()}")

# %% [markdown]
# ### CO2 Budget
#
# Mass balance integrator that closes the carbon cycle by calculating atmospheric
# CO2 concentration change from emissions and uptakes.

# %%
budget_params = {
    "gtc_per_ppm": 2.123,  # GtC per ppm CO2 conversion factor
    "co2_pi": 278.0,  # ppm, pre-industrial CO2
}

budget_component = CO2BudgetBuilder.from_parameters(budget_params).build()
print("CO2Budget component created")
print(f"  Inputs: {budget_component.input_names()}")
print(f"  Outputs: {budget_component.output_names()}")

# %% [markdown]
# ## Forcing Components
#
# ### Ozone Forcing
#
# Stratospheric and tropospheric ozone radiative forcing, including EESC-driven
# stratospheric ozone depletion and tropospheric ozone production from CH4 and
# NOx/CO/NMVOC emissions.

# %%
ozone_params = {
    "eesc_reference": 1420.0,  # ppt, reference EESC for ozone depletion
    "strat_o3_scale": -0.0043,  # W/m^2, stratospheric ozone forcing scale
    "strat_cl_exponent": 1.7,  # power-law exponent for EESC-RF relationship
    "trop_radeff": 0.032,  # W/m^2/DU, tropospheric radiative efficiency
    "trop_oz_ch4": 5.7,  # DU per ln(CH4/CH4_pi)
    "trop_oz_nox": 0.168,  # DU/(Mt N/yr), NOx sensitivity
    "trop_oz_co": 0.00396,  # DU/(Mt CO/yr), CO sensitivity
    "trop_oz_voc": 0.01008,  # DU/(Mt NMVOC/yr), NMVOC sensitivity
    "ch4_pi": 700.0,  # ppb, pre-industrial CH4
    "nox_pi": 0.0,  # Mt N/yr, pre-industrial NOx
    "co_pi": 0.0,  # Mt CO/yr, pre-industrial CO
    "nmvoc_pi": 0.0,  # Mt NMVOC/yr, pre-industrial NMVOC
    "temp_feedback_scale": -0.037,  # W/m^2/K, temperature feedback coefficient
}

ozone_component = OzoneForcingBuilder.from_parameters(ozone_params).build()
print("OzoneForcing component created")
print(f"  Inputs: {ozone_component.input_names()}")
print(f"  Outputs: {ozone_component.output_names()}")

# %% [markdown]
# ### Aerosol Direct
#
# Direct aerosol radiative effects from SOx, BC, OC, and NOx with FourBox regional
# distribution.

# %%
aerosol_direct_params = {
    "sox_coefficient": -0.0035,  # W/m^2 per Tg S/yr
    "bc_coefficient": 0.0077,  # W/m^2 per Tg BC/yr
    "oc_coefficient": -0.002,  # W/m^2 per Tg OC/yr
    "nitrate_coefficient": -0.001,  # W/m^2 per Tg N/yr
    # Regional distribution (NH Ocean, NH Land, SH Ocean, SH Land)
    "sox_regional": [0.15, 0.55, 0.10, 0.20],
    "bc_regional": [0.15, 0.50, 0.15, 0.20],
    "oc_regional": [0.15, 0.45, 0.15, 0.25],
    "nitrate_regional": [0.15, 0.50, 0.15, 0.20],
    # Pre-industrial baselines
    "sox_pi": 1.0,  # Tg S/yr
    "bc_pi": 2.5,  # Tg BC/yr
    "oc_pi": 10.0,  # Tg OC/yr
    "nox_pi": 10.0,  # Tg N/yr
    # Harmonisation (optional)
    "harmonize": False,
    "harmonize_year": 2019.0,
    "harmonize_target": -0.22,  # W/m^2
}

aerosol_direct_component = AerosolDirectBuilder.from_parameters(
    aerosol_direct_params
).build()
print("AerosolDirect component created")
print(f"  Inputs: {aerosol_direct_component.input_names()}")
print(f"  Outputs: {aerosol_direct_component.output_names()}")

# %% [markdown]
# ### Aerosol Indirect
#
# Indirect aerosol effects on cloud albedo using a logarithmic relationship
# with aerosol burden.

# %%
aerosol_indirect_params = {
    "cloud_albedo_coefficient": -1.0,  # W/m^2/ln-unit
    "reference_burden": 50.0,  # Tg/yr
    "sox_weight": 1.0,  # SOx weight in aerosol burden
    "oc_weight": 0.3,  # OC weight in aerosol burden
    "sox_pi": 1.0,  # Tg S/yr
    "oc_pi": 10.0,  # Tg OC/yr
    # Harmonisation (optional)
    "harmonize": False,
    "harmonize_year": 2019.0,
    "harmonize_target": -0.89,  # W/m^2
}

aerosol_indirect_component = AerosolIndirectBuilder.from_parameters(
    aerosol_indirect_params
).build()
print("AerosolIndirect component created")
print(f"  Inputs: {aerosol_indirect_component.input_names()}")
print(f"  Outputs: {aerosol_indirect_component.output_names()}")

# %% [markdown]
# ## Summary
#
# The MAGICC components available in RSCM:
#
# | Domain | Component | Description |
# |--------|-----------|-------------|
# | Climate | `ClimateUDEBBuilder` | 4-box UDEB model with 50-layer ocean |
# | Chemistry | `CH4ChemistryBuilder` | CH4 with Prather iteration |
# | Chemistry | `N2OChemistryBuilder` | N2O with stratospheric delay |
# | Chemistry | `HalocarbonChemistryBuilder` | CFCs, HCFCs, HFCs (41 species) |
# | Carbon | `TerrestrialCarbonBuilder` | 4-pool land carbon cycle |
# | Carbon | `OceanCarbonBuilder` | IRF-based ocean uptake |
# | Carbon | `CO2BudgetBuilder` | Carbon mass balance |
# | Forcing | `OzoneForcingBuilder` | Stratospheric + tropospheric O3 |
# | Forcing | `AerosolDirectBuilder` | Direct aerosol effects (FourBox) |
# | Forcing | `AerosolIndirectBuilder` | Cloud albedo effects |
#
# ## Next Steps
#
# To build a coupled MAGICC-style model:
#
# 1. Select the components you need
# 2. Use `ModelBuilder` to connect them (see [Coupled Models](coupled_model.py))
# 3. Provide exogenous emissions/concentrations
# 4. Set initial values for state variables
# 5. Run the model and extract results

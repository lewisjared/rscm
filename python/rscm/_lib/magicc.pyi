"""MAGICC climate model components.

This module provides Rust implementations of MAGICC (Model for the Assessment of
Greenhouse gas Induced Climate Change) component builders.

MAGICC is a reduced-complexity climate model widely used for climate policy
analysis. The components in this module implement MAGICC's key physical processes.

Available Components
--------------------
Climate:
- ClimateUDEBBuilder: 4-box UDEB climate model with ocean diffusion
- SeaLevelRiseBuilder: Sea level rise from glaciers and ice sheets

Chemistry:
- CH4ChemistryBuilder: CH4 atmospheric chemistry with Prather iteration
- N2OChemistryBuilder: N2O atmospheric chemistry with stratospheric delay
- HalocarbonChemistryBuilder: F-gases and Montreal Protocol gases

Carbon Cycle:
- TerrestrialCarbonBuilder: 3-pool terrestrial carbon cycle (MAGICC7)
- OceanCarbonBuilder: IRF-based ocean carbon uptake
- CO2BudgetBuilder: Carbon mass balance integrator
- PermafrostBuilder: Permafrost thaw releasing CO2 and CH4

Forcing:
- OzoneForcingBuilder: Stratospheric and tropospheric ozone forcing
- AerosolDirectBuilder: Direct aerosol radiative effects
- AerosolIndirectBuilder: Indirect aerosol cloud effects
"""

from typing import Any, Literal, TypeAlias, TypedDict, final

from rscm._lib.core import Component, ComponentBuilder

# Climate Components

class ClimateUDEBParams(TypedDict, total=False):
    """Parameters for ClimateUDEB component.

    All fields optional - defaults provided by Rust.
    """

    n_layers: int
    mixed_layer_depth: float
    layer_thickness: float
    kappa: float
    kappa_min: float
    kappa_dkdt: float
    w_initial: float
    w_variable_fraction: float
    w_threshold_temp_nh: float
    w_threshold_temp_sh: float
    ecs: float
    rf_2xco2: float
    rlo: float
    k_lo: float
    k_ns: float
    amplify_ocean_to_land: float
    nh_land_fraction: float
    sh_land_fraction: float
    efficacy_apply: int
    prescribed_efficacy_co2: float
    regional_forcing: bool
    time_varying_q2x: bool
    temp_adjust_alpha: float
    temp_adjust_gamma: float
    polar_sinking_ratio: float
    steps_per_year: int
    max_temperature: float
    thermal_expansion: ThermalExpansionParams | None

@final
class ClimateUDEBBuilder(ComponentBuilder):
    """Builder for the 4-box UDEB climate model component.

    The UDEB (Upwelling-Diffusion Energy Balance) model couples a 4-box atmosphere
    (Northern Ocean, Northern Land, Southern Ocean, Southern Land) to a multi-layer
    ocean with vertical diffusion and upwelling.

    # Parameters
    # ----------
    # n_layers : int
    #     Number of ocean layers (including mixed layer). Default: 50
    # mixed_layer_depth : float
    #     Mixed layer depth (m). Default: 60.0
    # layer_thickness : float
    #     Layer thickness for deeper ocean layers (m). Default: 100.0
    # kappa : float
    #     Base vertical diffusivity (cm^2/s). Default: 0.75
    # kappa_min : float
    #     Minimum vertical diffusivity (cm^2/s). Default: 0.1
    # kappa_dkdt : float
    #     Temperature gradient coefficient for diffusivity (cm^2/s/K). Default: -0.191
    # w_initial : float
    #     Initial upwelling rate (m/yr). Default: 3.5
    # w_variable_fraction : float
    #     Variable fraction of upwelling. Default: 0.7
    # w_threshold_temp_nh : float
    #     Temperature threshold for NH upwelling shutdown (K). Default: 8.0
    # w_threshold_temp_sh : float
    #     Temperature threshold for SH upwelling shutdown (K). Default: 8.0
    # ecs : float
    #     Equilibrium climate sensitivity (K). Default: 3.0
    # rf_2xco2 : float
    #     Radiative forcing for 2xCO2 (W/m^2). Default: 3.71
    # rlo : float
    #     Land-ocean warming ratio. Default: 1.317
    # k_lo : float
    #     Land-ocean heat exchange coefficient (W/m^2/K). Default: 1.44
    # k_ns : float
    #     Inter-hemispheric heat exchange coefficient (W/m^2/K). Default: 0.31
    # amplify_ocean_to_land : float
    #     Ocean-to-land heat exchange amplification factor. Default: 1.02
    # nh_land_fraction : float
    #     Northern Hemisphere land fraction. Default: 0.42
    # sh_land_fraction : float
    #     Southern Hemisphere land fraction. Default: 0.21
    # temp_adjust_alpha : float
    #     Ocean-to-atmosphere temperature adjustment alpha. Default: 1.04
    # temp_adjust_gamma : float
    #     Ocean-to-atmosphere temperature adjustment gamma (1/K). Default: -0.002
    # polar_sinking_ratio : float
    #     Polar sinking water temperature ratio. Default: 0.2
    # steps_per_year : int
    #     Steps per year for sub-annual integration. Default: 12
    # max_temperature : float
    #     Maximum temperature anomaly cap (K). Default: 25.0

    Inputs
    ------
    Effective Radiative Forcing|Total : float
        Total effective radiative forcing (W/m^2, FourBox grid)

    States
    ------
    Surface Temperature : float
        Regional surface temperature anomaly (K, FourBox grid)

    Outputs
    -------
    Heat Uptake : float
        Heat uptake (W/m^2)
    Ocean Heat Content : float
        Total ocean heat uptake (J)

    Examples
    --------
    >>> builder = ClimateUDEBBuilder.from_parameters(
    ...     {
    ...         "ecs": 3.0,
    ...         "rf_2xco2": 3.71,
    ...         "mixed_layer_depth": 60.0,
    ...     }
    ... )
    >>> component = builder.build()
    """

    def initial_values(self, erf: float) -> dict[str, float]:
        """Return initial state and diagnostics at zero temperature anomaly."""
    def internal_efficacy(self, rf_regions: tuple[float, float, float, float]) -> float:
        """Return the internal efficacy of an agent with this regional pattern."""
    @staticmethod
    def from_parameters(parameters: ClimateUDEBParams) -> ClimateUDEBBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the UDEB climate model component."""

class ThermalExpansionParams(TypedDict, total=False):
    """Thermal expansion of the ocean, calculated by ClimateUDEB.

    Setting it makes ClimateUDEB write ``Surface Temperature|Annual Mean`` (K)
    and ``Sea Level Rise|Thermal Expansion`` (mm).
    """

    coefficients: tuple[float, float, float, float, float, float]
    scaling: float
    start_year: float

class GlacierParams(TypedDict, total=False):
    """Mountain glaciers and small ice caps."""

    sensitivity: float
    exponent: float
    start_year: float
    equilibrium_temperature: list[float]
    equilibrium_sea_level: list[float]

class SurfaceMassBalanceParams(TypedDict, total=False):
    """Surface mass balance of an ice sheet."""

    sensitivity: float
    linear_fraction: float
    exponent: float
    start_year: float
    volume: float | None
    volume_exponent: float

class DischargeCaseParams(TypedDict, total=False):
    """One discharge case of the Greenland ice sheet."""

    volume: float
    sensitivity: float
    exponent: float

class GreenlandDischargeParams(TypedDict, total=False):
    """Greenland solid ice discharge, between a low and a high case."""

    low: DischargeCaseParams
    high: DischargeCaseParams
    case: float
    scaling: float
    start_year: float

class AntarcticRegionParams(TypedDict):
    """One Antarctic drainage region."""

    name: str
    delay: int
    temperature_scaling: float
    response: tuple[float, float, float, float, float]

class AntarcticDischargeParams(TypedDict, total=False):
    """Antarctic solid ice discharge (Levermann et al. 2014)."""

    basal_melt_sensitivity: float
    response_span: int
    scaling: float
    start_year: float
    regions: list[AntarcticRegionParams]

class SeaLevelRiseParams(TypedDict, total=False):
    """Parameters for the sea level rise component.

    All fields optional - defaults are MAGICC7's effective defaults.
    Fields missing from a nested group keep that group's defaults.
    """

    glaciers: GlacierParams
    greenland_smb: SurfaceMassBalanceParams
    greenland_discharge: GreenlandDischargeParams
    antarctic_smb: SurfaceMassBalanceParams
    antarctic_discharge: AntarcticDischargeParams

@final
class SeaLevelRiseBuilder(ComponentBuilder):
    """Builder for the sea level rise component (MAGICC7 module 14).

    Glaciers, Greenland and Antarctic surface mass balance and solid ice discharge
    respond to the annual mean warming. Thermal expansion comes from ClimateUDEB
    with ``thermal_expansion`` set. Steps must be one year long.

    Inputs
    ------
    Surface Temperature|Annual Mean : float
        Mean global surface warming over the step (K)
    Sea Level Rise|Thermal Expansion : float
        Thermal expansion of the ocean (mm)

    Outputs
    -------
    Sea Level Rise : float
        Sum of all contributions (mm)
    Sea Level Rise|Glaciers : float
        Glaciers and small ice caps (mm)
    Sea Level Rise|Greenland|Surface Mass Balance : float
        Greenland surface mass balance (mm)
    Sea Level Rise|Greenland|Solid Ice Discharge : float
        Greenland solid ice discharge (mm)
    Sea Level Rise|Antarctica|Surface Mass Balance : float
        Antarctic surface mass balance (mm)
    Sea Level Rise|Antarctica|Solid Ice Discharge : float
        Antarctic solid ice discharge (mm)

    Examples
    --------
    >>> builder = SeaLevelRiseBuilder.from_parameters(
    ...     {"glaciers": {"start_year": 1850}}
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: SeaLevelRiseParams) -> SeaLevelRiseBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the sea level rise component."""

# Chemistry Components

class CH4ChemistryParams(TypedDict, total=False):
    """Parameters for CH4 chemistry component.

    All fields optional - defaults provided by Rust.
    """

    feedback_start_year: float
    natural_emissions: float
    permafrost: bool
    tau_total_init: float
    tau_soil: float
    tau_strat: float
    tau_trop_cl: float
    ch4_self_feedback: float
    oh_sensitivity_scale: float
    oh_nox_sensitivity: float
    oh_co_sensitivity: float
    oh_nmvoc_sensitivity: float
    temp_sensitivity: float
    include_temp_feedback: bool
    include_emissions_feedback: bool
    concentration_switch_year: float | None
    ppb_to_tg: float
    mixbox_size: float

@final
class CH4ChemistryBuilder(ComponentBuilder):
    """Builder for the CH4 chemistry component.

    Implements atmospheric methane chemistry with the Prather iteration method,
    including OH feedback and temperature sensitivity. Defaults follow MAGICC7
    v7.5.3. Concentration and temperature feedbacks are measured from the year
    before ``feedback_start_year``; NOx, CO and NMVOC deltas are relative to
    emissions in ``feedback_start_year``.

    # Parameters
    # ----------
    # feedback_start_year : float
    #     Year from which feedbacks are measured. Default: 1927.0
    # natural_emissions : float
    #     Natural CH4 emissions (Tg CH4/yr). Default: 182.1908
    # tau_total_init : float
    #     Total lifetime at the reference state (years). Default: 9.9474
    # tau_soil : float
    #     Soil uptake sink lifetime (years). Default: 150.0
    # tau_strat : float
    #     Stratospheric sink lifetime (years). Default: 120.0
    # tau_trop_cl : float
    #     Tropospheric Cl sink lifetime (years). Default: 200.0
    # ch4_self_feedback : float
    #     CH4 self-feedback coefficient. Default: -0.53775
    # oh_sensitivity_scale : float
    #     OH sensitivity scaling factor. Default: 0.72448
    # oh_nox_sensitivity : float
    #     NOx emissions effect on OH ((Tg N/yr)^-1). Default: 0.0093376
    # oh_co_sensitivity : float
    #     CO emissions effect on OH ((Tg CO/yr)^-1). Default: -0.000113
    # oh_nmvoc_sensitivity : float
    #     NMVOC emissions effect on OH ((Tg NMVOC/yr)^-1). Default: -0.0003142
    # temp_sensitivity : float
    #     Temperature sensitivity of lifetime (K^-1). Default: 0.07
    # include_temp_feedback : bool
    #     Enable temperature feedback. Default: True
    # include_emissions_feedback : bool
    #     Enable NOx/CO/NMVOC emissions feedback. Default: True
    # ppb_to_tg : float
    #     Conversion factor ppb to Tg CH4. Default: 2.824
    # mixbox_size : float
    #     Fraction of the atmosphere mixed with surface CH4. Default: 0.973

    Inputs
    ------
    Emissions|CH4 : float
        Anthropogenic CH4 emissions (Tg CH4/yr)
    Surface Temperature|Global : float
        Global mean surface temperature anomaly (K)
    Emissions|NOx : float
        NOx emissions (Tg N/yr)
    Emissions|CO : float
        CO emissions (Tg CO/yr)
    Emissions|NMVOC : float
        NMVOC emissions (Tg NMVOC/yr)

    States
    ------
    Atmospheric Concentration|CH4 : float
        Atmospheric CH4 concentration (ppb)

    Outputs
    -------
    Lifetime|CH4 : float
        CH4 atmospheric lifetime (years)
    Lifetime|CH4|Effective : float
        Start-of-year burden over the CH4 removed during the step (years)

    Examples
    --------
    >>> builder = CH4ChemistryBuilder.from_parameters(
    ...     {
    ...         "tau_total_init": 9.9474,
    ...         "ch4_self_feedback": -0.53775,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: CH4ChemistryParams) -> CH4ChemistryBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the CH4 chemistry component."""

class CH4OxidationParams(TypedDict, total=False):
    """Parameters for the fossil CH4 oxidation component.

    All fields optional - defaults provided by Rust.
    """

    carbon_per_ch4: float
    permafrost: bool
    permafrost_fraction: float
    ppb_to_tg: float
    mixbox_size: float
    tau_other: float
    natural_emissions: float

@final
class CH4OxidationBuilder(ComponentBuilder):
    """Builder for the fossil CH4 oxidation component.

    Tracks atmospheric CH4 by source and the CO2 released as its fossil share
    is oxidised, which MAGICC7 adds to fossil CO2 emissions. The CH4 budget
    parameters must match the CH4 chemistry component, and it must be registered
    after it so that it reads the same step's concentration and lifetime.

    # Parameters
    # ----------
    # carbon_per_ch4 : float
    #     Mass of carbon per mass of CH4 oxidised. Default: 0.75

    Inputs
    ------
    Atmospheric Concentration|CH4 : float
        CH4 concentration at the start and end of the step (ppb)
    Lifetime|CH4|OH : float
        OH lifetime over the step (years)
    Emissions|CH4|Fossil, Emissions|CH4|AFOLU : float
        Anthropogenic CH4 emissions by sector (Mt CH4/yr)

    States
    ------
    Atmospheric Burden|CH4|Fossil, Atmospheric Burden|CH4|AFOLU : float
        Anthropogenic CH4 in the atmosphere (Mt CH4)
    Atmospheric Burden|CH4|Natural, Atmospheric Burden|CH4|Permafrost : float
        Natural and permafrost CH4 in the atmosphere (Mt CH4)

    Outputs
    -------
    Emissions|CO2|CH4 Oxidation : float
        CO2 released by oxidation (GtC/yr)
    """

    @staticmethod
    def from_parameters(parameters: CH4OxidationParams) -> CH4OxidationBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the CH4 oxidation component."""

class N2OChemistryParams(TypedDict, total=False):
    """Parameters for N2O chemistry component.

    All fields optional - defaults provided by Rust.
    """

    feedback_start_year: float
    natural_emissions: float
    tau_n2o: float
    lifetime_feedback: float
    strat_delay: int
    ppb_to_tg: float
    include_meridional_flux_feedback: bool
    meridional_flux_change_per_degree: float
    tau_sensitivity_to_meridional_flux: float
    meridional_flux_reference_year: float
    concentration_switch_year: float | None

@final
class N2OChemistryBuilder(ComponentBuilder):
    """Builder for the N2O chemistry component.

    Implements atmospheric nitrous oxide chemistry with concentration-dependent
    lifetime feedback, stratospheric transport delay and the MAGICC7
    meridional flux response to warming. Defaults follow MAGICC7 v7.5.3.

    # Parameters
    # ----------
    # feedback_start_year : float
    #     Year from which the concentration feedback is measured. Default: 1925.0
    # natural_emissions : float
    #     Natural N2O emissions (Tg N/yr). Default: 8.0345017
    # tau_n2o : float
    #     Base atmospheric lifetime (years). Default: 139.275
    # lifetime_feedback : float
    #     Lifetime feedback exponent. Default: -0.04
    # strat_delay : int
    #     Stratospheric mixing delay (years). Default: 1
    # ppb_to_tg : float
    #     Conversion factor ppb to Tg N. Default: 4.79
    # include_meridional_flux_feedback : bool
    #     Shorten the lifetime with warming. Default: True
    # meridional_flux_change_per_degree : float
    #     Meridional flux change per degree of warming (K^-1). Default: 0.15
    # tau_sensitivity_to_meridional_flux : float
    #     Lifetime sensitivity to meridional flux changes. Default: 0.04
    # meridional_flux_reference_year : float
    #     Year from which warming changes the meridional flux. Default: 1980.0

    Inputs
    ------
    Emissions|N2O : float
        Anthropogenic N2O emissions (Tg N/yr)
    Surface Temperature : float
        Global mean surface temperature anomaly (K)

    States
    ------
    Atmospheric Concentration|N2O : float
        Atmospheric N2O concentration (ppb)

    Outputs
    -------
    Lifetime|N2O : float
        N2O atmospheric lifetime (years)

    Examples
    --------
    >>> builder = N2OChemistryBuilder.from_parameters(
    ...     {
    ...         "tau_n2o": 139.275,
    ...         "lifetime_feedback": -0.04,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: N2OChemistryParams) -> N2OChemistryBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the N2O chemistry component."""

class HalocarbonSpecies(TypedDict, total=False):
    """A halocarbon species. Lifetimes in yr, radiative efficiency in W/m²/ppb."""

    name: str
    tau_total: float
    tau_oh: float
    tau_strat: float
    radiative_efficiency: float
    concentration_pi: float
    molecular_mass: float
    n_cl: int
    n_br: int
    fractional_release: float
    mix_box_size: float

class FgasRegions(TypedDict, total=False):
    """Regional pattern of F-gas forcing.

    Species with lifetimes of at least ``upper_tau`` follow ``rf_regions``
    normalised globally. Those below ``lower_tau`` follow it normalised within
    each hemisphere, scaled by each hemisphere's share of fossil CO2 emissions.
    Lifetimes in between mix the two linearly.
    """

    rf_regions: tuple[float, float, float, float]
    lower_tau: float
    upper_tau: float
    nh_land_fraction: float
    sh_land_fraction: float

class HalocarbonChemistryParams(TypedDict, total=False):
    """Parameters for halocarbon chemistry component.

    All fields optional, defaults provided by Rust.
    """

    fgases: list[HalocarbonSpecies]
    montreal_gases: list[HalocarbonSpecies]
    br_vs_cl_scale: float
    cfc11_release_factor: float
    eesc_delay: int
    air_molar_mass: float
    atmosphere_mass: float
    variable_oh_lifetime: bool
    ch4_oh_lifetime_init: float
    variable_strat_lifetime: bool
    meridional_flux_year: float
    meridional_flux_per_degree: float
    strat_lifetime_sensitivity: float
    fgas_switch_year: float | None
    mhalo_switch_year: float | None
    fgas_regions: FgasRegions | None
    """Makes ``Effective Radiative Forcing|F-Gases`` a four-box output."""

@final
class HalocarbonChemistryBuilder(ComponentBuilder):
    """Builder for the halocarbon chemistry component.

    Concentrations, forcing and EESC of F-gases and Montreal Protocol gases,
    following MAGICC7 Module 03.

    Inputs
    ------
    Emissions|{Species} : float
        Emissions for each halocarbon species (kt/yr)
    Lifetime|CH4|OH : float
        CH4 OH lifetime (yr), which scales OH lifetimes
    Surface Temperature : float
        Warming (K), which shortens stratospheric lifetimes
    Emissions|CO2|Fossil : float
        Fossil CO2 emissions (GtC/yr), read with ``fgas_regions``

    States
    ------
    Atmospheric Concentration|{Species} : float
        Concentration for each halocarbon species (ppt)

    Outputs
    -------
    Effective Radiative Forcing|F-Gases : float or FourBox
        F-gas ERF (W/m²), four-box with ``fgas_regions``
    Effective Radiative Forcing|Montreal Protocol Halogen Gases : float
        Montreal Protocol gas ERF (W/m²)
    EESC : float
        Equivalent Effective Stratospheric Chlorine (ppt)

    Examples
    --------
    >>> builder = HalocarbonChemistryBuilder.from_parameters({"eesc_delay": 3})
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(  # type: ignore[override]
        parameters: HalocarbonChemistryParams,
    ) -> HalocarbonChemistryBuilder:
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the halocarbon chemistry component."""

# Carbon Cycle Components

class TerrestrialCarbonParams(TypedDict, total=False):
    """Parameters for terrestrial carbon cycle component.

    All fields optional - defaults provided by Rust.
    """

    npp_pi: float
    co2_pi: float
    beta: float
    fertilization_method: float
    fertilization_factor2: float
    gifford_conc_for_zero_npp: float
    fertilization_yrstart: float
    npp_temp_sensitivity: float
    resp_temp_sensitivity: float
    detritus_temp_sensitivity: float
    soil_temp_sensitivity: float
    tempfeedback_yrstart: float
    plant_pool_pi: float
    detritus_pool_pi: float
    soil_pool_pi: float
    respiration_pi: float
    plantbox_resp_fertscale: float
    frac_npp_to_plant: float
    frac_npp_to_detritus: float
    frac_plant_to_detritus: float
    frac_detritus_to_soil: float
    frac_deforest_plant: float
    frac_deforest_detritus: float
    norgrwth_frac_defo: float
    nitrogen: NitrogenParams | None

class PrognosticFixationParams(TypedDict):
    """Michaelis-Menten nitrogen fixation from NPP."""

    vmax: float
    km: float

class NitrogenParams(TypedDict, total=False):
    """Parameters for the nitrogen limitation of NPP (MAGICC7 module 13).

    All fields optional - defaults provided by Rust.
    """

    ncratio_uptake: float
    ncratio_recycling: float
    recycling_cn_factor: float
    uptake_supply: float
    availability_factor_a: float
    availability_factor_c: float
    loss_turnover_time: float
    prognostic_fixation: PrognosticFixationParams | None

@final
class TerrestrialCarbonBuilder(ComponentBuilder):
    """Builder for the terrestrial carbon cycle component.

    Implements the MAGICC7 3-pool terrestrial carbon model with three CO2
    fertilization methods, two respiration methods, time-varying turnover
    times, no-feedback reference pools, and mass conservation correction.

    Inputs
    ------
    Atmospheric Concentration|CO2 : float
        Atmospheric CO2 concentration (ppm)
    Surface Temperature : float
        Global mean surface temperature anomaly (K). The feedbacks use the
        warming since ``tempfeedback_yrstart``, floored at zero.
    Emissions|CO2|Land Use : float
        Land use change CO2 emissions (GtC/yr)
    Nitrogen Deposition : float
        Nitrogen deposition (Gt N/yr), only with ``nitrogen``
    Nitrogen Fixation : float
        Nitrogen fixation (Gt N/yr), only with ``nitrogen`` and no
        ``prognostic_fixation``

    States
    ------
    Carbon Pool|Plant : float
        Plant carbon pool (GtC)
    Carbon Pool|Detritus : float
        Detritus carbon pool (GtC)
    Carbon Pool|Soil : float
        Soil carbon pool (GtC)

    Outputs
    -------
    Carbon Flux|Terrestrial : float
        Net atmosphere-to-land flux, including land-use emissions (GtC/yr)
    Emissions|CO2|Gross Deforestation : float
        Gross deforestation emissions (GtC/yr)
    Carbon Flux|Regrowth : float
        Regrowth carbon flux (GtC/yr)
    Net Primary Production : float
        NPP with feedbacks (GtC/yr)
    Respiration|Terrestrial : float
        Total terrestrial respiration (GtC/yr)
    Nitrogen Limitation Factor : float
        Factor applied to NPP, only with ``nitrogen``
    Nitrogen Pool : float
        Plant-available nitrogen pool (Gt N), only with ``nitrogen``
    Nitrogen Flux|Uptake, Recycling, Losses, Fixation, Deposition : float
        Nitrogen budget (Gt N/yr), only with ``nitrogen``

    Examples
    --------
    >>> builder = TerrestrialCarbonBuilder.from_parameters(
    ...     {
    ...         "npp_pi": 66.27,
    ...         "beta": 0.6486,
    ...         "fertilization_method": 1.10,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(  # type: ignore[override]
        parameters: TerrestrialCarbonParams,
    ) -> TerrestrialCarbonBuilder:
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the terrestrial carbon cycle component."""

class OceanCarbonParams(TypedDict, total=False):
    """Parameters for ocean carbon cycle component.

    All fields optional - defaults provided by Rust.
    """

    co2_pi: float
    pco2_pi: float | None
    gas_exchange_scale: float
    gas_exchange_tau: float
    temp_sensitivity: float
    irf_scale: float
    mixed_layer_depth: float
    ocean_surface_area: float
    sst_pi: float
    steps_per_year: int
    max_history_months: int
    irf_switch_time: float
    irf_early: dict[str, Any]
    irf_late: dict[str, Any]
    delta_ospp_offsets: list[float]
    delta_ospp_coefficients: list[float]
    enable_temp_feedback: bool
    tempfeedback_yrstart: float
    flux_change_limit: float
    gtc_per_ppm: float
    permafrost: bool
    concentration_switch_year: float | None

@final
class OceanCarbonBuilder(ComponentBuilder):
    """Builder for the ocean carbon cycle component.

    Implements an IRF-based ocean carbon model with air-sea exchange and
    temperature feedback, using the MAGICC7 default 3D-GFDL impulse response.
    Atmospheric CO2 evolves monthly within each year from the other carbon
    sources and the ocean uptake, as in MAGICC7.

    # Parameters
    # ----------
    # pco2_pi : float | None
    #     Unperturbed ocean surface pCO2 (ppm). Default: the initial pCO2
    # gas_exchange_scale : float
    #     Gas exchange rate scaling factor. Default: 1.833492
    # gas_exchange_tau : float
    #     Gas exchange timescale (years). Default: 7.66
    # temp_sensitivity : float
    #     Temperature sensitivity of pCO2 (K^-1). Default: 0.03717879
    # irf_scale : float
    #     IRF scaling factor. Default: 0.9492864
    # mixed_layer_depth : float
    #     Mixed layer depth (m). Default: 50.9
    # ocean_surface_area : float
    #     Ocean surface area (m^2). Default: 3.55e14
    # sst_pi : float
    #     Pre-industrial sea surface temperature (C). Default: 17.7
    # steps_per_year : int
    #     Sub-steps per year. Default: 12
    # max_history_months : int
    #     Maximum flux history length (months). Default: 6000
    # enable_temp_feedback : bool
    #     Enable temperature feedback on pCO2. Default: True
    # tempfeedback_yrstart : float
    #     Year from which warming drives the pCO2 feedback. Default: 1900.0
    # flux_change_limit : float
    #     Largest flux change between sub-steps (ppm/yr), disabled if non-positive.
    #     Default: 0.04
    # gtc_per_ppm : float
    #     Conversion factor GtC per ppm CO2. Default: 2.123

    Inputs
    ------
    Atmospheric Concentration|CO2 : float
        Atmospheric CO2 concentration at the start of the year (ppm)
    Surface Temperature : float
        Global mean surface temperature anomaly (K)
    Emissions|CO2|Fossil : float
        Fossil and industrial CO2 emissions (GtC/yr)
    Emissions|CO2|CH4 Oxidation : float
        CO2 from oxidised fossil CH4 (GtC/yr)
    Carbon Flux|Terrestrial : float
        Net land uptake, including land-use emissions (GtC/yr)

    States
    ------
    Ocean Surface pCO2 : float
        Ocean surface pCO2 (ppm)
    Cumulative Ocean Uptake : float
        Cumulative ocean carbon uptake (GtC)

    Outputs
    -------
    Carbon Flux|Ocean : float
        Annual mean ocean carbon uptake (GtC/yr)

    Examples
    --------
    >>> builder = OceanCarbonBuilder.from_parameters(
    ...     {
    ...         "gas_exchange_scale": 1.833492,
    ...         "enable_temp_feedback": True,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: OceanCarbonParams) -> OceanCarbonBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the ocean carbon cycle component."""

class CO2BudgetParams(TypedDict, total=False):
    """Parameters for CO2 budget component.

    All fields optional - defaults provided by Rust.
    """

    gtc_per_ppm: float
    permafrost: bool
    co2_pi: float
    concentration_switch_year: float | None
    ocean_end_of_year: bool

@final
class CO2BudgetBuilder(ComponentBuilder):
    """Builder for the CO2 budget component.

    Integrates emissions and uptakes to calculate atmospheric CO2 concentration change.

    # Parameters
    # ----------
    # gtc_per_ppm : float
    #     Conversion factor GtC per ppm CO2. Default: 2.123
    # co2_pi : float
    #     Pre-industrial CO2 concentration (ppm). Default: 278.0

    Inputs
    ------
    Emissions|CO2|Fossil : float
        Fossil fuel CO2 emissions (GtC/yr)
    Emissions|CO2|CH4 Oxidation : float
        CO2 from oxidised fossil CH4 (GtC/yr)
    Carbon Flux|Terrestrial : float
        Net atmosphere-to-land flux, including land-use emissions (GtC/yr)
    Carbon Flux|Ocean : float
        Ocean carbon uptake (GtC/yr)
    Atmospheric Concentration|CO2|Prescribed : float
        Concentration used up to ``concentration_switch_year`` (ppm)

    States
    ------
    Atmospheric Concentration|CO2 : float
        Atmospheric CO2 concentration (ppm)

    Outputs
    -------
    Emissions|CO2|Net : float
        Net CO2 emissions (GtC/yr)
    Airborne Fraction|CO2 : float
        Fraction of fossil sources remaining in the atmosphere (dimensionless)

    Examples
    --------
    >>> builder = CO2BudgetBuilder.from_parameters(
    ...     {
    ...         "gtc_per_ppm": 2.123,
    ...         "co2_pi": 278.0,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: CO2BudgetParams) -> CO2BudgetBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the CO2 budget component."""

class PermafrostSoilParams(TypedDict, total=False):
    """Parameters of one permafrost soil type (mineral soil or peat)."""

    thaw_rate: float
    thaw_exponent: float
    decomposition_ratio: float
    q10_aerobic_alpha: float
    q10_anaerobic_alpha: float
    anaerobic_initial_fraction: float
    anaerobic_max_fraction: float
    anaerobic_moisture_sensitivity: float
    ch4_oxidation_fraction: float

class PermafrostParams(TypedDict, total=False):
    """Parameters for the permafrost component.

    All fields optional - defaults provided by Rust.
    Fields missing from ``mineral_soil`` or ``peat`` keep that soil's defaults.
    """

    n_bands: int
    melting_temp_min: float
    melting_temp_max: float
    arctic_amplification: float
    soil_annual_cycle_amplitude: float
    aerobic_turnover_time: float
    anaerobic_decomposition_ratio: float
    q10_temp1: float
    q10_temp2: float
    soil_water_slope: float
    soil_water_offset: float
    soil_water_min: float
    total_pool: float
    mineral_soil_fraction_south: float
    mineral_soil_fraction_north: float
    zonal_pool_distribution: float
    mineral_soil: PermafrostSoilParams
    peat: PermafrostSoilParams

@final
class PermafrostBuilder(ComponentBuilder):
    """Builder for the permafrost carbon feedback component (MAGICC7 module 12).

    Thaws zonal bands of mineral soil and peat carbon as Arctic warming exceeds
    each band's melting threshold, then decomposes the thawed carbon to CO2 and CH4.
    Set ``permafrost`` on CH4 chemistry, CH4 oxidation, ocean carbon and the CO2
    budget to feed the emissions back.

    Inputs
    ------
    Surface Temperature : float
        Global mean surface temperature anomaly (K)

    Outputs
    -------
    Emissions|CO2|Permafrost : float
        CO2 released from permafrost (GtC/yr)
    Emissions|CH4|Permafrost : float
        CH4 released from permafrost (Mt CH4/yr)
    Permafrost Thawed Area : float
        Carbon weighted thawed fraction of the permafrost region

    Examples
    --------
    >>> builder = PermafrostBuilder.from_parameters({"total_pool": 800.0})
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: PermafrostParams) -> PermafrostBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the permafrost component."""

# Forcing Components

class OzoneForcingParams(TypedDict, total=False):
    """Parameters for the ozone forcing component.

    All fields optional - defaults provided by Rust.
    """

    strat_o3_scale: float
    strat_cl_exponent: float
    strat_threshold_year: float
    trop_radeff: float
    trop_oz_ch4: float
    trop_oz_nox: float
    trop_oz_co: float
    trop_oz_voc: float
    n2o_radeff: float
    temperature_scale: float
    trop_regions: TropOzoneRegions | None

class TropOzoneRegions(TypedDict, total=False):
    """Regional pattern of tropospheric ozone forcing, normalised per hemisphere."""

    rf_regions: list[float]
    nh_land_fraction: float
    sh_land_fraction: float

@final
class OzoneForcingBuilder(ComponentBuilder):
    """Builder for the ozone forcing component (MAGICC7 Module 04).

    Changes are relative to the first year of the run.

    Inputs
    ------
    EESC : float
        Equivalent effective stratospheric chlorine (ppt)
    Atmospheric Concentration|CH4 : float
        CH4 concentration (ppb)
    Atmospheric Concentration|N2O : float
        N2O concentration (ppb)
    Emissions|{NOx,CO,NMVOC}|{Fossil,AFOLU} : float
        Precursor emissions by sector (Mt N/yr, Mt CO/yr, Mt NMVOC/yr), lagged by a year
    Surface Temperature : float
        Global mean surface temperature anomaly (K), lagged by a year

    Outputs
    -------
    Effective Radiative Forcing|Stratospheric Ozone : float
        Stratospheric ozone forcing from EESC above its threshold-year value (W/m^2)
    Effective Radiative Forcing|Tropospheric Ozone : float or FourBox
        Tropospheric ozone forcing from CH4 and precursors (W/m^2),
        four-box with ``trop_regions``
    Effective Radiative Forcing|Ozone due to N2O : float
        Ozone forcing from N2O (W/m^2)
    Effective Radiative Forcing|Ozone due to Temperature : float
        Ozone response to warming (W/m^2)
    """

    @staticmethod
    def from_parameters(parameters: OzoneForcingParams) -> OzoneForcingBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the ozone forcing component."""

class AerosolDirectParams(TypedDict, total=False):
    """Parameters for direct aerosol forcing component.

    All fields optional - defaults provided by Rust.
    """

    sox_coefficient: float
    bc_coefficient: float
    oc_coefficient: float
    nitrate_coefficient: float
    sox_regional: list[float]
    bc_regional: list[float]
    oc_regional: list[float]
    nitrate_regional: list[float]
    sox_pi: float
    bc_pi: float
    oc_pi: float
    nox_pi: float
    harmonize: bool
    harmonize_year: float
    harmonize_target: float

@final
class AerosolDirectBuilder(ComponentBuilder):
    """Builder for the direct aerosol forcing component.

    Calculates direct aerosol radiative forcing as a linear combination of emissions
    from SOx, BC, OC, and nitrate, with optional regional distribution.

    # Parameters
    # ----------
    # sox_coefficient : float
    #     SOx forcing coefficient (W/m^2/(Tg S/yr)). Default: -0.0035
    # bc_coefficient : float
    #     BC forcing coefficient (W/m^2/(Tg BC/yr)). Default: 0.0077
    # oc_coefficient : float
    #     OC forcing coefficient (W/m^2/(Tg OC/yr)). Default: -0.002
    # nitrate_coefficient : float
    #     Nitrate forcing coefficient (W/m^2/(Tg N/yr)). Default: -0.001
    # sox_regional : list[float]
    #     SOx regional distribution (length 4). Default: [0.15, 0.55, 0.10, 0.20]
    # bc_regional : list[float]
    #     BC regional distribution weights (length 4). Default: [0.15, 0.50, 0.15, 0.20]
    # oc_regional : list[float]
    #     OC regional distribution weights (length 4). Default: [0.15, 0.45, 0.15, 0.25]
    # nitrate_regional : list[float]
    #     Nitrate regional distribution (length 4). Default: [0.15, 0.50, 0.15, 0.20]
    # sox_pi : float
    #     Pre-industrial SOx emissions (Tg S/yr). Default: 1.0
    # bc_pi : float
    #     Pre-industrial BC emissions (Tg BC/yr). Default: 2.5
    # oc_pi : float
    #     Pre-industrial OC emissions (Tg OC/yr). Default: 10.0
    # nox_pi : float
    #     Pre-industrial NOx emissions (Tg N/yr). Default: 10.0
    # harmonize : bool
    #     Enable harmonisation to reference year. Default: False
    # harmonize_year : float
    #     Reference year for harmonisation. Default: 2019.0
    # harmonize_target : float
    #     Target forcing at reference year (W/m^2). Default: -0.22

    Inputs
    ------
    Emissions|SOx : float
        SOx emissions (Tg S/yr)
    Emissions|BC : float
        BC emissions (Tg BC/yr)
    Emissions|OC : float
        OC emissions (Tg OC/yr)
    Emissions|NOx : float
        NOx emissions (Tg N/yr)

    Outputs
    -------
    Effective Radiative Forcing|Aerosol|Direct : float
        Direct aerosol forcing (W/m^2, FourBox grid)

    Examples
    --------
    >>> builder = AerosolDirectBuilder.from_parameters(
    ...     {
    ...         "sox_coefficient": -0.0035,
    ...         "bc_coefficient": 0.0077,
    ...         "harmonize": False,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: AerosolDirectParams) -> AerosolDirectBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the direct aerosol forcing component."""

class AerosolIndirectParams(TypedDict, total=False):
    """Parameters for indirect aerosol forcing component.

    All fields optional - defaults provided by Rust.
    """

    cloud_albedo_coefficient: float
    reference_burden: float
    sox_weight: float
    oc_weight: float
    sox_pi: float
    oc_pi: float
    harmonize: bool
    harmonize_year: float
    harmonize_target: float

@final
class AerosolIndirectBuilder(ComponentBuilder):
    """Builder for the indirect aerosol forcing component.

    Calculates indirect aerosol forcing (cloud albedo effect) using a logarithmic
    relationship with aerosol burden.

    # Parameters
    # ----------
    # cloud_albedo_coefficient : float
    #     Cloud albedo effect coefficient (W/m^2/ln-unit). Default: -1.0
    # reference_burden : float
    #     Reference aerosol burden (Tg/yr). Default: 50.0
    # sox_weight : float
    #     SOx weight in aerosol burden. Default: 1.0
    # oc_weight : float
    #     OC weight in aerosol burden. Default: 0.3
    # sox_pi : float
    #     Pre-industrial SOx emissions (Tg S/yr). Default: 1.0
    # oc_pi : float
    #     Pre-industrial OC emissions (Tg OC/yr). Default: 10.0
    # harmonize : bool
    #     Enable harmonisation to reference year. Default: False
    # harmonize_year : float
    #     Reference year for harmonisation. Default: 2019.0
    # harmonize_target : float
    #     Target forcing at reference year (W/m^2). Default: -0.89

    Inputs
    ------
    Emissions|SOx : float
        SOx emissions (Tg S/yr)
    Emissions|OC : float
        OC emissions (Tg OC/yr)

    Outputs
    -------
    Effective Radiative Forcing|Aerosol|Indirect : float
        Indirect aerosol forcing (W/m^2)

    Examples
    --------
    >>> builder = AerosolIndirectBuilder.from_parameters(
    ...     {
    ...         "cloud_albedo_coefficient": -1.0,
    ...         "reference_burden": 50.0,
    ...         "harmonize": False,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: AerosolIndirectParams) -> AerosolIndirectBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the indirect aerosol forcing component."""

# GHG Forcing

class GhgForcingParams(TypedDict, total=False):
    """Parameters for GhgForcing component.

    All fields optional - defaults provided by Rust.
    """

    method: Literal["Ipcctar", "Olbl"]
    co2_pi: float
    ch4_pi: float
    n2o_pi: float
    delq2xco2: float
    ch4_radeff: float
    n2o_radeff: float
    olbl_co2_a1: float
    olbl_co2_b1: float
    olbl_co2_c1: float
    olbl_co2_d1: float
    olbl_ch4_a3: float
    olbl_ch4_b3: float
    olbl_ch4_d3: float
    olbl_n2o_a2: float
    olbl_n2o_b2: float
    olbl_n2o_c2: float
    olbl_n2o_d2: float
    adjust_co2: float
    adjust_ch4: float
    adjust_n2o: float
    ch4_strat_h2o_fraction: float

@final
class GhgForcingBuilder(ComponentBuilder):
    """Builder for the GHG radiative forcing component.

    Calculates effective radiative forcing from CO2, CH4, and N2O
    concentrations using IPCCTAR or OLBL methods with optional
    rapid adjustment factors.

    # Parameters
    # ----------
    # method : str
    #     Forcing method: "Ipcctar" or "Olbl". Default: "Olbl"
    # co2_pi : float
    #     Pre-industrial CO2 (ppm). Default: 278.0
    # ch4_pi : float
    #     Pre-industrial CH4 (ppb). Default: 722.0
    # n2o_pi : float
    #     Pre-industrial N2O (ppb). Default: 270.0
    # delq2xco2 : float
    #     Forcing for CO2 doubling (W/m2). Default: 3.71
    # adjust_co2 : float
    #     Rapid adjustment factor for CO2. Default: 1.05
    # adjust_ch4 : float
    #     Rapid adjustment factor for CH4. Default: 0.86
    # adjust_n2o : float
    #     Rapid adjustment factor for N2O. Default: 1.0

    Inputs
    ------
    Atmospheric Concentration|CO2 : float
        CO2 concentration (ppm)
    Atmospheric Concentration|CH4 : float
        CH4 concentration (ppb)
    Atmospheric Concentration|N2O : float
        N2O concentration (ppb)

    Outputs
    -------
    Effective Radiative Forcing|CO2 : float
        CO2 effective radiative forcing (W/m^2)
    Effective Radiative Forcing|CH4 : float
        CH4 effective radiative forcing (W/m^2)
    Effective Radiative Forcing|N2O : float
        N2O effective radiative forcing (W/m^2)
    Effective Radiative Forcing|2xCO2 : float
        ERF of doubling CO2 at the current concentrations (W/m^2)
    Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O : float
        Stratospheric H2O forcing from CH4 oxidation (W/m^2)

    Examples
    --------
    >>> builder = GhgForcingBuilder.from_parameters(
    ...     {
    ...         "method": "Ipcctar",
    ...         "delq2xco2": 3.71,
    ...         "adjust_co2": 1.05,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: GhgForcingParams) -> GhgForcingBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the GHG radiative forcing component."""
    def calculate_forcings(
        self, co2: float, ch4: float, n2o: float
    ) -> dict[str, float]:
        """Evaluate boundary ERF in W/m², keyed by CO2, CH4 and N2O.

        Concentrations must be finite and positive, in ppm, ppb and ppb.
        """
    def co2_doubling_forcing(self) -> float:
        """ERF in W/m² from doubling CO2 above ``co2_pi``.

        CH4 and N2O are held at their preindustrial references. MAGICC7 derives
        its climate feedback from this value.
        """

# Forcing aggregation

class ForcingAgent(TypedDict, total=False):
    """A forcing agent read by ForcingAggregation. Only ``variable`` is required."""

    variable: str
    regional: bool
    rf_regions: tuple[float, float, float, float]
    hemispheric: bool
    efficacy: float
    internal_efficacy: float
    sum_of: list[str]
    """Variables added in order into this agent before its efficacy applies.

    Empty reads ``variable`` itself.
    """

ForcingTerm: TypeAlias = str | list["ForcingTerm"]
"""An agent ``variable``, or a list of terms summed before the enclosing sum."""

class EfficacyClimate(TypedDict, total=False):
    """Climate settings for recalculating internal efficacies each year.

    The efficacies come from LAMCALC at the previous year's
    ``Equilibrium Climate Sensitivity`` and ``Effective Radiative Forcing|2xCO2``.
    """

    ecs: float
    rf_2xco2: float
    k_lo: float
    k_ns: float
    rlo: float
    amplify_ocean_to_land: float
    rf_regions_co2: tuple[float, float, float, float]

class ForcingAggregationParams(TypedDict, total=False):
    """Parameters for ForcingAggregation."""

    agents: list[ForcingAgent]
    summation: list[ForcingTerm]
    """Order in which agents are added, by ``variable``.

    Nested lists are added first. Empty adds the agents in order.
    """
    efficacy_apply: int
    nh_land_fraction: float
    sh_land_fraction: float
    climate: EfficacyClimate

@final
class ForcingAggregationBuilder(ComponentBuilder):
    """Builder for the component combining agent ERF into four-box climate forcing.

    Reads each agent's ERF and writes the total ERF and the four-box
    ``Effective Radiative Forcing|Efficacy Adjusted`` read by ClimateUDEB
    when ``regional_forcing`` is set.
    """

    @staticmethod
    def from_parameters(  # type: ignore[override]
        parameters: ForcingAggregationParams,
    ) -> ForcingAggregationBuilder:
        """Create a builder from a parameter dictionary."""
    def aggregate(
        self, values: list[list[float]]
    ) -> tuple[float, tuple[float, float, float, float]]:
        """Return total ERF and box forcing from each agent's value or box values."""
    def internal_efficacies(self, ecs: float, q_2xco2: float) -> list[float]:
        """Return the internal efficacy of each agent at climate sensitivity ``ecs``.

        ``q_2xco2`` is the CO2 doubling forcing in W/m².
        Without ``climate`` these are the agents' own ``internal_efficacy``.
        """
    def build(self) -> Component:
        """Build the forcing aggregation component."""

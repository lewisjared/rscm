"""MAGICC climate model components.

This module provides Rust implementations of MAGICC (Model for the Assessment of
Greenhouse gas Induced Climate Change) component builders.

MAGICC is a reduced-complexity climate model widely used for climate policy
analysis. The components in this module implement MAGICC's key physical processes.

Available Components
--------------------
Climate:

- ClimateUDEBBuilder: four-box UDEB climate model with ocean diffusion
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

from typing import Any, Literal, NotRequired, TypedDict, final

from rscm._lib.core import ComponentBuilder, RustComponent

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
    """Builder for the four-box UDEB climate model component.

    The UDEB (Upwelling-Diffusion Energy Balance) model couples a four-box atmosphere
    (Northern Ocean, Northern Land, Southern Ocean, Southern Land) to a multi-layer
    ocean with vertical diffusion and upwelling.

    **Parameter keys**

    - `n_layers` (`int`, default `50`): Number of ocean layers (including mixed layer).
    - `mixed_layer_depth` (`float`, default `60.0`): Mixed layer depth (m).
    - `layer_thickness` (`float`, default `100.0`): Layer thickness for deeper ocean
      layers (m).
    - `kappa` (`float`, default `0.75`): Base vertical diffusivity (cm^2/s).
    - `kappa_min` (`float`, default `0.1`): Minimum vertical diffusivity (cm^2/s).
    - `kappa_dkdt` (`float`, default `-0.191`): Temperature gradient coefficient for
      diffusivity (cm^2/s/K).
    - `w_initial` (`float`, default `3.5`): Initial upwelling rate (m/yr).
    - `w_variable_fraction` (`float`, default `0.7`): Variable fraction of upwelling.
    - `w_threshold_temp_nh` (`float`, default `8.0`): Temperature threshold for NH
      upwelling shutdown (K).
    - `w_threshold_temp_sh` (`float`, default `8.0`): Temperature threshold for SH
      upwelling shutdown (K).
    - `ecs` (`float`, default `3.0`): Equilibrium climate sensitivity (K).
    - `rf_2xco2` (`float`, default `3.71`): Radiative forcing for 2xCO2 (W/m^2).
    - `rlo` (`float`, default `1.317`): Land-ocean warming ratio.
    - `k_lo` (`float`, default `1.44`): Land-ocean heat exchange coefficient (W/m^2/K).
    - `k_ns` (`float`, default `0.31`): Inter-hemispheric heat exchange coefficient
      (W/m^2/K).
    - `amplify_ocean_to_land` (`float`, default `1.02`): Ocean-to-land heat exchange
      amplification factor.
    - `nh_land_fraction` (`float`, default `0.42`): Northern Hemisphere land fraction.
    - `sh_land_fraction` (`float`, default `0.21`): Southern Hemisphere land fraction.
    - `temp_adjust_alpha` (`float`, default `1.04`): Ocean-to-atmosphere temperature
      adjustment alpha.
    - `temp_adjust_gamma` (`float`, default `-0.002`): Ocean-to-atmosphere temperature
      adjustment gamma (1/K).
    - `polar_sinking_ratio` (`float`, default `0.2`): Polar sinking water temperature
      ratio.
    - `steps_per_year` (`int`, default `12`): Steps per year for sub-annual integration.
    - `max_temperature` (`float`, default `25.0`): Maximum temperature anomaly cap (K).

    Inputs, outputs and states are listed on the
    [ClimateUDEB component page][component-climateudeb].

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
    def build(self) -> RustComponent:
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
    """Builder for the sea level rise component.

    Glaciers, Greenland and Antarctic surface mass balance and solid ice discharge
    respond to the annual mean warming. Thermal expansion comes from ClimateUDEB
    with ``thermal_expansion`` set. Steps must be one year long.

    Inputs, outputs and states are listed on the
    [SeaLevelRise component page][component-sealevelrise].

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
    def build(self) -> RustComponent:
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

    **Parameter keys**

    - `feedback_start_year` (`float`, default `1927.0`): Year from which feedbacks are
      measured.
    - `natural_emissions` (`float`, default `182.1908`): Natural CH4 emissions (Tg
      CH4/yr).
    - `tau_total_init` (`float`, default `9.9474`): Total lifetime at the reference
      state (years).
    - `tau_soil` (`float`, default `150.0`): Soil uptake sink lifetime (years).
    - `tau_strat` (`float`, default `120.0`): Stratospheric sink lifetime (years).
    - `tau_trop_cl` (`float`, default `200.0`): Tropospheric Cl sink lifetime (years).
    - `ch4_self_feedback` (`float`, default `-0.53775`): CH4 self-feedback coefficient.
    - `oh_sensitivity_scale` (`float`, default `0.72448`): OH sensitivity scaling
      factor.
    - `oh_nox_sensitivity` (`float`, default `0.0093376`): NOx emissions effect on OH
      ((Tg N/yr)^-1).
    - `oh_co_sensitivity` (`float`, default `-0.000113`): CO emissions effect on OH ((Tg
      CO/yr)^-1).
    - `oh_nmvoc_sensitivity` (`float`, default `-0.0003142`): NMVOC emissions effect on
      OH ((Tg NMVOC/yr)^-1).
    - `temp_sensitivity` (`float`, default `0.07`): Temperature sensitivity of lifetime
      (K^-1).
    - `include_temp_feedback` (`bool`, default `True`): Enable temperature feedback.
    - `include_emissions_feedback` (`bool`, default `True`): Enable NOx/CO/NMVOC
      emissions feedback.
    - `ppb_to_tg` (`float`, default `2.824`): Conversion factor ppb to Tg CH4.
    - `mixbox_size` (`float`, default `0.973`): Fraction of the atmosphere mixed with
      surface CH4.

    Inputs, outputs and states are listed on the
    [CH4Chemistry component page][component-ch4chemistry].

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
    def build(self) -> RustComponent:
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

    **Parameter keys**

    - `carbon_per_ch4` (`float`, default `0.75`): Mass of carbon per mass of CH4
      oxidised.

    Inputs, outputs and states are listed on the
    [CH4Oxidation component page][component-ch4oxidation].
    """

    @staticmethod
    def from_parameters(parameters: CH4OxidationParams) -> CH4OxidationBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
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

    **Parameter keys**

    - `feedback_start_year` (`float`, default `1925.0`): Year from which the
      concentration feedback is measured.
    - `natural_emissions` (`float`, default `8.0345017`): Natural N2O emissions (Tg
      N/yr).
    - `tau_n2o` (`float`, default `139.275`): Base atmospheric lifetime (years).
    - `lifetime_feedback` (`float`, default `-0.04`): Lifetime feedback exponent.
    - `strat_delay` (`int`, default `1`): Stratospheric mixing delay (years).
    - `ppb_to_tg` (`float`, default `4.79`): Conversion factor ppb to Tg N.
    - `include_meridional_flux_feedback` (`bool`, default `True`): Shorten the lifetime
      with warming.
    - `meridional_flux_change_per_degree` (`float`, default `0.15`): Meridional flux
      change per degree of warming (K^-1).
    - `tau_sensitivity_to_meridional_flux` (`float`, default `0.04`): Lifetime
      sensitivity to meridional flux changes.
    - `meridional_flux_reference_year` (`float`, default `1980.0`): Year from which
      warming changes the meridional flux.

    Inputs, outputs and states are listed on the
    [N2OChemistry component page][component-n2ochemistry].

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
    def build(self) -> RustComponent:
        """Build the N2O chemistry component."""

class HalocarbonSpecies(TypedDict):
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
    mix_box_size: NotRequired[float]

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
    following MAGICC7.

    **Inputs**

    - `Emissions|{Species}`: Emissions for each halocarbon species (kt/yr).
    - `Lifetime|CH4|OH`: CH4 OH lifetime (yr), which scales OH lifetimes.
    - `Surface Temperature`: Warming (K), which shortens stratospheric
      lifetimes.
    - `Emissions|CO2|Fossil`: Fossil CO2 emissions (GtC/yr), read with
      ``fgas_regions``.

    **States**

    - `Atmospheric Concentration|{Species}`: Concentration for each halocarbon
      species (ppt).

    **Outputs**

    - `Effective Radiative Forcing|F-Gases` (float or FourBox): F-gas ERF (W/m²),
      four-box with ``fgas_regions``.
    - `Effective Radiative Forcing|Montreal Protocol Halogen Gases`: Montreal
      Protocol gas ERF (W/m²).
    - `EESC`: Equivalent Effective Stratospheric Chlorine (ppt).

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
    def build(self) -> RustComponent:
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
    """Parameters for the nitrogen limitation of NPP.

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

    Inputs, outputs and states are listed on the
    [TerrestrialCarbon component page][component-terrestrialcarbon].

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
    def build(self) -> RustComponent:
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

    **Parameter keys**

    - `pco2_pi` (`float | None`, default `the initial pCO2`): Unperturbed ocean surface
      pCO2 (ppm).
    - `gas_exchange_scale` (`float`, default `1.833492`): Gas exchange rate scaling
      factor.
    - `gas_exchange_tau` (`float`, default `7.66`): Gas exchange timescale (years).
    - `temp_sensitivity` (`float`, default `0.03717879`): Temperature sensitivity of
      pCO2 (K^-1).
    - `irf_scale` (`float`, default `0.9492864`): IRF scaling factor.
    - `mixed_layer_depth` (`float`, default `50.9`): Mixed layer depth (m).
    - `ocean_surface_area` (`float`, default `3.55e14`): Ocean surface area (m^2).
    - `sst_pi` (`float`, default `17.7`): Pre-industrial sea surface temperature (C).
    - `steps_per_year` (`int`, default `12`): Substeps per year.
    - `max_history_months` (`int`, default `6000`): Maximum flux history length
      (months).
    - `enable_temp_feedback` (`bool`, default `True`): Enable temperature feedback on
      pCO2.
    - `tempfeedback_yrstart` (`float`, default `1900.0`): Year from which warming drives
      the pCO2 feedback.
    - `flux_change_limit` (`float`, default `0.04`): Largest flux change between
      substeps (ppm/yr), disabled if non-positive.
    - `gtc_per_ppm` (`float`, default `2.123`): Conversion factor GtC per ppm CO2.

    Inputs, outputs and states are listed on the
    [OceanCarbon component page][component-oceancarbon].

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
    def build(self) -> RustComponent:
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

    **Parameter keys**

    - `gtc_per_ppm` (`float`, default `2.123`): Conversion factor GtC per ppm CO2.
    - `co2_pi` (`float`, default `278.0`): Pre-industrial CO2 concentration (ppm).

    Inputs, outputs and states are listed on the
    [CO2Budget component page][component-co2budget].

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
    def build(self) -> RustComponent:
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
    """Builder for the permafrost carbon feedback component.

    Thaws zonal bands of mineral soil and peat carbon as Arctic warming exceeds
    each band's melting threshold, then decomposes the thawed carbon to CO2 and CH4.
    Set ``permafrost`` on CH4 chemistry, CH4 oxidation, ocean carbon and the CO2
    budget to feed the emissions back.

    Inputs, outputs and states are listed on the
    [Permafrost component page][component-permafrost].

    Examples
    --------
    >>> builder = PermafrostBuilder.from_parameters({"total_pool": 800.0})
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: PermafrostParams) -> PermafrostBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
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
    """Builder for the ozone forcing component.

    Changes are relative to the first year of the run.

    **Inputs**

    - `EESC`: Equivalent effective stratospheric chlorine (ppt).
    - `Atmospheric Concentration|CH4`: CH4 concentration (ppb).
    - `Atmospheric Concentration|N2O`: N2O concentration (ppb).
    - `Emissions|{NOx,CO,NMVOC}|{Fossil,AFOLU}`: Precursor emissions by sector
      (Mt N/yr, Mt CO/yr, Mt NMVOC/yr), lagged by a year.
    - `Surface Temperature`: Global mean surface temperature anomaly (K), lagged
      by a year.

    **Outputs**

    - `Effective Radiative Forcing|Stratospheric Ozone`: Stratospheric ozone
      forcing from EESC above its threshold-year value (W/m^2).
    - `Effective Radiative Forcing|Tropospheric Ozone` (float or FourBox): Tropospheric
      ozone forcing from CH4 and precursors (W/m^2), four-box with ``trop_regions``.
    - `Effective Radiative Forcing|Ozone due to N2O`: Ozone forcing from N2O
      (W/m^2).
    - `Effective Radiative Forcing|Ozone due to Temperature`: Ozone response to
      warming (W/m^2).
    """

    @staticmethod
    def from_parameters(parameters: OzoneForcingParams) -> OzoneForcingBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
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

    **Parameter keys**

    - `sox_coefficient` (`float`, default `-0.0035`): SOx forcing coefficient (W/m^2/(Tg
      S/yr)).
    - `bc_coefficient` (`float`, default `0.0077`): BC forcing coefficient (W/m^2/(Tg
      BC/yr)).
    - `oc_coefficient` (`float`, default `-0.002`): OC forcing coefficient (W/m^2/(Tg
      OC/yr)).
    - `nitrate_coefficient` (`float`, default `-0.001`): Nitrate forcing coefficient
      (W/m^2/(Tg N/yr)).
    - `sox_regional` (`list[float]`, default `[0.15, 0.55, 0.10, 0.20]`): SOx regional
      distribution (length 4).
    - `bc_regional` (`list[float]`, default `[0.15, 0.50, 0.15, 0.20]`): BC regional
      distribution weights (length 4).
    - `oc_regional` (`list[float]`, default `[0.15, 0.45, 0.15, 0.25]`): OC regional
      distribution weights (length 4).
    - `nitrate_regional` (`list[float]`, default `[0.15, 0.50, 0.15, 0.20]`): Nitrate
      regional distribution (length 4).
    - `sox_pi` (`float`, default `1.0`): Pre-industrial SOx emissions (Tg S/yr).
    - `bc_pi` (`float`, default `2.5`): Pre-industrial BC emissions (Tg BC/yr).
    - `oc_pi` (`float`, default `10.0`): Pre-industrial OC emissions (Tg OC/yr).
    - `nox_pi` (`float`, default `10.0`): Pre-industrial NOx emissions (Tg N/yr).
    - `harmonize` (`bool`, default `False`): Enable harmonisation to reference year.
    - `harmonize_year` (`float`, default `2019.0`): Reference year for harmonisation.
    - `harmonize_target` (`float`, default `-0.22`): Target forcing at reference year
      (W/m^2).

    Inputs, outputs and states are listed on the
    [AerosolDirect component page][component-aerosoldirect].

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
    def build(self) -> RustComponent:
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

    **Parameter keys**

    - `cloud_albedo_coefficient` (`float`, default `-1.0`): Cloud albedo effect
      coefficient (W/m^2/ln-unit).
    - `reference_burden` (`float`, default `50.0`): Reference aerosol burden (Tg/yr).
    - `sox_weight` (`float`, default `1.0`): SOx weight in aerosol burden.
    - `oc_weight` (`float`, default `0.3`): OC weight in aerosol burden.
    - `sox_pi` (`float`, default `1.0`): Pre-industrial SOx emissions (Tg S/yr).
    - `oc_pi` (`float`, default `10.0`): Pre-industrial OC emissions (Tg OC/yr).
    - `harmonize` (`bool`, default `False`): Enable harmonisation to reference year.
    - `harmonize_year` (`float`, default `2019.0`): Reference year for harmonisation.
    - `harmonize_target` (`float`, default `-0.89`): Target forcing at reference year
      (W/m^2).

    Inputs, outputs and states are listed on the
    [AerosolIndirect component page][component-aerosolindirect].

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
    def build(self) -> RustComponent:
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

    **Parameter keys**

    - `method` (`str`, default `"Olbl"`): Forcing method: "Ipcctar" or "Olbl".
    - `co2_pi` (`float`, default `278.0`): Pre-industrial CO2 (ppm).
    - `ch4_pi` (`float`, default `722.0`): Pre-industrial CH4 (ppb).
    - `n2o_pi` (`float`, default `270.0`): Pre-industrial N2O (ppb).
    - `delq2xco2` (`float`, default `3.71`): Forcing for CO2 doubling (W/m2).
    - `adjust_co2` (`float`, default `1.05`): Rapid adjustment factor for CO2.
    - `adjust_ch4` (`float`, default `0.86`): Rapid adjustment factor for CH4.
    - `adjust_n2o` (`float`, default `1.0`): Rapid adjustment factor for N2O.

    Inputs, outputs and states are listed on the
    [GhgForcing component page][component-ghgforcing].

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
    def build(self) -> RustComponent:
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

type ForcingTerm = str | list[ForcingTerm]
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
    def build(self) -> RustComponent:
        """Build the forcing aggregation component."""

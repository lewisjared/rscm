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

from typing import Any, Literal, NotRequired, Required, TypedDict, final

from rscm._lib.core import ComponentBuilder, RustComponent

# Climate Components

class ClimateUDEBParameters(TypedDict, total=False):
    """Parameters for ClimateUDEB component.

    All fields optional - defaults provided by Rust.
    """

    n_layers: int
    mixed_layer_depth: float
    layer_thickness: float
    vertical_diffusivity: float
    minimum_vertical_diffusivity: float
    vertical_diffusivity_gradient_sensitivity: float
    initial_upwelling_rate: float
    upwelling_variable_fraction: float
    upwelling_threshold_temperature_nh: float
    upwelling_threshold_temperature_sh: float
    ecs: float
    erf_2xco2: float
    land_ocean_warming_ratio: float
    land_ocean_heat_exchange: float
    north_south_heat_exchange: float
    amplify_ocean_to_land: float
    nh_land_fraction: float
    sh_land_fraction: float
    efficacy_method: Literal["none", "unit", "ar6"]
    co2_efficacy: float
    include_regional_forcing: bool
    time_varying_erf_2xco2: bool
    temperature_adjustment_alpha: float
    temperature_adjustment_gamma: float
    polar_sinking_ratio: float
    steps_per_year: int
    maximum_temperature: float
    thermal_expansion: ThermalExpansion | None

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
    - `vertical_diffusivity` (`float`, default `0.75`): Base vertical diffusivity
      (cm^2/s).
    - `minimum_vertical_diffusivity` (`float`, default `0.1`): Minimum vertical
      diffusivity (cm^2/s).
    - `vertical_diffusivity_gradient_sensitivity` (`float`, default `-0.191`):
      Temperature gradient coefficient for diffusivity (cm^2/s/K).
    - `initial_upwelling_rate` (`float`, default `3.5`): Initial upwelling rate (m/yr).
    - `upwelling_variable_fraction` (`float`, default `0.7`): Variable fraction of
      upwelling.
    - `upwelling_threshold_temperature_nh` (`float`, default `8.0`): Temperature
      threshold for NH upwelling shutdown (K).
    - `upwelling_threshold_temperature_sh` (`float`, default `8.0`): Temperature
      threshold for SH upwelling shutdown (K).
    - `ecs` (`float`, default `3.0`): Equilibrium climate sensitivity (K).
    - `erf_2xco2` (`float`, default `3.71`): 2xCO2 effective radiative forcing (W/m^2).
    - `land_ocean_warming_ratio` (`float`, default `1.317`): Land-ocean warming ratio.
    - `land_ocean_heat_exchange` (`float`, default `1.44`): Land-ocean heat exchange
      coefficient (W/m^2/K).
    - `north_south_heat_exchange` (`float`, default `0.31`): Inter-hemispheric heat
      exchange coefficient (W/m^2/K).
    - `amplify_ocean_to_land` (`float`, default `1.02`): Ocean-to-land heat exchange
      amplification factor.
    - `nh_land_fraction` (`float`, default `0.42`): Northern Hemisphere land fraction.
    - `sh_land_fraction` (`float`, default `0.21`): Southern Hemisphere land fraction.
    - `temperature_adjustment_alpha` (`float`, default `1.04`): Ocean-to-atmosphere
      temperature adjustment alpha.
    - `temperature_adjustment_gamma` (`float`, default `-0.002`): Ocean-to-atmosphere
      temperature adjustment gamma (1/K).
    - `polar_sinking_ratio` (`float`, default `0.2`): Polar sinking water temperature
      ratio.
    - `steps_per_year` (`int`, default `12`): Substeps per year.
    - `maximum_temperature` (`float`, default `25.0`): Maximum temperature anomaly cap
      (K).

    Inputs, outputs and states are listed on the
    [ClimateUDEB component page][component-climateudeb].

    Examples
    --------
    >>> builder = ClimateUDEBBuilder.from_parameters(
    ...     {
    ...         "ecs": 3.0,
    ...         "erf_2xco2": 3.71,
    ...         "mixed_layer_depth": 60.0,
    ...     }
    ... )
    >>> component = builder.build()
    """

    def initial_values(self, erf: float) -> dict[str, float]:
        """
        Return initial values and diagnostics for an unperturbed climate.

        The supplied ERF sets the initial heat uptake.
        """
    def internal_efficacy(
        self, regional_pattern: tuple[float, float, float, float]
    ) -> float:
        """Return the internal efficacy of an agent with this regional pattern."""
    @staticmethod
    def from_parameters(parameters: ClimateUDEBParameters) -> ClimateUDEBBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the UDEB climate model component."""

class ThermalExpansion(TypedDict, total=False):
    """Thermal expansion of the ocean, calculated by ClimateUDEB.

    Setting it makes ClimateUDEB write
    ``Annual Mean Surface Air Temperature Change`` (K)
    and ``Sea Level Rise|Thermal Expansion`` (mm).
    """

    coefficients: tuple[float, float, float, float, float, float]
    scaling: float
    start_year: float

class Glacier(TypedDict, total=False):
    """Mountain glaciers and small ice caps."""

    sensitivity: float
    exponent: float
    start_year: float
    equilibrium_temperature: list[float]
    equilibrium_sea_level: list[float]

class SurfaceMassBalance(TypedDict, total=False):
    """Surface mass balance of an ice sheet."""

    sensitivity: float
    linear_fraction: float
    exponent: float
    start_year: float
    volume: float | None
    volume_exponent: float

class DischargeCase(TypedDict, total=False):
    """One discharge case of the Greenland ice sheet."""

    volume: float
    sensitivity: float
    exponent: float

class GreenlandDischarge(TypedDict, total=False):
    """Greenland solid ice discharge, between a low and a high case."""

    low: DischargeCase
    high: DischargeCase
    case: float
    scaling: float
    start_year: float

class AntarcticRegion(TypedDict):
    """One Antarctic drainage region."""

    name: str
    delay: int
    temperature_scaling: float
    response: tuple[float, float, float, float, float]

class AntarcticDischarge(TypedDict, total=False):
    """Antarctic solid ice discharge (Levermann et al. 2014)."""

    basal_melt_sensitivity: float
    response_span: int
    scaling: float
    start_year: float
    regions: list[AntarcticRegion]

class SeaLevelRiseParameters(TypedDict, total=False):
    """Parameters for the sea level rise component.

    All fields optional - defaults are MAGICC7's effective defaults.
    Fields missing from a nested group keep that group's defaults.
    """

    glaciers: Glacier
    greenland_surface_mass_balance: SurfaceMassBalance
    greenland_discharge: GreenlandDischarge
    antarctic_surface_mass_balance: SurfaceMassBalance
    antarctic_discharge: AntarcticDischarge

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
    def from_parameters(parameters: SeaLevelRiseParameters) -> SeaLevelRiseBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the sea level rise component."""

# Chemistry Components

class CH4ChemistryParameters(TypedDict, total=False):
    """Parameters for CH4 chemistry component.

    All fields optional - defaults provided by Rust.
    """

    feedback_start_year: float
    natural_emissions: float
    include_permafrost: bool
    initial_ch4_lifetime: float
    soil_lifetime: float
    stratospheric_lifetime: float
    tropospheric_cl_lifetime: float
    ch4_self_feedback: float
    oh_sensitivity_scale: float
    oh_nox_sensitivity: float
    oh_co_sensitivity: float
    oh_nmvoc_sensitivity: float
    temperature_sensitivity: float
    include_temperature_feedback: bool
    include_emissions_feedback: bool
    concentration_switch_year: float | None
    tg_per_ppb: float
    mixing_box_size: float

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
    - `initial_ch4_lifetime` (`float`, default `9.9474`): Total lifetime at the
      reference state (years).
    - `soil_lifetime` (`float`, default `150.0`): Soil uptake sink lifetime (years).
    - `stratospheric_lifetime` (`float`, default `120.0`): Stratospheric sink lifetime
      (years).
    - `tropospheric_cl_lifetime` (`float`, default `200.0`): Tropospheric Cl sink
      lifetime (years).
    - `ch4_self_feedback` (`float`, default `-0.53775`): CH4 self-feedback coefficient.
    - `oh_sensitivity_scale` (`float`, default `0.72448`): OH sensitivity scaling
      factor.
    - `oh_nox_sensitivity` (`float`, default `0.0093376`): NOx emissions effect on OH
      ((Tg N/yr)^-1).
    - `oh_co_sensitivity` (`float`, default `-0.000113`): CO emissions effect on OH ((Tg
      CO/yr)^-1).
    - `oh_nmvoc_sensitivity` (`float`, default `-0.0003142`): NMVOC emissions effect on
      OH ((Tg NMVOC/yr)^-1).
    - `temperature_sensitivity` (`float`, default `0.07`): Temperature sensitivity of
      lifetime (K^-1).
    - `include_temperature_feedback` (`bool`, default `True`): Enable temperature
      feedback.
    - `include_emissions_feedback` (`bool`, default `True`): Enable NOx/CO/NMVOC
      emissions feedback.
    - `tg_per_ppb` (`float`, default `2.824`): Conversion factor ppb to Tg CH4.
    - `mixing_box_size` (`float`, default `0.973`): Fraction of the atmosphere mixed
      with surface CH4.

    Inputs, outputs and states are listed on the
    [CH4Chemistry component page][component-ch4chemistry].

    Examples
    --------
    >>> builder = CH4ChemistryBuilder.from_parameters(
    ...     {
    ...         "initial_ch4_lifetime": 9.9474,
    ...         "ch4_self_feedback": -0.53775,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: CH4ChemistryParameters) -> CH4ChemistryBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the CH4 chemistry component."""

class CH4OxidationParameters(TypedDict, total=False):
    """Parameters for the fossil CH4 oxidation component.

    All fields optional - defaults provided by Rust.
    """

    carbon_per_ch4: float
    include_permafrost: bool
    permafrost_fraction: float
    tg_per_ppb: float
    mixing_box_size: float
    non_oh_lifetime: float
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
    def from_parameters(parameters: CH4OxidationParameters) -> CH4OxidationBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the CH4 oxidation component."""

class N2OChemistryParameters(TypedDict, total=False):
    """Parameters for N2O chemistry component.

    All fields optional - defaults provided by Rust.
    """

    feedback_start_year: float
    natural_emissions: float
    initial_n2o_lifetime: float
    lifetime_feedback: float
    stratospheric_delay: int
    tg_per_ppb: float
    include_meridional_flux_feedback: bool
    meridional_flux_change_per_degree: float
    lifetime_sensitivity_to_meridional_flux: float
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
    - `initial_n2o_lifetime` (`float`, default `139.275`): Base atmospheric lifetime
      (years).
    - `lifetime_feedback` (`float`, default `-0.04`): Lifetime feedback exponent.
    - `stratospheric_delay` (`int`, default `1`): Stratospheric mixing delay (years).
    - `tg_per_ppb` (`float`, default `4.79`): Conversion factor ppb to Tg N.
    - `include_meridional_flux_feedback` (`bool`, default `True`): Shorten the lifetime
      with warming.
    - `meridional_flux_change_per_degree` (`float`, default `0.15`): Meridional flux
      change per degree of warming (K^-1).
    - `lifetime_sensitivity_to_meridional_flux` (`float`, default `0.04`): Lifetime
      sensitivity to meridional flux changes.
    - `meridional_flux_reference_year` (`float`, default `1980.0`): Year from which
      warming changes the meridional flux.

    Inputs, outputs and states are listed on the
    [N2OChemistry component page][component-n2ochemistry].

    Examples
    --------
    >>> builder = N2OChemistryBuilder.from_parameters(
    ...     {
    ...         "initial_n2o_lifetime": 139.275,
    ...         "lifetime_feedback": -0.04,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: N2OChemistryParameters) -> N2OChemistryBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the N2O chemistry component."""

class HalocarbonSpecies(TypedDict):
    """A halocarbon species. Lifetimes in yr, radiative efficiency in W/m²/ppb."""

    name: str
    total_lifetime: float
    oh_lifetime: float
    stratospheric_lifetime: float
    radiative_efficiency: float
    concentration_pi: float
    molecular_mass: float
    cl_atoms: int
    br_atoms: int
    fractional_release: float
    mixing_box_size: NotRequired[float]

class FGasRegions(TypedDict, total=False):
    """Regional pattern of F-gas forcing.

    Species with lifetimes of at least ``upper_lifetime`` follow ``regional_pattern``
    normalised globally. Those below ``lower_lifetime`` follow it normalised within
    each hemisphere, scaled by each hemisphere's share of fossil CO2 emissions.
    Lifetimes in between mix the two linearly.
    """

    regional_pattern: tuple[float, float, float, float]
    lower_lifetime: float
    upper_lifetime: float
    nh_land_fraction: float
    sh_land_fraction: float

class HalocarbonChemistryParameters(TypedDict, total=False):
    """Parameters for halocarbon chemistry component.

    All fields optional, defaults provided by Rust.
    """

    fgases: list[HalocarbonSpecies]
    montreal_gases: list[HalocarbonSpecies]
    br_cl_ratio: float
    cfc11_release_factor: float
    stratospheric_delay: int
    air_molar_mass: float
    atmosphere_mass: float
    include_variable_oh_lifetime: bool
    initial_ch4_oh_lifetime: float
    include_variable_stratospheric_lifetime: bool
    meridional_flux_reference_year: float
    meridional_flux_change_per_degree: float
    stratospheric_lifetime_sensitivity: float
    fgas_switch_year: float | None
    montreal_gases_switch_year: float | None
    fgas_regions: FGasRegions | None
    """Makes ``Effective Radiative Forcing|F-Gases`` a four-box output."""

@final
class HalocarbonChemistryBuilder(ComponentBuilder):
    """Builder for the halocarbon chemistry component.

    Concentrations, forcing and EESC of F-gases and Montreal Protocol gases,
    following MAGICC7.

    **Inputs**

    - `Emissions|{Species}`: Emissions for each halocarbon species (`kt <species>/yr`).
    - `Lifetime|CH4|OH`: CH4 OH lifetime (yr), which scales OH lifetimes.
    - `Surface Air Temperature Change`: Warming (K), which shortens stratospheric
      lifetimes.
    - `Emissions|CO2|Fossil`: Fossil CO2 emissions (GtC/yr), read with
      ``fgas_regions``.

    **States**

    - `Atmospheric Concentrations|{Species}`: Concentration for each halocarbon
      species (ppt).

    **Outputs**

    - `Effective Radiative Forcing|F-Gases` (float or FourBox): F-gas ERF (W/m²),
      four-box with ``fgas_regions``.
    - `Effective Radiative Forcing|Montreal Protocol Halogen Gases`: Montreal
      Protocol gas ERF (W/m²).
    - `Equivalent Effective Stratospheric Chlorine` (ppt).

    Examples
    --------
    >>> builder = HalocarbonChemistryBuilder.from_parameters({"stratospheric_delay": 3})
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(  # type: ignore[override]
        parameters: HalocarbonChemistryParameters,
    ) -> HalocarbonChemistryBuilder:
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the halocarbon chemistry component."""

# Carbon Cycle Components

class TerrestrialCarbonParameters(TypedDict, total=False):
    """Parameters for terrestrial carbon cycle component.

    All fields optional - defaults provided by Rust.
    """

    npp_pi: float
    co2_pi: float
    fertilisation_factor: float
    fertilisation_method: float
    fertilisation_factor2: float
    gifford_concentration_for_zero_npp: float
    fertilisation_start_year: float
    npp_temperature_sensitivity: float
    respiration_temperature_sensitivity: float
    detritus_temperature_sensitivity: float
    soil_temperature_sensitivity: float
    temperature_feedback_start_year: float
    plant_pool_pi: float
    detritus_pool_pi: float
    soil_pool_pi: float
    respiration_pi: float
    plant_respiration_fertilisation_scale: float
    npp_to_plant_fraction: float
    npp_to_detritus_fraction: float
    plant_to_detritus_fraction: float
    detritus_to_soil_fraction: float
    deforestation_plant_fraction: float
    deforestation_detritus_fraction: float
    deforestation_no_regrowth_fraction: float
    nitrogen: NitrogenParameters | None

class PrognosticFixation(TypedDict):
    """Michaelis-Menten nitrogen fixation from NPP."""

    maximum_fixation_rate: float
    half_saturation_npp: float

class NitrogenParameters(TypedDict, total=False):
    """Parameters for the nitrogen limitation of NPP.

    All fields optional - defaults provided by Rust.
    """

    uptake_nc_ratio: float
    recycling_nc_ratio: float
    recycling_cn_factor: float
    uptake_supply: float
    availability_factor_a: float
    availability_factor_c: float
    loss_turnover_time: float
    prognostic_fixation: PrognosticFixation | None

@final
class TerrestrialCarbonBuilder(ComponentBuilder):
    """Builder for the terrestrial carbon cycle component.

    Implements the MAGICC7 3-pool terrestrial carbon model with three CO2
    fertilisation methods, two respiration methods, time-varying turnover
    times, no-feedback reference pools, and mass conservation correction.

    Inputs, outputs and states are listed on the
    [TerrestrialCarbon component page][component-terrestrialcarbon].

    Examples
    --------
    >>> builder = TerrestrialCarbonBuilder.from_parameters(
    ...     {
    ...         "npp_pi": 66.27,
    ...         "fertilisation_factor": 0.6486,
    ...         "fertilisation_method": 1.10,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(  # type: ignore[override]
        parameters: TerrestrialCarbonParameters,
    ) -> TerrestrialCarbonBuilder:
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the terrestrial carbon cycle component."""

class OceanCarbonParameters(TypedDict, total=False):
    """Parameters for ocean carbon cycle component.

    All fields optional - defaults provided by Rust.
    """

    co2_pi: float
    pco2_pi: float | None
    gas_exchange_scale: float
    gas_exchange_tau: float
    temperature_sensitivity: float
    irf_scale: float
    mixed_layer_depth: float
    ocean_surface_area: float
    sst_pi: float
    steps_per_year: int
    maximum_history_months: int
    irf_switch_time: float
    irf_early: dict[str, Any]
    irf_late: dict[str, Any]
    delta_ospp_offsets: list[float]
    delta_ospp_coefficients: list[float]
    include_temperature_feedback: bool
    temperature_feedback_start_year: float
    flux_change_limit: float
    gtc_per_ppm: float
    include_permafrost: bool
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
    - `temperature_sensitivity` (`float`, default `0.03717879`): Temperature sensitivity
      of pCO2 (K^-1).
    - `irf_scale` (`float`, default `0.9492864`): IRF scaling factor.
    - `mixed_layer_depth` (`float`, default `50.9`): Mixed layer depth (m).
    - `ocean_surface_area` (`float`, default `3.55e14`): Ocean surface area (m^2).
    - `sst_pi` (`float`, default `17.7`): Pre-industrial sea surface temperature (C).
    - `steps_per_year` (`int`, default `12`): Substeps per year.
    - `maximum_history_months` (`int`, default `6000`): Maximum flux history length
      (months).
    - `include_temperature_feedback` (`bool`, default `True`): Enable temperature
      feedback on pCO2.
    - `temperature_feedback_start_year` (`float`, default `1900.0`): Year from which
      warming drives the pCO2 feedback.
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
    ...         "include_temperature_feedback": True,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: OceanCarbonParameters) -> OceanCarbonBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the ocean carbon cycle component."""

class CO2BudgetParameters(TypedDict, total=False):
    """Parameters for CO2 budget component.

    All fields optional - defaults provided by Rust.
    """

    gtc_per_ppm: float
    include_permafrost: bool
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
    def from_parameters(parameters: CO2BudgetParameters) -> CO2BudgetBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the CO2 budget component."""

class PermafrostSoil(TypedDict, total=False):
    """Parameters of one permafrost soil type (mineral soil or peat)."""

    thaw_rate: float
    thaw_exponent: float
    decomposition_ratio: float
    q10_aerobic_alpha: float
    q10_anaerobic_alpha: float
    initial_anaerobic_fraction: float
    maximum_anaerobic_fraction: float
    anaerobic_moisture_sensitivity: float
    ch4_oxidation_fraction: float

class PermafrostParameters(TypedDict, total=False):
    """Parameters for the permafrost component.

    All fields optional - defaults provided by Rust.
    Fields missing from ``mineral_soil`` or ``peat`` keep that soil's defaults.
    """

    n_bands: int
    minimum_melting_temperature: float
    maximum_melting_temperature: float
    arctic_amplification: float
    soil_annual_cycle_amplitude: float
    aerobic_turnover_time: float
    anaerobic_decomposition_ratio: float
    q10_temperature1: float
    q10_temperature2: float
    soil_water_slope: float
    soil_water_offset: float
    minimum_soil_water: float
    total_pool: float
    mineral_soil_fraction_south: float
    mineral_soil_fraction_north: float
    zonal_pool_distribution: float
    mineral_soil: PermafrostSoil
    peat: PermafrostSoil

@final
class PermafrostBuilder(ComponentBuilder):
    """Builder for the permafrost carbon feedback component.

    Thaws zonal bands of mineral soil and peat carbon as Arctic warming exceeds
    each band's melting threshold, then decomposes the thawed carbon to CO2 and CH4.
    Set ``include_permafrost`` on CH4 chemistry, CH4 oxidation, ocean carbon and the CO2
    budget to feed the emissions back.

    Inputs, outputs and states are listed on the
    [Permafrost component page][component-permafrost].

    Examples
    --------
    >>> builder = PermafrostBuilder.from_parameters({"total_pool": 800.0})
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: PermafrostParameters) -> PermafrostBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the permafrost component."""

# Forcing Components

class OzoneForcingParameters(TypedDict, total=False):
    """Parameters for the ozone forcing component.

    All fields optional - defaults provided by Rust.
    """

    stratospheric_o3_scale: float
    stratospheric_cl_exponent: float
    stratospheric_reference_year: float
    tropospheric_ozone_radiative_efficiency: float
    tropospheric_ozone_ch4_sensitivity: float
    tropospheric_ozone_nox_sensitivity: float
    tropospheric_ozone_co_sensitivity: float
    tropospheric_ozone_nmvoc_sensitivity: float
    n2o_ozone_radiative_efficiency: float
    temperature_scale: float
    tropospheric_ozone_regions: TroposphericOzoneRegions | None

class TroposphericOzoneRegions(TypedDict, total=False):
    """Regional pattern of tropospheric ozone forcing, normalised per hemisphere."""

    regional_pattern: list[float]
    nh_land_fraction: float
    sh_land_fraction: float

@final
class OzoneForcingBuilder(ComponentBuilder):
    """Builder for the ozone forcing component.

    Changes are relative to the first year of the run.

    **Inputs**

    - `Equivalent Effective Stratospheric Chlorine` (ppt).
    - `Atmospheric Concentrations|CH4`: CH4 concentration (ppb).
    - `Atmospheric Concentrations|N2O`: N2O concentration (ppb).
    - `Emissions|{NOx,CO,NMVOC}|{Fossil,Biosphere}`: Precursor emissions by sector
      (Mt N/yr, Mt CO/yr, Mt NMVOC/yr), lagged by a year.
    - `Surface Air Temperature Change`: Global mean surface air temperature change (K),
      lagged by a year.

    **Outputs**

    - `Effective Radiative Forcing|Ozone|Stratospheric`: Stratospheric ozone
      forcing from EESC above its threshold-year value (W/m^2).
    - `Effective Radiative Forcing|Ozone|Tropospheric` (float or FourBox): Tropospheric
      ozone forcing from CH4 and precursors (W/m^2), four-box with
      ``tropospheric_ozone_regions``.
    - `Effective Radiative Forcing|Ozone|N2O Feedback`: Ozone forcing from N2O
      (W/m^2).
    - `Effective Radiative Forcing|Ozone|Temperature Feedback`: Ozone response to
      warming (W/m^2).
    """

    @staticmethod
    def from_parameters(parameters: OzoneForcingParameters) -> OzoneForcingBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the ozone forcing component."""

class AerosolDirectParameters(TypedDict, total=False):
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
    include_harmonisation: bool
    harmonisation_reference_year: float
    harmonisation_target: float

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
    - `include_harmonisation` (`bool`, default `False`): Enable harmonisation to
      reference year.
    - `harmonisation_reference_year` (`float`, default `2019.0`): Reference year for
      harmonisation.
    - `harmonisation_target` (`float`, default `-0.22`): Target forcing at reference
      year (W/m^2).

    Inputs, outputs and states are listed on the
    [AerosolDirect component page][component-aerosoldirect].

    Examples
    --------
    >>> builder = AerosolDirectBuilder.from_parameters(
    ...     {
    ...         "sox_coefficient": -0.0035,
    ...         "bc_coefficient": 0.0077,
    ...         "include_harmonisation": False,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: AerosolDirectParameters) -> AerosolDirectBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the direct aerosol forcing component."""

class AerosolIndirectParameters(TypedDict, total=False):
    """Parameters for indirect aerosol forcing component.

    All fields optional - defaults provided by Rust.
    """

    cloud_albedo_coefficient: float
    reference_burden: float
    sox_weight: float
    oc_weight: float
    sox_pi: float
    oc_pi: float
    include_harmonisation: bool
    harmonisation_reference_year: float
    harmonisation_target: float

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
    - `include_harmonisation` (`bool`, default `False`): Enable harmonisation to
      reference year.
    - `harmonisation_reference_year` (`float`, default `2019.0`): Reference year for
      harmonisation.
    - `harmonisation_target` (`float`, default `-0.89`): Target forcing at reference
      year (W/m^2).

    Inputs, outputs and states are listed on the
    [AerosolIndirect component page][component-aerosolindirect].

    Examples
    --------
    >>> builder = AerosolIndirectBuilder.from_parameters(
    ...     {
    ...         "cloud_albedo_coefficient": -1.0,
    ...         "reference_burden": 50.0,
    ...         "include_harmonisation": False,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(  # type: ignore[override]
        parameters: AerosolIndirectParameters,
    ) -> AerosolIndirectBuilder:
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the indirect aerosol forcing component."""

# GHG Forcing

class GHGForcingParameters(TypedDict, total=False):
    """Parameters for GHGForcing component.

    All fields optional - defaults provided by Rust.
    """

    method: Literal["IPCCTAR", "OLBL"]
    co2_pi: float
    ch4_pi: float
    n2o_pi: float
    erf_2xco2: float
    ch4_radiative_efficiency: float
    n2o_radiative_efficiency: float
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
    ch4_stratospheric_h2o_fraction: float

@final
class GHGForcingBuilder(ComponentBuilder):
    """Builder for the GHG radiative forcing component.

    Calculates effective radiative forcing from CO2, CH4, and N2O
    concentrations using IPCCTAR or OLBL methods with optional
    rapid adjustment factors.

    **Parameter keys**

    - `method` (`str`, default `"OLBL"`): Forcing method: "IPCCTAR" or "OLBL".
    - `co2_pi` (`float`, default `278.0`): Pre-industrial CO2 (ppm).
    - `ch4_pi` (`float`, default `722.0`): Pre-industrial CH4 (ppb).
    - `n2o_pi` (`float`, default `270.0`): Pre-industrial N2O (ppb).
    - `erf_2xco2` (`float`, default `3.71`): 2xCO2 effective radiative forcing (W/m^2).
    - `adjust_co2` (`float`, default `1.05`): Rapid adjustment factor for CO2.
    - `adjust_ch4` (`float`, default `0.86`): Rapid adjustment factor for CH4.
    - `adjust_n2o` (`float`, default `1.0`): Rapid adjustment factor for N2O.

    Inputs, outputs and states are listed on the
    [GHGForcing component page][component-ghgforcing].

    Examples
    --------
    >>> builder = GHGForcingBuilder.from_parameters(
    ...     {
    ...         "method": "IPCCTAR",
    ...         "erf_2xco2": 3.71,
    ...         "adjust_co2": 1.05,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: GHGForcingParameters) -> GHGForcingBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the GHG radiative forcing component."""
    def calculate_forcings(
        self, co2: float, ch4: float, n2o: float
    ) -> dict[str, float]:
        """Evaluate boundary ERF in W/m², keyed by CO2, CH4 and N2O.

        Concentrations must be finite and positive, in ppm, ppb and ppb.
        """
    def erf_2xco2(self) -> float:
        """ERF in W/m² from doubling CO2 above ``co2_pi``.

        CH4 and N2O are held at their preindustrial references. MAGICC7 derives
        its climate feedback from this value.
        """

# Forcing aggregation

class ForcingAgent(TypedDict, total=False):
    """A forcing agent read by ForcingAggregation.

    Only ``variable`` and ``category`` are required.
    """

    variable: Required[str]
    category: Required[ForcingCategory]
    """Subtotals the agent's ERF counts towards."""
    regional: bool
    regional_pattern: tuple[float, float, float, float]
    hemispheric: bool
    use_co2_internal_efficacy: bool
    """Take the internal efficacy from LAMCALC's CO2 response, as MAGICC7 does."""
    efficacy: float
    internal_efficacy: float
    sum_of: list[str]
    """Variables added in order into this agent before its efficacy applies.

    Empty reads ``variable`` itself.
    """

type ForcingCategory = Literal["greenhouse_gas", "aerosol", "anthropogenic", "natural"]
"""Which of the greenhouse gas, aerosol and anthropogenic ERF an agent counts towards.

Every category but ``natural`` counts towards the anthropogenic ERF.
"""

type ForcingTerm = str | list[ForcingTerm]
"""An agent ``variable``, or a list of terms summed before the enclosing sum."""

class EfficacyClimate(TypedDict, total=False):
    """Climate settings for recalculating internal efficacies each year.

    The efficacies come from LAMCALC at the previous year's
    ``Equilibrium Climate Sensitivity`` and ``2xCO2 Effective Radiative Forcing``.
    """

    ecs: float
    erf_2xco2: float
    land_ocean_heat_exchange: float
    north_south_heat_exchange: float
    land_ocean_warming_ratio: float
    amplify_ocean_to_land: float
    co2_regional_pattern: tuple[float, float, float, float]

class ForcingAggregationParameters(TypedDict, total=False):
    """Parameters for ForcingAggregation."""

    agents: list[ForcingAgent]
    summation: list[ForcingTerm]
    """Order in which agents are added, by ``variable``.

    Nested lists are added first. Empty adds the agents in order.
    """
    efficacy_method: Literal["none", "unit", "ar6"]
    nh_land_fraction: float
    sh_land_fraction: float
    climate: EfficacyClimate

@final
class ForcingAggregationBuilder(ComponentBuilder):
    """Builder for the component combining agent ERF into four-box climate forcing.

    Reads each agent's ERF and writes the total effective radiative forcing and
    the four-box ``Efficacy Adjusted Effective Radiative Forcing`` read by
    ClimateUDEB when ``include_regional_forcing`` is set.
    """

    @staticmethod
    def from_parameters(  # type: ignore[override]
        parameters: ForcingAggregationParameters,
    ) -> ForcingAggregationBuilder:
        """Create a builder from a parameter dictionary."""
    def aggregate(
        self, values: list[list[float]]
    ) -> tuple[float, tuple[float, float, float, float]]:
        """Return total effective radiative forcing and box forcing.

        Takes each agent's value or box values.
        """
    def internal_efficacies(self, ecs: float, erf_2xco2: float) -> list[float]:
        """Return each agent's internal efficacy at climate sensitivity ``ecs``.

        ``erf_2xco2`` is the 2xCO2 effective radiative forcing in W/m².
        Without ``climate`` these are the agents' own ``internal_efficacy``.
        """
    def build(self) -> RustComponent:
        """Build the forcing aggregation component."""

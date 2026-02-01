"""MAGICC climate model components.

This module provides Rust implementations of MAGICC (Model for the Assessment of
Greenhouse gas Induced Climate Change) component builders.

MAGICC is a reduced-complexity climate model widely used for climate policy
analysis. The components in this module implement MAGICC's key physical processes.

Available Components
--------------------
Climate:
- ClimateUDEBBuilder: 4-box UDEB climate model with ocean diffusion

Chemistry:
- CH4ChemistryBuilder: CH4 atmospheric chemistry with Prather iteration
- N2OChemistryBuilder: N2O atmospheric chemistry with stratospheric delay
- HalocarbonChemistryBuilder: F-gases and Montreal Protocol gases

Carbon Cycle:
- TerrestrialCarbonBuilder: 4-pool terrestrial carbon cycle
- OceanCarbonBuilder: IRF-based ocean carbon uptake
- CO2BudgetBuilder: Carbon mass balance integrator

Forcing:
- OzoneForcingBuilder: Stratospheric and tropospheric ozone forcing
- AerosolDirectBuilder: Direct aerosol radiative effects
- AerosolIndirectBuilder: Indirect aerosol cloud effects
"""

from typing import TypedDict, final

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
    temp_adjust_alpha: float
    temp_adjust_gamma: float
    polar_sinking_ratio: float
    steps_per_year: int
    max_temperature: float

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

    @staticmethod
    def from_parameters(parameters: ClimateUDEBParams) -> ClimateUDEBBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the UDEB climate model component."""

# Chemistry Components

class CH4ChemistryParams(TypedDict, total=False):
    """Parameters for CH4 chemistry component.

    All fields optional - defaults provided by Rust.
    """

    ch4_pi: float
    natural_emissions: float
    tau_oh: float
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
    ppb_to_tg: float
    nox_reference: float
    co_reference: float
    nmvoc_reference: float

@final
class CH4ChemistryBuilder(ComponentBuilder):
    """Builder for the CH4 chemistry component.

    Implements atmospheric methane chemistry with the Prather iteration method,
    including OH feedback and temperature sensitivity.

    # Parameters
    # ----------
    # ch4_pi : float
    #     Pre-industrial CH4 concentration (ppb). Default: 722.0
    # natural_emissions : float
    #     Natural CH4 emissions (Tg CH4/yr). Default: 209.0
    # tau_oh : float
    #     Base tropospheric OH sink lifetime (years). Default: 9.3
    # tau_soil : float
    #     Soil uptake sink lifetime (years). Default: 150.0
    # tau_strat : float
    #     Stratospheric sink lifetime (years). Default: 120.0
    # tau_trop_cl : float
    #     Tropospheric Cl sink lifetime (years). Default: 200.0
    # ch4_self_feedback : float
    #     CH4 self-feedback coefficient. Default: -0.32
    # oh_sensitivity_scale : float
    #     OH sensitivity scaling factor. Default: 0.72
    # oh_nox_sensitivity : float
    #     NOx emissions effect on OH ((Tg N/yr)^-1). Default: 0.0042
    # oh_co_sensitivity : float
    #     CO emissions effect on OH ((Tg CO/yr)^-1). Default: -0.000105
    # oh_nmvoc_sensitivity : float
    #     NMVOC emissions effect on OH ((Tg NMVOC/yr)^-1). Default: -0.000315
    # temp_sensitivity : float
    #     Temperature sensitivity of lifetime (K^-1). Default: 0.0316
    # include_temp_feedback : bool
    #     Enable temperature feedback. Default: True
    # include_emissions_feedback : bool
    #     Enable NOx/CO/NMVOC emissions feedback. Default: True
    # ppb_to_tg : float
    #     Conversion factor ppb to Tg CH4. Default: 2.75
    # nox_reference : float
    #     Reference NOx emissions (Tg N/yr). Default: 0.0
    # co_reference : float
    #     Reference CO emissions (Tg CO/yr). Default: 0.0
    # nmvoc_reference : float
    #     Reference NMVOC emissions (Tg NMVOC/yr). Default: 0.0

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

    Examples
    --------
    >>> builder = CH4ChemistryBuilder.from_parameters(
    ...     {
    ...         "ch4_pi": 722.0,
    ...         "tau_oh": 9.3,
    ...         "ch4_self_feedback": -0.32,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: CH4ChemistryParams) -> CH4ChemistryBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the CH4 chemistry component."""

class N2OChemistryParams(TypedDict, total=False):
    """Parameters for N2O chemistry component.

    All fields optional - defaults provided by Rust.
    """

    n2o_pi: float
    natural_emissions: float
    tau_n2o: float
    lifetime_feedback: float
    strat_delay: int
    ppb_to_tg: float

@final
class N2OChemistryBuilder(ComponentBuilder):
    """Builder for the N2O chemistry component.

    Implements atmospheric nitrous oxide chemistry with concentration-dependent
    lifetime feedback and stratospheric transport delay.

    # Parameters
    # ----------
    # n2o_pi : float
    #     Pre-industrial N2O concentration (ppb). Default: 270.0
    # natural_emissions : float
    #     Natural N2O emissions (Tg N/yr). Default: 11.0
    # tau_n2o : float
    #     Base atmospheric lifetime (years). Default: 139.275
    # lifetime_feedback : float
    #     Lifetime feedback exponent. Default: -0.04
    # strat_delay : int
    #     Stratospheric mixing delay (years). Default: 1
    # ppb_to_tg : float
    #     Conversion factor ppb to Tg N. Default: 4.79

    Inputs
    ------
    Emissions|N2O : float
        Anthropogenic N2O emissions (Tg N/yr)

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
    ...         "n2o_pi": 270.0,
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

class HalocarbonChemistryParams(TypedDict, total=False):
    """Parameters for halocarbon chemistry component.

    All fields optional - defaults provided by Rust.
    """

    br_multiplier: float
    cfc11_release_normalisation: float
    eesc_delay: float
    air_molar_mass: float
    atmospheric_mass_tg: float
    mixing_box_fraction: float

@final
class HalocarbonChemistryBuilder(ComponentBuilder):
    """Builder for the halocarbon chemistry component.

    Implements exponential decay for F-gases and Montreal Protocol gases,
    calculating concentrations and EESC (Equivalent Effective Stratospheric Chlorine).

    # Parameters
    # ----------
    # fgases : list[dict]
    #     F-gas species data (HFCs, PFCs, SF6, etc.). Each dict contains:
    #     name, lifetime, radiative_efficiency, concentration_pi, molecular_weight,
    #     n_cl, n_br, fractional_release
    # montreal_gases : list[dict]
    #     Montreal Protocol gases (CFCs, HCFCs, halons). Same format as fgases
    # br_multiplier : float
    #     Bromine efficiency multiplier for EESC. Default: 60.0
    # cfc11_release_normalisation : float
    #     CFC-11 fractional release factor for EESC normalisation. Default: 0.47
    # eesc_delay : float
    #     EESC stratospheric mixing delay (years). Default: 3.0
    # air_molar_mass : float
    #     Molar mass of air (g/mol). Default: 28.97
    # atmospheric_mass_tg : float
    #     Total atmospheric mass (Tg). Default: 5.133e9
    # mixing_box_fraction : float
    #     Effective mixing box fraction. Default: 0.949

    Inputs
    ------
    Emissions|{Species} : float
        Emissions for each halocarbon species (kt/yr)

    States
    ------
    Atmospheric Concentration|{Species} : float
        Concentration for each halocarbon species (ppt)

    Outputs
    -------
    Forcing|Halocarbons : float
        Total halocarbon forcing (W/m^2)
    Forcing|F-gases : float
        F-gas forcing (W/m^2)
    Forcing|Montreal Gases : float
        Montreal Protocol gas forcing (W/m^2)
    EESC : float
        Equivalent Effective Stratospheric Chlorine (ppt)

    Examples
    --------
    >>> builder = HalocarbonChemistryBuilder.from_parameters(
    ...     {
    ...         "br_multiplier": 60.0,
    ...         "eesc_delay": 3.0,
    ...     }
    ... )
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
    npp_temp_sensitivity: float
    resp_temp_sensitivity: float
    detritus_temp_sensitivity: float
    soil_temp_sensitivity: float
    humus_temp_sensitivity: float
    plant_pool_pi: float
    detritus_pool_pi: float
    soil_pool_pi: float
    humus_pool_pi: float
    respiration_pi: float
    frac_npp_to_plant: float
    frac_npp_to_detritus: float
    frac_plant_to_detritus: float
    frac_detritus_to_soil: float
    frac_soil_to_humus: float
    enable_fertilization: bool
    enable_temp_feedback: bool

@final
class TerrestrialCarbonBuilder(ComponentBuilder):
    """Builder for the terrestrial carbon cycle component.

    Implements a 4-pool terrestrial carbon model with CO2 fertilization and
    temperature feedbacks (plant biomass, detritus, soil, humus).

    # Parameters
    # ----------
    # npp_pi : float
    #     Pre-industrial Net Primary Production (GtC/yr). Default: 66.27
    # co2_pi : float
    #     Pre-industrial CO2 concentration (ppm). Default: 278.0
    # beta : float
    #     CO2 fertilization factor. Default: 0.6486
    # npp_temp_sensitivity : float
    #     NPP temperature sensitivity coefficient (K^-1). Default: 0.0107
    # resp_temp_sensitivity : float
    #     Respiration temperature sensitivity (K^-1). Default: 0.0685
    # detritus_temp_sensitivity : float
    #     Detritus decay temperature sensitivity (K^-1). Default: 0.1358
    # soil_temp_sensitivity : float
    #     Soil decay temperature sensitivity (K^-1). Default: 0.1541
    # humus_temp_sensitivity : float
    #     Humus decay temperature sensitivity (K^-1). Default: 0.05
    # plant_pool_pi : float
    #     Pre-industrial plant pool (GtC). Default: 884.86
    # detritus_pool_pi : float
    #     Pre-industrial detritus pool (GtC). Default: 92.77
    # soil_pool_pi : float
    #     Pre-industrial soil pool (GtC). Default: 1681.53
    # humus_pool_pi : float
    #     Pre-industrial humus pool (GtC). Default: 836.0
    # respiration_pi : float
    #     Pre-industrial respiration (GtC/yr). Default: 12.26
    # frac_npp_to_plant : float
    #     Fraction of NPP to plant pool. Default: 0.4483
    # frac_npp_to_detritus : float
    #     Fraction of NPP to detritus pool. Default: 0.3998
    # frac_plant_to_detritus : float
    #     Fraction of plant turnover to detritus. Default: 0.9989
    # frac_detritus_to_soil : float
    #     Fraction of detritus decay to soil. Default: 0.3
    # frac_soil_to_humus : float
    #     Fraction of soil decay to humus. Default: 0.1
    # enable_fertilization : bool
    #     Enable CO2 fertilization feedback. Default: True
    # enable_temp_feedback : bool
    #     Enable temperature feedback. Default: True

    Inputs
    ------
    Atmospheric Concentration|CO2 : float
        Atmospheric CO2 concentration (ppm)
    Surface Temperature|Global : float
        Global mean surface temperature anomaly (K)
    Emissions|CO2|Land Use : float
        Land use change CO2 emissions (GtC/yr)

    States
    ------
    Carbon Pool|Plant : float
        Plant carbon pool (GtC)
    Carbon Pool|Detritus : float
        Detritus carbon pool (GtC)
    Carbon Pool|Soil : float
        Soil carbon pool (GtC)
    Carbon Pool|Humus : float
        Humus carbon pool (GtC)

    Outputs
    -------
    Carbon Flux|Terrestrial : float
        Net terrestrial carbon uptake (GtC/yr)

    Examples
    --------
    >>> builder = TerrestrialCarbonBuilder.from_parameters(
    ...     {
    ...         "npp_pi": 66.27,
    ...         "beta": 0.6486,
    ...         "enable_fertilization": True,
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
    pco2_pi: float
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
    irf_early_coefficients: list[float]
    irf_early_timescales: list[float]
    irf_late_coefficients: list[float]
    irf_late_timescales: list[float]
    delta_ospp_offsets: list[float]
    delta_ospp_coefficients: list[float]
    enable_temp_feedback: bool

@final
class OceanCarbonBuilder(ComponentBuilder):
    """Builder for the ocean carbon cycle component.

    Implements an IRF-based ocean carbon model with air-sea exchange and
    temperature feedback using the 2D-BERN impulse response function.

    # Parameters
    # ----------
    # co2_pi : float
    #     Pre-industrial atmospheric CO2 (ppm). Default: 278.0
    # pco2_pi : float
    #     Pre-industrial ocean surface pCO2 (ppm). Default: 278.0
    # gas_exchange_scale : float
    #     Gas exchange rate scaling factor. Default: 1.833492
    # gas_exchange_tau : float
    #     Gas exchange timescale (years). Default: 7.46
    # temp_sensitivity : float
    #     Temperature sensitivity of pCO2 (K^-1). Default: 0.0423
    # irf_scale : float
    #     IRF scaling factor. Default: 0.9492864
    # mixed_layer_depth : float
    #     Mixed layer depth (m). Default: 50.0
    # ocean_surface_area : float
    #     Ocean surface area (m^2). Default: 3.5375e14
    # sst_pi : float
    #     Pre-industrial sea surface temperature (C). Default: 18.2997
    # steps_per_year : int
    #     Sub-steps per year. Default: 12
    # max_history_months : int
    #     Maximum flux history length (months). Default: 6000
    # irf_switch_time : float
    #     IRF switch time (years). Default: 9.9
    # irf_early_coefficients : list[float]
    #     Early IRF exponential coefficients
    # irf_early_timescales : list[float]
    #     Early IRF exponential timescales (years)
    # irf_late_coefficients : list[float]
    #     Late IRF exponential coefficients
    # irf_late_timescales : list[float]
    #     Late IRF exponential timescales (years)
    # delta_ospp_offsets : list[float]
    #     Joos A24 polynomial offsets (length 5)
    # delta_ospp_coefficients : list[float]
    #     Joos A24 polynomial coefficients (length 5)
    # enable_temp_feedback : bool
    #     Enable temperature feedback on pCO2. Default: True

    Inputs
    ------
    Atmospheric Concentration|CO2 : float
        Atmospheric CO2 concentration (ppm)
    Sea Surface Temperature : float
        Sea surface temperature anomaly (K)

    States
    ------
    Ocean Surface pCO2 : float
        Ocean surface pCO2 (ppm)
    Cumulative Ocean Uptake : float
        Cumulative ocean carbon uptake (GtC)

    Outputs
    -------
    Carbon Flux|Ocean : float
        Net ocean carbon uptake (GtC/yr)

    Examples
    --------
    >>> builder = OceanCarbonBuilder.from_parameters(
    ...     {
    ...         "co2_pi": 278.0,
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
    co2_pi: float

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
    Emissions|CO2|Land Use : float
        Land use change CO2 emissions (GtC/yr)
    Carbon Flux|Terrestrial : float
        Terrestrial carbon uptake (GtC/yr)
    Carbon Flux|Ocean : float
        Ocean carbon uptake (GtC/yr)

    States
    ------
    Atmospheric Concentration|CO2 : float
        Atmospheric CO2 concentration (ppm)

    Outputs
    -------
    Emissions|CO2|Net : float
        Net CO2 emissions (GtC/yr)
    Airborne Fraction|CO2 : float
        Airborne fraction of CO2 emissions (dimensionless)

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

# Forcing Components

class OzoneForcingParams(TypedDict, total=False):
    """Parameters for ozone forcing component.

    All fields optional - defaults provided by Rust.
    """

    eesc_reference: float
    strat_o3_scale: float
    strat_cl_exponent: float
    trop_radeff: float
    trop_oz_ch4: float
    trop_oz_nox: float
    trop_oz_co: float
    trop_oz_voc: float
    ch4_pi: float
    nox_pi: float
    co_pi: float
    nmvoc_pi: float
    temp_feedback_scale: float

@final
class OzoneForcingBuilder(ComponentBuilder):
    """Builder for the ozone forcing component.

    Calculates stratospheric and tropospheric ozone radiative forcing with
    temperature feedback.

    # Parameters
    # ----------
    # eesc_reference : float
    #     EESC reference value at threshold year (ppt). Default: 1420.0
    # strat_o3_scale : float
    #     Stratospheric ozone forcing scale (W/m^2). Default: -0.0043
    # strat_cl_exponent : float
    #     Power-law exponent for EESC-RF relationship. Default: 1.7
    # trop_radeff : float
    #     Tropospheric radiative efficiency (W/m^2/DU). Default: 0.032
    # trop_oz_ch4 : float
    #     Ozone change per ln(CH4/CH4_pi) (DU). Default: 5.7
    # trop_oz_nox : float
    #     NOx sensitivity (DU/(Mt N/yr)). Default: 0.168
    # trop_oz_co : float
    #     CO sensitivity (DU/(Mt CO/yr)). Default: 0.00396
    # trop_oz_voc : float
    #     NMVOC sensitivity (DU/(Mt NMVOC/yr)). Default: 0.01008
    # ch4_pi : float
    #     Pre-industrial CH4 concentration (ppb). Default: 700.0
    # nox_pi : float
    #     Pre-industrial NOx emissions (Mt N/yr). Default: 0.0
    # co_pi : float
    #     Pre-industrial CO emissions (Mt CO/yr). Default: 0.0
    # nmvoc_pi : float
    #     Pre-industrial NMVOC emissions (Mt NMVOC/yr). Default: 0.0
    # temp_feedback_scale : float
    #     Temperature feedback coefficient (W/m^2/K). Default: -0.037

    Inputs
    ------
    EESC : float
        Equivalent Effective Stratospheric Chlorine (ppt)
    Atmospheric Concentration|CH4 : float
        CH4 concentration (ppb)
    Emissions|NOx : float
        NOx emissions (Mt N/yr)
    Emissions|CO : float
        CO emissions (Mt CO/yr)
    Emissions|NMVOC : float
        NMVOC emissions (Mt NMVOC/yr)
    Surface Temperature|Global : float
        Global mean surface temperature anomaly (K)

    Outputs
    -------
    Effective Radiative Forcing|O3|Stratospheric : float
        Stratospheric ozone forcing (W/m^2)
    Effective Radiative Forcing|O3|Tropospheric : float
        Tropospheric ozone forcing (W/m^2)
    Effective Radiative Forcing|O3|Temperature Feedback : float
        Ozone temperature feedback forcing (W/m^2)

    Examples
    --------
    >>> builder = OzoneForcingBuilder.from_parameters(
    ...     {
    ...         "eesc_reference": 1420.0,
    ...         "strat_o3_scale": -0.0043,
    ...         "trop_radeff": 0.032,
    ...     }
    ... )
    >>> component = builder.build()
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

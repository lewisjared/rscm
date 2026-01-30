"""Generic climate model components.

This module provides reusable climate model components that can be combined
to build custom climate models. Each component is defined in Rust and exposed
to Python through PyO3 bindings.

Components are instantiated using the builder pattern:

    builder = CO2ERFBuilder.from_parameters({
        "erf_2xco2": 3.7,
        "conc_pi": 278.0,
    })
    component = builder.build()

The built component can then be added to a model via ModelBuilder.
"""

from typing import final

from rscm._lib.core import Component, ComponentBuilder

@final
class CarbonCycleBuilder(ComponentBuilder):
    """Builder for the one-box carbon cycle component.

    This component models the carbon cycle using a simple one-box model where:
    - CO2 emissions increase atmospheric concentrations
    - Land uptake removes CO2 at a rate that depends on the concentration anomaly
    - The uptake rate is temperature-dependent

    # Parameters
    # ----------
    # tau : float
    #     Timescale of the box's response (years).
    # conc_pi : float
    #     Pre-industrial atmospheric CO2 concentration (ppm).
    # alpha_temperature : float
    #     Sensitivity of lifetime to changes in global-mean temperature (1/K).

    Inputs
    ------
    Emissions|CO2|Anthropogenic : float
        Anthropogenic CO2 emissions (GtC/yr)
    Surface Temperature : float
        Global mean surface temperature anomaly (K)

    Outputs
    -------
    Atmospheric Concentration|CO2 : float
        Atmospheric CO2 concentration (ppm)
    Cumulative Emissions|CO2 : float
        Cumulative CO2 emissions (GtC)
    Cumulative Land Uptake : float
        Cumulative land carbon uptake (GtC)

    Examples
    --------
    >>> builder = CarbonCycleBuilder.from_parameters(
    ...     {
    ...         "tau": 30.0,
    ...         "conc_pi": 278.0,
    ...         "alpha_temperature": 0.0,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: dict[str, float]) -> CarbonCycleBuilder:
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the carbon cycle component."""

@final
class CO2ERFBuilder(ComponentBuilder):
    """Builder for the CO2 effective radiative forcing component.

    Computes effective radiative forcing (ERF) from CO2 concentrations using
    the standard logarithmic relationship:

        ERF = (ERF_2xCO2 / ln(2)) * ln(1 + (C - C0) / C0)

    where ERF_2xCO2 is the forcing for a CO2 doubling, C is the current
    concentration, and C0 is the pre-industrial concentration.

    # Parameters
    # ----------
    # erf_2xco2 : float
    #     ERF due to a doubling of atmospheric CO2 concentrations (W/m^2).
    #     Typical value: 3.7 W/m^2.
    # conc_pi : float
    #     Pre-industrial atmospheric CO2 concentration (ppm).
    #     Typical value: 278 ppm.

    Inputs
    ------
    Atmospheric Concentration|CO2 : float
        Atmospheric CO2 concentration (ppm)

    Outputs
    -------
    Effective Radiative Forcing|CO2 : float
        CO2 effective radiative forcing (W/m^2)

    Examples
    --------
    >>> builder = CO2ERFBuilder.from_parameters(
    ...     {
    ...         "erf_2xco2": 3.7,
    ...         "conc_pi": 278.0,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: dict[str, float]) -> CO2ERFBuilder:
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the CO2 ERF component."""

@final
class FourBoxOceanHeatUptakeBuilder(ComponentBuilder):
    """Builder for the four-box ocean heat uptake component.

    This component takes scalar effective radiative forcing (ERF) as input
    and produces regional ocean heat uptake values for a four-box grid
    (Northern Ocean, Northern Land, Southern Ocean, Southern Land).

    The component demonstrates disaggregation from scalar to grid resolution.
    Regional uptake is calculated as: regional = global_ERF * ratio

    # Parameters
    # ----------
    # northern_ocean_ratio : float
    #     Ratio of Northern Ocean uptake to global ERF. Default: 1.2
    # northern_land_ratio : float
    #     Ratio of Northern Land uptake to global ERF. Default: 0.6
    # southern_ocean_ratio : float
    #     Ratio of Southern Ocean uptake to global ERF. Default: 1.6
    # southern_land_ratio : float
    #     Ratio of Southern Land uptake to global ERF. Default: 0.6

    Note: Ratios must average to 1.0 for conservation.

    Inputs
    ------
    Effective Radiative Forcing|Aggregated : float
        Global mean effective radiative forcing (W/m^2)

    Outputs
    -------
    Heat Uptake|Ocean : FourBoxSlice
        Regional ocean heat uptake (W/m^2) for each of the four boxes

    Examples
    --------
    >>> builder = FourBoxOceanHeatUptakeBuilder.from_parameters(
    ...     {
    ...         "northern_ocean_ratio": 1.2,
    ...         "northern_land_ratio": 0.6,
    ...         "southern_ocean_ratio": 1.6,
    ...         "southern_land_ratio": 0.6,
    ...     }
    ... )
    >>> component = builder.build()
    """

    @staticmethod
    def from_parameters(parameters: dict[str, float]) -> FourBoxOceanHeatUptakeBuilder:
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the four-box ocean heat uptake component."""

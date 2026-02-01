"""MAGICC climate model components.

This module provides component builders for MAGICC-derived climate processes,
organised by domain:

- Climate: ClimateUDEBBuilder (upwelling-diffusion energy balance)
- Chemistry: CH4ChemistryBuilder, N2OChemistryBuilder, HalocarbonChemistryBuilder
- Carbon: TerrestrialCarbonBuilder, OceanCarbonBuilder, CO2BudgetBuilder
- Forcing: OzoneForcingBuilder, AerosolDirectBuilder, AerosolIndirectBuilder
"""

from rscm._lib.magicc import (
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

__all__ = [
    "AerosolDirectBuilder",
    "AerosolIndirectBuilder",
    "CH4ChemistryBuilder",
    "CO2BudgetBuilder",
    "ClimateUDEBBuilder",
    "HalocarbonChemistryBuilder",
    "N2OChemistryBuilder",
    "OceanCarbonBuilder",
    "OzoneForcingBuilder",
    "TerrestrialCarbonBuilder",
]

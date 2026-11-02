"""MAGICC climate model components.

This module provides component builders for MAGICC-derived climate processes,
organised by domain:

- Climate: ClimateUDEBBuilder (upwelling-diffusion energy balance)
- Chemistry: CH4ChemistryBuilder, CH4OxidationBuilder, N2OChemistryBuilder,
  HalocarbonChemistryBuilder
- Carbon: TerrestrialCarbonBuilder, OceanCarbonBuilder, CO2BudgetBuilder
- Forcing: OzoneForcingBuilder, AerosolDirectBuilder, AerosolIndirectBuilder
"""

from rscm._lib.magicc import (
    AerosolDirectBuilder,
    AerosolIndirectBuilder,
    CH4ChemistryBuilder,
    CH4OxidationBuilder,
    ClimateUDEBBuilder,
    CO2BudgetBuilder,
    GhgForcingBuilder,
    HalocarbonChemistryBuilder,
    N2OChemistryBuilder,
    OceanCarbonBuilder,
    OzoneForcingBuilder,
    TerrestrialCarbonBuilder,
)
from rscm.magicc.emissions import Emissions, run_magicc_emissions
from rscm.magicc.runner import CO2Emissions, MAGICCResult, run_magicc

__all__ = [
    "AerosolDirectBuilder",
    "AerosolIndirectBuilder",
    "CH4ChemistryBuilder",
    "CH4OxidationBuilder",
    "CO2BudgetBuilder",
    "CO2Emissions",
    "ClimateUDEBBuilder",
    "Emissions",
    "GhgForcingBuilder",
    "HalocarbonChemistryBuilder",
    "MAGICCResult",
    "N2OChemistryBuilder",
    "OceanCarbonBuilder",
    "OzoneForcingBuilder",
    "TerrestrialCarbonBuilder",
    "run_magicc",
    "run_magicc_emissions",
]

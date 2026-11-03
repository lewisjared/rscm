"""MAGICC climate model components.

This module provides component builders for MAGICC-derived climate processes,
organised by domain:

- Climate: ClimateUDEBBuilder (upwelling-diffusion energy balance),
  SeaLevelRiseBuilder
- Chemistry: CH4ChemistryBuilder, CH4OxidationBuilder, N2OChemistryBuilder,
  HalocarbonChemistryBuilder
- Carbon: TerrestrialCarbonBuilder, OceanCarbonBuilder, CO2BudgetBuilder,
  PermafrostBuilder
- Forcing: GhgForcingBuilder, OzoneForcingBuilder, AerosolDirectBuilder,
  AerosolIndirectBuilder, ForcingAggregationBuilder

`run_magicc7` runs the full MAGICC7 chain from emissions, see `rscm.magicc.full`.
"""

from rscm._lib.magicc import (
    AerosolDirectBuilder,
    AerosolIndirectBuilder,
    CH4ChemistryBuilder,
    CH4OxidationBuilder,
    ClimateUDEBBuilder,
    CO2BudgetBuilder,
    ForcingAggregationBuilder,
    GhgForcingBuilder,
    HalocarbonChemistryBuilder,
    N2OChemistryBuilder,
    OceanCarbonBuilder,
    OzoneForcingBuilder,
    PermafrostBuilder,
    SeaLevelRiseBuilder,
    TerrestrialCarbonBuilder,
)
from rscm.magicc.emissions import Emissions, run_magicc_emissions
from rscm.magicc.full import run_magicc7
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
    "ForcingAggregationBuilder",
    "GhgForcingBuilder",
    "HalocarbonChemistryBuilder",
    "MAGICCResult",
    "N2OChemistryBuilder",
    "OceanCarbonBuilder",
    "OzoneForcingBuilder",
    "PermafrostBuilder",
    "SeaLevelRiseBuilder",
    "TerrestrialCarbonBuilder",
    "run_magicc",
    "run_magicc7",
    "run_magicc_emissions",
]

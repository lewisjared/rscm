"""MAGICC climate model components.

This module provides component builders for MAGICC-derived climate processes,
organised by domain:

- Climate: [ClimateUDEBBuilder][] (upwelling-diffusion energy balance),
  [SeaLevelRiseBuilder][]
- Chemistry: [CH4ChemistryBuilder][], [CH4OxidationBuilder][],
  [N2OChemistryBuilder][], [HalocarbonChemistryBuilder][]
- Carbon: [TerrestrialCarbonBuilder][], [OceanCarbonBuilder][],
  [CO2BudgetBuilder][], [PermafrostBuilder][]
- Forcing: [GHGForcingBuilder][], [OzoneForcingBuilder][], [AerosolDirectBuilder][],
  [AerosolIndirectBuilder][], [ForcingAggregationBuilder][]

Two models couple them:

- [rscm.magicc.emissions_driven][] calculates concentrations from emissions,
  after an optional period of prescribed concentrations.
- [rscm.magicc.concentration_driven][] prescribes concentrations for the whole run.

Each sets up, builds and runs from a configuration and prescribed inputs,
see [rscm.magicc.model][].
`setup_model` returns a [ModelSetup][rscm.spec.ModelSetup] to inspect before building.
[from_magicc7][rscm.magicc.full.from_magicc7] translates a MAGICC7 configuration,
its input files and emissions into them,
and [with_lags][rscm.magicc.model.with_lags] adds MAGICC7's feedback lags to a builder.
"""

from rscm._lib.magicc import (
    AerosolDirectBuilder,
    AerosolIndirectBuilder,
    CH4ChemistryBuilder,
    CH4OxidationBuilder,
    ClimateUDEBBuilder,
    CO2BudgetBuilder,
    ForcingAggregationBuilder,
    GHGForcingBuilder,
    HalocarbonChemistryBuilder,
    N2OChemistryBuilder,
    OceanCarbonBuilder,
    OzoneForcingBuilder,
    PermafrostBuilder,
    SeaLevelRiseBuilder,
    TerrestrialCarbonBuilder,
)
from rscm.magicc import concentration_driven, emissions_driven
from rscm.magicc.full import from_magicc7
from rscm.magicc.model import with_lags

__all__ = [
    "AerosolDirectBuilder",
    "AerosolIndirectBuilder",
    "CH4ChemistryBuilder",
    "CH4OxidationBuilder",
    "CO2BudgetBuilder",
    "ClimateUDEBBuilder",
    "ForcingAggregationBuilder",
    "GHGForcingBuilder",
    "HalocarbonChemistryBuilder",
    "N2OChemistryBuilder",
    "OceanCarbonBuilder",
    "OzoneForcingBuilder",
    "PermafrostBuilder",
    "SeaLevelRiseBuilder",
    "TerrestrialCarbonBuilder",
    "concentration_driven",
    "emissions_driven",
    "from_magicc7",
    "with_lags",
]

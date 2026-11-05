"""
FaIR v2 components.

A port of FaIR v2.2.5 (Leach et al. 2021, Smith et al. 2024) that reproduces its default
methods: Meinshausen et al. (2020) greenhouse gas forcing and either the generic or the
Thornhill et al. (2021) CH4 lifetime.

[build_components][rscm.fair.config.build_components] turns a FaIR configuration into
components and [run_fair][rscm.fair.run.run_fair] runs one.
[rscm.fair.loaders][] reads configurations from FaIR's species CSV
and fair-calibrate's posterior CSV.
[run_ensemble][rscm.fair.ensemble.run_ensemble] runs many and collects the results
in FaIR's layout.
[from_fair][rscm.fair.convert.from_fair] converts a configured `fair.FAIR`.
Variables use RSCM names (see [rscm.fair.naming][]) and FaIR's units.
"""

from rscm._lib.fair import (
    FairAerosolCloudBuilder,
    FairCH4LifetimeBuilder,
    FairEescBuilder,
    FairEnergyBalanceBuilder,
    FairForcingAggregationBuilder,
    FairGasCycleBuilder,
    FairGhgForcingBuilder,
    FairLinearForcingBuilder,
)
from rscm.fair.config import FairConfig, build_components, initial_values, load_config
from rscm.fair.convert import from_fair
from rscm.fair.ensemble import run_ensemble
from rscm.fair.loaders import make_config, read_calibration, read_config, read_species
from rscm.fair.run import build_model, prescribed_forcing_scales, run_fair

__all__ = [
    "FairAerosolCloudBuilder",
    "FairCH4LifetimeBuilder",
    "FairConfig",
    "FairEescBuilder",
    "FairEnergyBalanceBuilder",
    "FairForcingAggregationBuilder",
    "FairGasCycleBuilder",
    "FairGhgForcingBuilder",
    "FairLinearForcingBuilder",
    "build_components",
    "build_model",
    "from_fair",
    "initial_values",
    "load_config",
    "make_config",
    "prescribed_forcing_scales",
    "read_calibration",
    "read_config",
    "read_species",
    "run_ensemble",
    "run_fair",
]

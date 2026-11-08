"""
FaIR v2 components.

A port of FaIR v2.2.5 (Leach et al. 2021, Smith et al. 2024) that reproduces its default
methods: Meinshausen et al. (2020) greenhouse gas forcing and either the generic or the
Thornhill et al. (2021) CH4 lifetime.

[setup_model][rscm.fair.run.setup_model] gives the [setup][rscm.spec.ModelSetup]
of a FaIR configuration: its components, lags, initial values and prescribed data,
ready to inspect, build and run.
[component_specs][rscm.fair.config.component_specs] and [lags][rscm.fair.config.lags]
give the parts on their own.
[rscm.fair.loaders][] reads configurations from FaIR's species CSV
and fair-calibrate's posterior CSV.
[run_ensemble][rscm.fair.ensemble.run_ensemble] runs many and collects the results
in FaIR's layout.
[from_fair][rscm.fair.convert.from_fair] converts a configured `fair.FAIR`.
Variables use RSCM names (see [rscm.fair.naming][]) and FaIR's units.
"""

from rscm._lib.fair import (
    AerosolCloudBuilder,
    CH4LifetimeBuilder,
    EESCBuilder,
    EnergyBalanceBuilder,
    ForcingAggregationBuilder,
    GasCycleBuilder,
    GHGForcingBuilder,
    LinearForcingBuilder,
)
from rscm.fair.config import (
    FairConfig,
    component_specs,
    components,
    initial_values,
    lags,
    load_config,
    with_lags,
)
from rscm.fair.convert import from_fair
from rscm.fair.ensemble import run_ensemble
from rscm.fair.loaders import make_config, read_calibration, read_config, read_species
from rscm.fair.run import prescribed_forcing_scales, setup_model

__all__ = [
    "AerosolCloudBuilder",
    "CH4LifetimeBuilder",
    "EESCBuilder",
    "EnergyBalanceBuilder",
    "FairConfig",
    "ForcingAggregationBuilder",
    "GHGForcingBuilder",
    "GasCycleBuilder",
    "LinearForcingBuilder",
    "component_specs",
    "components",
    "from_fair",
    "initial_values",
    "lags",
    "load_config",
    "make_config",
    "prescribed_forcing_scales",
    "read_calibration",
    "read_config",
    "read_species",
    "run_ensemble",
    "setup_model",
    "with_lags",
]

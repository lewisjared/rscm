"""
The concentration-driven MAGICC model.

The concentrations of CO2, CH4, N2O and every halocarbon are prescribed
for the whole run.
They drive the same forcing and climate as in the emissions-driven model,
and the carbon cycle, so land and ocean fluxes and pools follow the prescribed CO2.

The model has the components of [rscm.magicc.emissions_driven][].
`setup_model` sets every concentration switch year to infinity,
whatever the configuration holds.
So a configuration of the emissions-driven model also builds this model,
given inputs that prescribe every concentration over the whole run.

`CH4Chemistry` writes the anthropogenic CH4 emissions that reproduce
the prescribed concentration as ``Inverse Emissions|CH4``,
net of any permafrost emissions.
The model calculates no other inverse emissions, so ``Net Emissions|CO2`` of `CO2Budget`
is the net source the prescribed CO2 implies rather than fossil emissions.
The components still read their emissions inputs:
fossil CO2 spreads short-lived F-gases between the hemispheres,
and NOx, CO and NMVOC drive tropospheric ozone.

See [rscm.magicc.model][] for the configuration and inputs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from rscm.magicc import model
from rscm.magicc.emissions_driven import COMPONENTS, OPTIONAL_COMPONENTS

if TYPE_CHECKING:
    from rscm.core import TimeAxis
    from rscm.magicc.model import Inputs, ModelConfig
    from rscm.spec import ModelSetup

__all__ = [
    "COMPONENTS",
    "OPTIONAL_COMPONENTS",
    "setup_model",
]

_PRESCRIBED: dict[str, dict[str, Any]] = {
    "CH4Chemistry": {"concentration_switch_year": math.inf},
    "N2OChemistry": {"concentration_switch_year": math.inf},
    "HalocarbonChemistry": {
        "fgas_switch_year": math.inf,
        "montreal_gases_switch_year": math.inf,
    },
    "OceanCarbon": {"concentration_switch_year": math.inf},
    "CO2Budget": {"concentration_switch_year": math.inf},
}


def setup_model(config: ModelConfig, inputs: Inputs, time_axis: TimeAxis) -> ModelSetup:
    """Return the setup of a concentration-driven model, ready to inspect and build.

    ``time_axis`` has annual steps.
    ``inputs`` holds ``Prescribed Atmospheric Concentrations|<gas>`` of every gas
    at every time bound, the emissions in the units the components declare
    and the prescribed forcing of each ForcingAggregation agent.
    The setup holds the switch years set to infinity.
    """
    components = {
        name: {**parameters, **_PRESCRIBED.get(name, {})}
        for name, parameters in config["components"].items()
    }
    return model.setup(
        {**config, "components": components},
        inputs,
        time_axis,
        required=COMPONENTS,
        optional=OPTIONAL_COMPONENTS,
        name="MAGICC concentration-driven",
    )

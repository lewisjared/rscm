"""
The emissions-driven MAGICC model.

Chemistry and the carbon cycle calculate the concentrations of CO2, CH4, N2O
and every halocarbon from emissions, with temperature feeding back on them.
GHG and ozone forcing, the prescribed forcing and the monthly volcanic forcing
are aggregated with efficacies and regional patterns to drive ClimateUDEB.

| Component             | Role                                                    |
| --------------------- | ------------------------------------------------------- |
| `Permafrost`          | Permafrost CO2 and CH4 emissions (optional)             |
| `CH4Chemistry`        | CH4 concentration and lifetime                          |
| `CH4Oxidation`        | CH4 by source and the CO2 from fossil CH4 oxidation     |
| `N2OChemistry`        | N2O concentration and lifetime                          |
| `HalocarbonChemistry` | Halocarbon concentrations, forcing and EESC             |
| `TerrestrialCarbon`   | Land carbon pools and the net land flux                 |
| `OceanCarbon`         | Ocean carbon uptake                                     |
| `CO2Budget`           | CO2 concentration                                       |
| `GHGForcing`          | CO2, CH4 and N2O forcing and stratospheric H2O from CH4 |
| `OzoneForcing`        | Stratospheric and tropospheric ozone forcing            |
| `ForcingAggregation`  | Total and efficacy-adjusted forcing                     |
| `ClimateUDEB`         | Temperature, ocean heat uptake and thermal expansion    |
| `SeaLevelRise`        | Sea level rise (optional)                               |

As in MAGICC7, each gas can follow prescribed concentrations up to the switch year
in its components' parameters, such as ``concentration_switch_year`` of `CO2Budget`,
and is calculated from emissions afterwards.
The inputs then hold ``Prescribed Atmospheric Concentrations|<gas>`` until that year.
A model with `Permafrost` sets ``include_permafrost`` on `CH4Chemistry`,
`CH4Oxidation`, `OceanCarbon` and `CO2Budget`.

See [rscm.magicc.model][] for the configuration and inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rscm.magicc import model

if TYPE_CHECKING:
    from rscm.core import TimeAxis
    from rscm.magicc.model import Inputs, ModelConfig
    from rscm.spec import ModelSetup

__all__ = [
    "COMPONENTS",
    "OPTIONAL_COMPONENTS",
    "setup_model",
]

COMPONENTS = (
    "CH4Chemistry",
    "CH4Oxidation",
    "N2OChemistry",
    "HalocarbonChemistry",
    "TerrestrialCarbon",
    "OceanCarbon",
    "CO2Budget",
    "GHGForcing",
    "OzoneForcing",
    "ForcingAggregation",
    "ClimateUDEB",
)
"""Components the configuration must give parameters for."""

OPTIONAL_COMPONENTS = ("Permafrost", "SeaLevelRise")
"""Components the model includes when the configuration gives their parameters."""


def setup_model(config: ModelConfig, inputs: Inputs, time_axis: TimeAxis) -> ModelSetup:
    """Return the setup of an emissions-driven model, ready to inspect and build.

    ``time_axis`` has annual steps.
    ``inputs`` holds the emissions in the units the components declare,
    the prescribed forcing of each ForcingAggregation agent
    and the concentrations prescribed before each switch year.
    """
    return model.setup(
        config,
        inputs,
        time_axis,
        required=COMPONENTS,
        optional=OPTIONAL_COMPONENTS,
        name="MAGICC emissions-driven",
    )

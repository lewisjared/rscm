"""FaIR v2 components.

Builders for the components of FaIR v2.2.5, in the order a model needs them.
`rscm.fair.build_components` builds them from a FaIR configuration.
"""

from collections.abc import Mapping
from typing import Any, final

from rscm._lib.core import Component, ComponentBuilder

@final
class FairCH4LifetimeBuilder(ComponentBuilder):
    """CH4 lifetime scaling from precursors (Thornhill et al. 2021).

    **Parameter keys**

    - `emissions` and `concentrations`: precursors with `variable`, `unit`,
      `baseline` and `sensitivity`
    - `temperature_sensitivity`, `ch4_min`, `surface_temperature`, `output`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> FairCH4LifetimeBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the component, checking the parameters."""

@final
class FairGasCycleBuilder(ComponentBuilder):
    """Greenhouse gas cycle with lifetime boxes, forward or inverse.

    **Parameter keys**

    - `species`: one dict per gas, see `rscm.fair.config`
    - `iirf_max`, `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> FairGasCycleBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the component, checking the parameters."""

@final
class FairEescBuilder(ComponentBuilder):
    """Equivalent effective stratospheric chlorine.

    **Parameter keys**

    - `halogens`: `concentration`, `unit`, `cl_atoms`, `br_atoms`,
      `fractional_release`
    - `cfc11_fractional_release`, `br_cl_ratio`, `output`, `unit`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> FairEescBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the component, checking the parameters."""

@final
class FairGhgForcingBuilder(ComponentBuilder):
    """Meinshausen et al. (2020) greenhouse gas forcing.

    **Parameter keys**

    - `co2`, `ch4`, `n2o` and the `minor` gases: `concentration`,
      `concentration_unit`, `forcing`, `reference_concentration`,
      `baseline_concentration`, `forcing_scale`, `tropospheric_adjustment`,
      `radiative_efficiency`, `temperature_feedback`
    - `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> FairGhgForcingBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the component, checking the parameters."""

@final
class FairLinearForcingBuilder(ComponentBuilder):
    """Forcing linear in emissions, concentrations or their sum.

    **Parameter keys**

    - `forcing`
    - `drivers`: `variable`, `unit`, `kind`, `baseline`, `radiative_efficiency`
    - `forcing_scale`, `tropospheric_adjustment`, `temperature_feedback`,
      `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> FairLinearForcingBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the component, checking the parameters."""

@final
class FairAerosolCloudBuilder(ComponentBuilder):
    """Logarithmic aerosol-cloud interaction forcing.

    **Parameter keys**

    - `forcing`, `scale`
    - `drivers`: `variable`, `unit`, `kind`, `baseline`, `shape`
    - `forcing_scale`, `tropospheric_adjustment`, `temperature_feedback`,
      `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> FairAerosolCloudBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the component, checking the parameters."""

@final
class FairForcingAggregationBuilder(ComponentBuilder):
    """Total and efficacy-adjusted forcing.

    **Parameter keys**

    - `agents`: `forcing`, `prescribed`, `efficacy`, `temperature_feedback`
    - `total`, `efficacy_adjusted`, `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> FairForcingAggregationBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the component, checking the parameters."""

@final
class FairEnergyBalanceBuilder(ComponentBuilder):
    """n-layer energy balance model (Cummins et al. 2020).

    **Parameter keys**

    - `ocean_heat_capacity`, `ocean_heat_transfer`, `deep_ocean_efficacy`
    - `gamma_autocorrelation`, `surface_temperature`, `forcing`,
      `efficacy_adjusted_forcing`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> FairEnergyBalanceBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> Component:
        """Build the component, checking the parameters."""

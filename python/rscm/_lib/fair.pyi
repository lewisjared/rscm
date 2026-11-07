"""FaIR v2 components.

Builders for the components of FaIR v2.2.5, in the order a model needs them.
`rscm.fair.components` builds them from a FaIR configuration.
"""

from collections.abc import Mapping
from typing import Any, final

from rscm._lib.core import ComponentBuilder, RustComponent

@final
class CH4LifetimeBuilder(ComponentBuilder):
    """CH4 lifetime scaling from precursors (Thornhill et al. 2021).

    **Parameter keys**

    - `emissions` and `concentrations`: precursors with `variable`, `unit`,
      `baseline` and `sensitivity`
    - `temperature_sensitivity`, `minimum_scaling`, `surface_temperature`, `output`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> CH4LifetimeBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the component, checking the parameters."""

@final
class GasCycleBuilder(ComponentBuilder):
    """Greenhouse gas cycle with lifetime boxes, forward or inverse.

    **Parameter keys**

    - `species`: one dict per gas, see `rscm.fair.config`
    - `maximum_iirf`, `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> GasCycleBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the component, checking the parameters."""

@final
class EescBuilder(ComponentBuilder):
    """Equivalent effective stratospheric chlorine.

    **Parameter keys**

    - `halogens`: `concentration`, `unit`, `cl_atoms`, `br_atoms`,
      `fractional_release`
    - `cfc11_fractional_release`, `br_cl_ratio`, `output`, `unit`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> EescBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the component, checking the parameters."""

@final
class GhgForcingBuilder(ComponentBuilder):
    """Meinshausen et al. (2020) greenhouse gas forcing.

    **Parameter keys**

    - `co2`, `ch4`, `n2o` and the `minor` gases: `concentration`,
      `concentration_unit`, `forcing`, `reference_concentration`,
      `baseline_concentration`, `forcing_scale`, `tropospheric_adjustment`,
      `radiative_efficiency`, `temperature_feedback`
    - `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> GhgForcingBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the component, checking the parameters."""

@final
class LinearForcingBuilder(ComponentBuilder):
    """Forcing linear in emissions, concentrations or their sum.

    **Parameter keys**

    - `forcing`
    - `drivers`: `variable`, `unit`, `driver`, `baseline`, `radiative_efficiency`
    - `forcing_scale`, `tropospheric_adjustment`, `temperature_feedback`,
      `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> LinearForcingBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the component, checking the parameters."""

@final
class AerosolCloudBuilder(ComponentBuilder):
    """Logarithmic aerosol-cloud interaction forcing.

    **Parameter keys**

    - `forcing`, `scale`
    - `drivers`: `variable`, `unit`, `driver`, `baseline`, `shape`
    - `forcing_scale`, `tropospheric_adjustment`, `temperature_feedback`,
      `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> AerosolCloudBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the component, checking the parameters."""

@final
class ForcingAggregationBuilder(ComponentBuilder):
    """Total and efficacy-adjusted forcing.

    **Parameter keys**

    - `agents`: `forcing`, `category`, `prescribed`, `efficacy`, `temperature_feedback`
    - `total`, `efficacy_adjusted`, `surface_temperature`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> ForcingAggregationBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the component, checking the parameters."""

@final
class EnergyBalanceBuilder(ComponentBuilder):
    """n-layer energy balance model (Cummins et al. 2020).

    **Parameter keys**

    - `ocean_heat_capacity`, `ocean_heat_transfer`, `deep_ocean_efficacy`
    - `gamma_autocorrelation`, `surface_temperature`, `forcing`,
      `efficacy_adjusted_forcing`
    """

    @staticmethod
    def from_parameters(parameters: Mapping[str, Any]) -> EnergyBalanceBuilder:  # type: ignore[override]
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the component, checking the parameters."""

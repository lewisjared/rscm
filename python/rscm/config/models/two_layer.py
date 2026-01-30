"""
Two-layer energy balance model parameters.

This module provides TwoLayerParameters dataclass with full metadata
for the two-layer energy balance model described in Held et al. (2010).

Example
-------
    >>> from rscm.config.models.two_layer import TwoLayerParameters
    >>> params = TwoLayerParameters(lambda0=1.0, a=0.0)
    >>> params.lambda0
    1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rscm.config.base import ModelConfig
from rscm.config.parameters import parameter, validate_parameters
from rscm.config.registry import component_registry
from rscm.two_layer import TwoLayerBuilder

__all__ = ["TwoLayerConfig", "TwoLayerParameters"]


@dataclass
class TwoLayerParameters:
    """Two-layer energy balance model parameters.

    Based on the two-layer model described in:
    Held, I. M., Winton, M., Takahashi, K., Delworth, T., Zeng, F.,
    & Vallis, G. K. (2010).
    Probing the fast and slow components of global warming by returning
    abruptly to preindustrial forcing. Journal of Climate, 23(9), 2418-2427.

    Attributes
    ----------
    lambda0 : float
        Climate feedback parameter at zero warming (W/(m² K))
    a : float
        Nonlinear feedback coefficient (W/(m² K²)). Set to 0 for linear model.
    efficacy : float
        Ocean heat uptake efficacy (dimensionless)
    eta : float
        Heat exchange coefficient between surface and deep layers (W/(m² K))
    heat_capacity_surface : float
        Heat capacity of surface layer (W yr/(m² K))
    heat_capacity_deep : float
        Heat capacity of deep ocean layer (W yr/(m² K))
    """

    lambda0: float = parameter(
        default=1.0,
        unit="W/(m² K)",
        description="Climate feedback parameter at zero warming",
        range=(0.1, 5.0),
        typical_range=(0.8, 1.5),
        source="Held et al. (2010)",
    )

    a: float = parameter(
        default=0.0,
        unit="W/(m² K²)",
        description="Nonlinear feedback coefficient. Set to 0 for linear model.",
        range=(0.0, 1.0),
        typical_range=(0.0, 0.1),
    )

    efficacy: float = parameter(
        default=1.0,
        unit="dimensionless",
        description="Ocean heat uptake efficacy",
        range=(0.5, 3.0),
        typical_range=(1.0, 1.8),
    )

    eta: float = parameter(
        default=0.7,
        unit="W/(m² K)",
        description="Heat exchange coefficient between surface and deep layers",
        range=(0.1, 2.0),
        typical_range=(0.5, 1.0),
    )

    heat_capacity_surface: float = parameter(
        default=8.0,
        unit="W yr/(m² K)",
        description="Heat capacity of surface layer (mixed layer + atmosphere)",
        range=(1.0, 50.0),
        typical_range=(5.0, 15.0),
    )

    heat_capacity_deep: float = parameter(
        default=100.0,
        unit="W yr/(m² K)",
        description="Heat capacity of deep ocean layer",
        range=(10.0, 500.0),
        typical_range=(50.0, 200.0),
    )

    def __post_init__(self) -> None:
        """Validate parameters after initialisation."""
        errors = validate_parameters(self)
        if errors:
            msg = f"Invalid parameters: {errors}"
            raise ValueError(msg)


@dataclass
class TwoLayerConfig(ModelConfig):
    """Complete configuration for two-layer energy balance model.

    This class combines the base model configuration (inherited from ModelConfig)
    with two-layer-specific climate parameters.

    Attributes
    ----------
    model_type : str
        Model type identifier (defaults to "two-layer")
    climate : TwoLayerParameters
        Climate physics parameters for the two-layer model
    name : str
        Model name (inherited from ModelConfig)
    version : str
        Model version (inherited from ModelConfig)
    config_schema : str
        Configuration schema version (inherited from ModelConfig)
    description : str
        Model description (inherited from ModelConfig)
    time : TimeConfig | None
        Time axis configuration (inherited from ModelConfig)
    inputs : dict[str, InputSpec]
        Input data specifications (inherited from ModelConfig)
    initial_values : dict[str, float]
        Initial values for state variables (inherited from ModelConfig)

    Example
    -------
        >>> from rscm.config.models.two_layer import TwoLayerConfig
        >>> from rscm.config.base import TimeConfig
        >>> config = TwoLayerConfig(
        ...     name="test-run",
        ...     time=TimeConfig(start=1750, end=2100),
        ... )
        >>> config.model_type
        'two-layer'
    """

    model_type: str = "two-layer"
    climate: TwoLayerParameters = field(default_factory=TwoLayerParameters)


# Register TwoLayerBuilder with the component registry
component_registry.register("TwoLayer", TwoLayerBuilder)

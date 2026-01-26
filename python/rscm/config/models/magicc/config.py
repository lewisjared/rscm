"""MAGICC model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from rscm.config.base import ModelConfig

__all__ = ["AggregationConfig", "ClimateConfig", "ForcingConfig", "MAGICCConfig"]


@dataclass
class ClimateConfig:
    """MAGICC climate model parameters."""

    climate_sensitivity: float = 3.0
    """Equilibrium climate sensitivity (K)."""

    forcing_2xco2: float = 3.71
    """Radiative forcing for doubling CO2 (W/m²)."""


@dataclass
class ForcingConfig:
    """MAGICC forcing parameters."""

    solar_scale: float = 1.0
    """Solar forcing scale factor."""

    volcanic_scale: float = 1.0
    """Volcanic forcing scale factor."""


@dataclass
class AggregationConfig:
    """MAGICC forcing aggregation settings."""

    run_modus: str = "ALL"
    """Run mode (CO2, GHG, AEROSOL, ALL, etc.)."""


@dataclass
class MAGICCConfig(ModelConfig):
    """Configuration for MAGICC model.

    This is a skeleton configuration that will be expanded as MAGICC
    components are implemented in RSCM.

    Parameters
    ----------
    name
        Model name
    climate
        Climate model parameters
    forcing
        Forcing parameters
    aggregation
        Aggregation settings

    Attributes
    ----------
    model_type
        Always "magicc"
    """

    model_type: str = "magicc"
    climate: ClimateConfig = field(default_factory=ClimateConfig)
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)

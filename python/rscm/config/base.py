"""
Base classes for RSCM configuration.

This module defines the foundational dataclasses used by all model configurations:
- TimeConfig: Time axis specification
- InputSpec: Input data file specification
- ModelConfig: Base configuration for all model types
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["InputSpec", "ModelConfig", "TimeConfig"]


@dataclass
class TimeConfig:
    """
    Time axis configuration.

    Parameters
    ----------
    start
        Start year (inclusive)
    end
        End year (inclusive)

    Raises
    ------
    ValueError
        If end <= start
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate that end > start."""
        if self.end <= self.start:
            msg = f"end ({self.end}) must be greater than start ({self.start})"
            raise ValueError(msg)

    def to_time_axis(self) -> tuple[int, int]:
        """
        Return time axis as tuple.

        Returns
        -------
        tuple[int, int]
            (start, end) tuple
        """
        return (self.start, self.end)


@dataclass
class InputSpec:
    """
    Input data specification.

    Parameters
    ----------
    file
        Path to input data file (optional)
    unit
        Physical unit of the input (optional)
    required
        Whether this input is required for model execution
    """

    file: str | None = None
    unit: str | None = None
    required: bool = False

    def is_complete(self) -> bool:
        """
        Check if specification is complete.

        Returns
        -------
        bool
            True if file and unit are both specified
        """
        return self.file is not None and self.unit is not None


@dataclass
class ModelConfig:
    """
    Base model configuration.

    Parameters
    ----------
    name
        Model name
    model_type
        Type of model (e.g., "two-layer", "magicc")
    version
        Model version
    config_schema
        Configuration schema version
    description
        Model description
    time
        Time axis configuration
    inputs
        Input data specifications by variable name
    initial_values
        Initial values for state variables
    """

    name: str
    model_type: str = ""
    version: str = "1.0.0"
    config_schema: str = "1.0.0"
    description: str = ""
    time: TimeConfig | None = None
    inputs: dict[str, InputSpec] = field(default_factory=dict)
    initial_values: dict[str, float] = field(default_factory=dict)

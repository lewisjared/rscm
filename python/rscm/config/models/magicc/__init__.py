"""
MAGICC model configuration and legacy format support.

This package provides:
- MAGICCConfig: Configuration class for MAGICC models
- Legacy .CFG format import/export
- Parameter registry with support status tracking
"""

from __future__ import annotations

from rscm.config.models.magicc.config import (
    AggregationConfig,
    ClimateConfig,
    ForcingConfig,
    MAGICCConfig,
)
from rscm.config.models.magicc.legacy import (
    LEGACY_MAPPING,
    from_legacy_dict,
    to_legacy_dict,
)
from rscm.config.models.magicc.parameters import (
    MAGICC_PARAMETERS,
    ParameterInfo,
    ParameterStatus,
    get_coverage_report,
    get_coverage_stats,
)

__all__ = [
    "LEGACY_MAPPING",
    "MAGICC_PARAMETERS",
    "AggregationConfig",
    "ClimateConfig",
    "ForcingConfig",
    "MAGICCConfig",
    "ParameterInfo",
    "ParameterStatus",
    "from_legacy_dict",
    "get_coverage_report",
    "get_coverage_stats",
    "to_legacy_dict",
]

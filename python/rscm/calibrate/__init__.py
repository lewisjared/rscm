"""
Calibration and parameter estimation for RSCM models.

This package provides tools for:

- Defining parameter priors and sampling parameter spaces
- Specifying observation targets and computing likelihoods
- MCMC sampling via affine-invariant ensemble sampling
- Point estimation via optimisation algorithms
- Convergence diagnostics and chain analysis
"""

from rscm._lib.calibrate import (
    Bound,
    Chain,
    # MCMC ensemble sampling
    EnsembleSampler,
    # Likelihood computation
    GaussianLikelihood,
    LogNormal,
    # Model runner interface
    ModelRunner,
    Normal,
    # Observations and targets
    Observation,
    OptimisationResult,
    Optimiser,
    # Parameter set specification
    ParameterSet,
    # Point estimation
    PointEstimator,
    ProgressInfo,
    Target,
    # Distributions for parameter priors
    Uniform,
    VariableTarget,
    WalkerInit,
)

from . import progress
from .pandas_helpers import chain_to_dataframe, target_from_dataframe


def _chain_to_dataframe(self, discard=0):
    """
    Convert chain to pandas DataFrame.

    Parameters
    ----------
    discard : int, optional
        Number of initial samples to discard as burn-in (default: 0)

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns for each parameter and log_prob, with a
        multi-index (walker, iteration) for rows.

    Examples
    --------
    >>> df = chain.to_dataframe(discard=100)
    >>> print(df.head())
    >>> print(df.describe())
    """
    return chain_to_dataframe(self, discard=discard)


Chain.to_dataframe = _chain_to_dataframe  # type: ignore[attr-defined]
Target.from_dataframe = staticmethod(target_from_dataframe)  # type: ignore[attr-defined]

__all__ = [
    "Bound",
    "Chain",
    # MCMC
    "EnsembleSampler",
    # Likelihood
    "GaussianLikelihood",
    "LogNormal",
    # Model runner
    "ModelRunner",
    "Normal",
    # Observations
    "Observation",
    "OptimisationResult",
    "Optimiser",
    # Parameters
    "ParameterSet",
    # Point estimation
    "PointEstimator",
    "ProgressInfo",
    "Target",
    # Distributions
    "Uniform",
    "VariableTarget",
    "WalkerInit",
    "chain_to_dataframe",
    # Progress utilities
    "progress",
    "target_from_dataframe",
]

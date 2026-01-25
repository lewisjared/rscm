"""
Calibration and parameter estimation for RSCM models.

This package provides tools for:
- Defining parameter priors and sampling parameter spaces
- Specifying observation targets and computing likelihoods
- MCMC sampling via affine-invariant ensemble sampling
- Point estimation via optimization algorithms
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
    OptimizationResult,
    Optimizer,
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

# Import pandas helpers (will fail gracefully if pandas not installed)
try:
    from .pandas_helpers import chain_to_dataframe, target_from_dataframe

    # Monkey-patch Chain.to_dataframe method
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

    Chain.to_dataframe = _chain_to_dataframe

    # Add Target.from_dataframe static method
    Target.from_dataframe = staticmethod(target_from_dataframe)

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    chain_to_dataframe = None
    target_from_dataframe = None

# Import progress utilities (in separate submodule for clarity)
from . import progress

__all__ = [
    # Helpers
    "HAS_PANDAS",
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
    "OptimizationResult",
    "Optimizer",
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

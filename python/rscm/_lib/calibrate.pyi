"""
Type stubs for rscm._lib.calibrate module.

This module provides calibration and parameter estimation functionality.
"""

from collections.abc import Callable, Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray

# Type aliases
Arr = NDArray[np.float64]
F = np.float64 | float

# ==================== Distribution Types ====================

class Uniform:
    """
    Uniform distribution over a bounded interval.

    Parameters
    ----------
    low : float
        Lower bound (inclusive)
    high : float
        Upper bound (inclusive)

    Raises
    ------
    ValueError
        If low >= high
    """

    def __init__(self, low: F, high: F) -> None: ...
    def sample(self) -> F:
        """Sample a random value from the distribution."""
    def ln_pdf(self, x: F) -> F:
        """Compute the log probability density at x."""
    def bounds(self) -> tuple[F, F]:
        """Return the (low, high) bounds of the distribution."""
    @property
    def low(self) -> F:
        """Lower bound of the distribution."""
    @property
    def high(self) -> F:
        """Upper bound of the distribution."""

class Normal:
    """
    Normal (Gaussian) distribution.

    Parameters
    ----------
    mean : float
        Mean of the distribution
    std_dev : float
        Standard deviation (must be positive)

    Raises
    ------
    ValueError
        If std_dev <= 0
    """

    def __init__(self, mean: F, std_dev: F) -> None: ...
    def sample(self) -> F:
        """Sample a random value from the distribution."""
    def ln_pdf(self, x: F) -> F:
        """Compute the log probability density at x."""
    def bounds(self) -> None:
        """Return None (unbounded distribution)."""
    @property
    def mean(self) -> F:
        """Mean of the distribution."""
    @property
    def std_dev(self) -> F:
        """Standard deviation of the distribution."""

class LogNormal:
    """
    Log-normal distribution.

    Can be constructed using either (mu, sigma) parameters of the underlying
    normal distribution, or (mean, std) of the log-normal distribution itself.

    Parameters
    ----------
    mu : float, optional
        Mean of the underlying normal distribution (log-space)
    sigma : float, optional
        Standard deviation of underlying normal distribution (log-space, positive)
    mean : float, optional
        Mean of the log-normal distribution (real-space, must be positive)
    std : float, optional
        Standard deviation of the log-normal distribution (real-space, must be positive)

    Notes
    -----
    Must provide exactly one of:
    - (mu, sigma): log-space parameters
    - (mean, std): real-space parameters

    Raises
    ------
    ValueError
        If invalid parameter combination provided or if parameters are out of range
    """

    def __init__(
        self,
        *,
        mu: F | None = None,
        sigma: F | None = None,
        mean: F | None = None,
        std: F | None = None,
    ) -> None: ...
    def sample(self) -> F:
        """Sample a random value from the distribution."""
    def ln_pdf(self, x: F) -> F:
        """Compute the log probability density at x."""
    def bounds(self) -> None:
        """Return None (unbounded distribution, though x > 0)."""
    @property
    def mu(self) -> F:
        """Mean of the underlying normal distribution (log-space)."""
    @property
    def sigma(self) -> F:
        """Standard deviation of the underlying normal distribution (log-space)."""

class Bound:
    """
    Bounded wrapper for any distribution.

    Constrains samples from an underlying distribution to lie within [low, high]
    using rejection sampling.

    Parameters
    ----------
    distribution : Uniform | Normal | LogNormal | Bound
        The distribution to bound
    low : float
        Lower bound (inclusive)
    high : float
        Upper bound (inclusive)

    Raises
    ------
    ValueError
        If low >= high
    """

    def __init__(
        self, distribution: Uniform | Normal | LogNormal | Bound, low: F, high: F
    ) -> None: ...
    def sample(self) -> F:
        """Sample a random value from the bounded distribution."""
    def ln_pdf(self, x: F) -> F:
        """Compute the log probability density at x (unnormalized)."""
    def bounds(self) -> tuple[F, F]:
        """Return the (low, high) bounds."""
    @property
    def low(self) -> F:
        """Lower bound."""
    @property
    def high(self) -> F:
        """Upper bound."""

# ==================== Parameter Set ====================

class ParameterSet:
    """
    Collection of named parameters with prior distributions.

    Can be constructed from a dictionary or built incrementally using the
    fluent add() method.

    Parameters
    ----------
    params : dict[str, Uniform | Normal | LogNormal | Bound], optional
        Dictionary mapping parameter names to distributions

    Examples
    --------
    >>> # Dict constructor
    >>> params = ParameterSet({"x": Uniform(0, 1), "y": Normal(0, 1)})
    >>>
    >>> # Fluent builder
    >>> params = ParameterSet().add("x", Uniform(0, 1)).add("y", Normal(0, 1))
    """

    def __init__(
        self, params: dict[str, Uniform | Normal | LogNormal | Bound] | None = None
    ) -> None: ...
    def add(
        self, name: str, distribution: Uniform | Normal | LogNormal | Bound
    ) -> Self:
        """
        Add a parameter to the set.

        Parameters
        ----------
        name : str
            Parameter name
        distribution : Distribution
            Prior distribution for this parameter

        Returns
        -------
        Self
            Returns self for method chaining
        """
    def sample_random(self, n: int) -> Arr:
        """
        Sample n parameter vectors randomly from the priors.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        NDArray[np.float64]
            Array of shape (n, n_params) with random samples
        """
    def sample_lhs(self, n: int) -> Arr:
        """
        Sample n parameter vectors using Latin Hypercube Sampling.

        LHS provides better parameter space coverage than random sampling.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        NDArray[np.float64]
            Array of shape (n, n_params) with LHS samples
        """
    def log_prior(self, params: Sequence[F]) -> F:
        """
        Compute log prior probability for a parameter vector.

        Parameters
        ----------
        params : array-like
            Parameter values in the order given by param_names

        Returns
        -------
        float
            Log prior probability

        Raises
        ------
        ValueError
            If params length doesn't match number of parameters
        """
    def bounds(self) -> tuple[Arr, Arr]:
        """
        Get parameter bounds for optimization.

        Returns
        -------
        lower : NDArray[np.float64]
            Lower bounds for each parameter (-inf for unbounded)
        upper : NDArray[np.float64]
            Upper bounds for each parameter (+inf for unbounded)
        """
    @property
    def param_names(self) -> list[str]:
        """Get parameter names in definition order."""

# ==================== Observations and Targets ====================

class Observation:
    """
    Single observation at a specific time.

    Parameters
    ----------
    time : float
        Time of observation
    value : float
        Observed value
    uncertainty : float
        Observation uncertainty (standard deviation)

    Raises
    ------
    ValueError
        If uncertainty <= 0
    """

    def __init__(self, time: F, value: F, uncertainty: F) -> None: ...
    @property
    def time(self) -> F:
        """Time of observation."""
    @property
    def value(self) -> F:
        """Observed value."""
    @property
    def uncertainty(self) -> F:
        """Observation uncertainty (standard deviation)."""

class VariableTarget:
    """
    Collection of observations for a single variable.

    Properties
    ----------
    name : str
        Variable name
    observations : list[Observation]
        List of observations for this variable
    reference_period : tuple[float, float] | None
        Reference period for anomaly calculation, if set
    """

    @property
    def name(self) -> str:
        """Variable name."""
    @property
    def observations(self) -> list[Observation]:
        """List of observations."""
    @property
    def reference_period(self) -> tuple[F, F] | None:
        """Reference period (start, end) for anomaly calculation, or None."""

class Target:
    """
    Collection of observation targets for multiple variables.

    Examples
    --------
    >>> target = Target()
    >>> target.add_observation("Temperature|Global", 2020, 1.2, 0.1)
    >>> target.add_observation("Temperature|Global", 2021, 1.3, 0.1)
    >>> target.set_reference_period("Temperature|Global", 1850, 1900)
    >>>
    >>> # Using relative uncertainty
    >>> target.add_observation_relative("OHC|Total", 2020, 200, 0.1)  # 10% error
    """

    def __init__(self) -> None: ...
    def add_observation(self, variable: str, time: F, value: F, uncertainty: F) -> Self:
        """
        Add an observation with absolute uncertainty.

        Parameters
        ----------
        variable : str
            Variable name
        time : float
            Observation time
        value : float
            Observed value
        uncertainty : float
            Absolute uncertainty (standard deviation)

        Returns
        -------
        Self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If uncertainty <= 0
        """
    def add_observation_relative(
        self, variable: str, time: F, value: F, relative_error: F
    ) -> Self:
        """
        Add an observation with relative uncertainty.

        Computes uncertainty as relative_error * abs(value).

        Parameters
        ----------
        variable : str
            Variable name
        time : float
            Observation time
        value : float
            Observed value
        relative_error : float
            Relative error as a fraction (e.g., 0.1 for 10%)

        Returns
        -------
        Self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If relative_error <= 0 or computed uncertainty <= 0
        """
    def set_reference_period(self, variable: str, start: F, end: F) -> Self:
        """
        Set reference period for anomaly calculation.

        Parameters
        ----------
        variable : str
            Variable name
        start : float
            Start of reference period
        end : float
            End of reference period

        Returns
        -------
        Self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If variable not found
        """
    def get_variable(self, name: str) -> VariableTarget:
        """
        Get observations for a specific variable.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        VariableTarget
            Clone of the variable target

        Raises
        ------
        ValueError
            If variable not found
        """
    @property
    def variable_names(self) -> list[str]:
        """Get names of all variables with observations."""

    @staticmethod
    def from_dataframe(
        df,
        time_col: str = "time",
        value_col: str = "value",
        uncertainty_col: str | None = None,
        relative_error: F | None = None,
    ) -> Target:
        """
        Create a Target from a pandas DataFrame.

        This is a convenience constructor added by the pandas_helpers module
        when pandas is available. See rscm.calibrate.pandas_helpers for details.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with observations
        time_col : str, optional
            Name of time column (default: "time")
        value_col : str, optional
            Name of value column (default: "value")
        uncertainty_col : str, optional
            Name of uncertainty column (default: None)
        relative_error : float, optional
            Relative error fraction (default: None)

        Returns
        -------
        Target
            Target object with observations from DataFrame
        """

# ==================== Likelihood ====================

class GaussianLikelihood:
    """
    Gaussian likelihood function for model-data comparison.

    Computes log-likelihood assuming independent Gaussian errors with
    observation-specific uncertainties.

    Parameters
    ----------
    normalize : bool, optional
        Whether to include normalization constant (default: False for MCMC)

    Examples
    --------
    >>> likelihood = GaussianLikelihood()
    >>> log_like = likelihood.compute(model_output, target)
    """

    def __init__(self, normalize: bool = False) -> None: ...
    def compute(self, model_output: dict[str, dict[F, F]], target: Target) -> F:
        """
        Compute log-likelihood for model output vs observations.

        Parameters
        ----------
        model_output : dict[str, dict[float, float]]
            Model outputs as {variable_name: {time: value}}
        target : Target
            Observation targets

        Returns
        -------
        float
            Log-likelihood value

        Raises
        ------
        ValueError
            If required variables/times missing or model output is invalid
        """

# ==================== Model Runner ====================

class ModelRunner:
    """
    Model runner that executes a model callable with parameter vectors.

    This class wraps a Python callable that takes a parameter dict and
    returns model output in the required format.

    Parameters
    ----------
    model_factory : callable
        Function that takes dict[str, float] and returns dict[str, dict[float, float]]
    param_names : list[str]
        Parameter names in the order expected by indexed parameter vectors
    output_variables : list[str]
        Names of output variables to extract from model

    Examples
    --------
    >>> def run_model(params):
    ...     # Run model with params
    ...     return {"Temperature|Global": {2020: 1.2, 2021: 1.3}}
    >>>
    >>> runner = ModelRunner(run_model, ["param1", "param2"], ["Temperature|Global"])
    >>> result = runner.run([0.5, 1.0])
    """

    def __init__(
        self,
        model_factory: Callable[[dict[str, F]], dict[str, dict[F, F]]],
        param_names: list[str],
        output_variables: list[str],
    ) -> None: ...
    def run(self, params: Sequence[F]) -> dict[str, dict[F, F]]:
        """
        Run model with parameter vector.

        Parameters
        ----------
        params : array-like
            Parameter values in param_names order

        Returns
        -------
        dict[str, dict[float, float]]
            Model outputs as {variable_name: {time: value}}

        Raises
        ------
        Exception
            If model execution fails
        """
    @property
    def param_names(self) -> list[str]:
        """Get parameter names."""

# ==================== MCMC Ensemble Sampling ====================

class ProgressInfo:
    """
    Progress information for MCMC sampling.

    Properties
    ----------
    iteration : int
        Current iteration number
    total_iterations : int
        Total iterations to run
    acceptance_rate : float
        Mean acceptance rate across walkers
    mean_log_prob : float
        Mean log probability across walkers
    """

    @property
    def iteration(self) -> int:
        """Current iteration."""
    @property
    def total_iterations(self) -> int:
        """Total iterations."""
    @property
    def acceptance_rate(self) -> F:
        """Mean acceptance rate."""
    @property
    def mean_log_prob(self) -> F:
        """Mean log probability."""

class WalkerInit:
    """
    Walker initialization strategy for MCMC sampler.

    Use static methods to construct initialization strategies.

    Examples
    --------
    >>> # Sample from prior
    >>> init = WalkerInit.from_prior()
    >>>
    >>> # Initialize in ball around point
    >>> init = WalkerInit.ball([0.5, 1.0], radius=0.1)
    >>>
    >>> # Explicit positions
    >>> positions = np.random.rand(32, 2)  # 32 walkers, 2 params
    >>> init = WalkerInit.explicit(positions)
    """

    @staticmethod
    def from_prior() -> WalkerInit:
        """Sample initial walker positions from parameter priors."""
    @staticmethod
    def ball(center: Sequence[F], radius: F = 0.01) -> WalkerInit:
        """
        Initialize walkers in a ball around a point.

        Parameters
        ----------
        center : array-like
            Center point for initialization
        radius : float, optional
            Radius of ball (default: 0.01)
        """
    @staticmethod
    def explicit(positions: Arr) -> WalkerInit:
        """
        Initialize walkers at explicit positions.

        Parameters
        ----------
        positions : NDArray[np.float64]
            Walker positions with shape (n_walkers, n_params)
        """

class Chain:
    """
    MCMC chain storage with diagnostics.

    Properties
    ----------
    param_names : list[str]
        Parameter names
    thin : int
        Thinning interval
    total_iterations : int
        Total iterations run

    Examples
    --------
    >>> # Extract samples
    >>> samples = chain.flat_samples(discard=100)
    >>> log_probs = chain.flat_log_probs(discard=100)
    >>>
    >>> # Check convergence
    >>> r_hat = chain.r_hat(discard=100)
    >>> print(f"R-hat: {r_hat}")
    >>> converged = chain.is_converged(discard=100, threshold=1.1)
    >>>
    >>> # Get parameter dict
    >>> param_dict = chain.to_param_dict(discard=100)
    """

    def flat_samples(self, discard: int = 0) -> Arr:
        """
        Get flattened samples array.

        Parameters
        ----------
        discard : int, optional
            Number of initial samples to discard as burn-in

        Returns
        -------
        NDArray[np.float64]
            Array of shape (n_samples * n_walkers, n_params) with all samples
        """
    def flat_log_probs(self, discard: int = 0) -> Arr:
        """
        Get flattened log probabilities.

        Parameters
        ----------
        discard : int, optional
            Number of initial samples to discard as burn-in

        Returns
        -------
        NDArray[np.float64]
            Array of shape (n_samples * n_walkers,) with log probabilities
        """
    def to_param_dict(self, discard: int = 0) -> dict[str, Arr]:
        """
        Get samples as dictionary of parameter arrays.

        Parameters
        ----------
        discard : int, optional
            Number of initial samples to discard as burn-in

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Dictionary mapping parameter names to sample arrays
        """
    def r_hat(self, discard: int = 0) -> dict[str, F]:
        """
        Compute Gelman-Rubin convergence diagnostic (R-hat).

        Parameters
        ----------
        discard : int, optional
            Number of initial samples to discard as burn-in

        Returns
        -------
        dict[str, float]
            R-hat statistic for each parameter. Values < 1.1 indicate convergence.
        """
    def ess(self, discard: int = 0) -> dict[str, F]:
        """
        Compute effective sample size.

        Parameters
        ----------
        discard : int, optional
            Number of initial samples to discard as burn-in

        Returns
        -------
        dict[str, float]
            Effective sample size for each parameter
        """
    def autocorr_time(self, discard: int = 0) -> dict[str, F]:
        """
        Compute autocorrelation time.

        Parameters
        ----------
        discard : int, optional
            Number of initial samples to discard as burn-in

        Returns
        -------
        dict[str, float]
            Autocorrelation time for each parameter
        """
    def is_converged(self, discard: int = 0, threshold: F = 1.1) -> bool:
        """
        Check if chain has converged based on R-hat.

        Parameters
        ----------
        discard : int, optional
            Number of initial samples to discard as burn-in
        threshold : float, optional
            R-hat threshold for convergence (default: 1.1)

        Returns
        -------
        bool
            True if all parameters have R-hat < threshold
        """
    def save(self, path: str) -> None:
        """
        Save chain to file.

        Parameters
        ----------
        path : str
            File path for saving
        """
    @staticmethod
    def load(path: str) -> Chain:
        """
        Load chain from file.

        Parameters
        ----------
        path : str
            File path to load from

        Returns
        -------
        Chain
            Loaded chain object
        """
    def merge(self, other: Chain) -> None:
        """
        Merge another chain into this one (in-place).

        Parameters
        ----------
        other : Chain
            Chain to merge into this one (must have same parameters and thin)

        Raises
        ------
        ValueError
            If chains have incompatible parameters or thinning
        """
    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
    @property
    def thin(self) -> int:
        """Thinning interval."""
    @property
    def total_iterations(self) -> int:
        """Total iterations run."""
    def __len__(self) -> int:
        """Return number of stored samples per walker."""

    def to_dataframe(self, discard: int = 0):
        """
        Convert chain to pandas DataFrame.

        This method is added by the pandas_helpers module when pandas is available.
        See rscm.calibrate.pandas_helpers for details.

        Parameters
        ----------
        discard : int, optional
            Number of initial samples to discard as burn-in

        Returns
        -------
        pandas.DataFrame
            DataFrame with multi-index (walker, iteration) and parameter columns
        """

class EnsembleSampler:
    """
    Affine-invariant ensemble MCMC sampler.

    Implements the Goodman & Weare (2010) stretch move algorithm for
    MCMC sampling with parallel walkers.

    Parameters
    ----------
    params : ParameterSet
        Parameter set with prior distributions
    runner : ModelRunner
        Model runner for evaluating likelihoods
    likelihood : GaussianLikelihood
        Likelihood function
    target : Target
        Observation targets
    stretch_param : float, optional
        Stretch move parameter (default: 2.0)

    Examples
    --------
    >>> sampler = EnsembleSampler(params, runner, likelihood, target)
    >>> chain = sampler.run(
    ...     n_iterations=1000, n_walkers=32, init=WalkerInit.from_prior(), thin=1
    ... )
    """

    def __init__(
        self,
        params: ParameterSet,
        runner: ModelRunner,
        likelihood: GaussianLikelihood,
        target: Target,
        stretch_param: F = 2.0,
    ) -> None: ...
    def run(
        self,
        n_iterations: int,
        n_walkers: int | None = None,
        init: WalkerInit | None = None,
        thin: int = 1,
    ) -> Chain:
        """
        Run MCMC sampling.

        Parameters
        ----------
        n_iterations : int
            Number of iterations to run
        n_walkers : int, optional
            Number of walkers (default: max(2 * n_params, 32))
        init : WalkerInit, optional
            Initialization strategy (default: from_prior)
        thin : int, optional
            Store every Nth sample (default: 1)

        Returns
        -------
        Chain
            MCMC chain with samples
        """
    def run_with_progress(
        self,
        n_iterations: int,
        callback: Callable[[ProgressInfo], None],
        n_walkers: int | None = None,
        init: WalkerInit | None = None,
        thin: int = 1,
    ) -> Chain:
        """
        Run MCMC with progress callback.

        Parameters
        ----------
        n_iterations : int
            Number of iterations to run
        callback : callable
            Function called with ProgressInfo after each iteration
        n_walkers : int, optional
            Number of walkers (default: max(2 * n_params, 32))
        init : WalkerInit, optional
            Initialization strategy (default: from_prior)
        thin : int, optional
            Store every Nth sample (default: 1)

        Returns
        -------
        Chain
            MCMC chain with samples
        """
    def run_with_checkpoint(
        self,
        n_iterations: int,
        checkpoint_path: str,
        checkpoint_every: int,
        n_walkers: int | None = None,
        init: WalkerInit | None = None,
        thin: int = 1,
        callback: Callable[[ProgressInfo], None] | None = None,
    ) -> Chain:
        """
        Run MCMC with checkpointing.

        Parameters
        ----------
        n_iterations : int
            Number of iterations to run
        checkpoint_path : str
            Path for checkpoint files
        checkpoint_every : int
            Checkpoint interval
        n_walkers : int, optional
            Number of walkers (default: max(2 * n_params, 32))
        init : WalkerInit, optional
            Initialization strategy (default: from_prior)
        thin : int, optional
            Store every Nth sample (default: 1)
        callback : callable, optional
            Progress callback function

        Returns
        -------
        Chain
            MCMC chain with samples
        """
    def resume_from_checkpoint(
        self,
        n_iterations: int,
        thin: int,
        checkpoint_every: int,
        checkpoint_path: str,
        progress_callback: Callable[[ProgressInfo], None] | None = None,
    ) -> Chain:
        """
        Resume sampling from checkpoint.

        Parameters
        ----------
        n_iterations : int
            Total number of iterations to reach (including already completed)
        thin : int
            Thinning interval (must match original run)
        checkpoint_every : int
            Checkpoint interval
        checkpoint_path : str
            Base path for checkpoint files
        progress_callback : callable, optional
            Progress callback function

        Returns
        -------
        Chain
            MCMC chain with all samples (original + new)
        """

# ==================== Point Estimation ====================

class Optimizer:
    """
    Optimization algorithm selector.

    Use static methods to construct optimizer instances.

    Examples
    --------
    >>> optimizer = Optimizer.random_search()
    """

    @staticmethod
    def random_search() -> Optimizer:
        """Random search optimizer (samples n points, returns best)."""

class OptimizationResult:
    """
    Result from point estimation optimization.

    Properties
    ----------
    best_params : list[float]
        Best parameter values found
    best_log_likelihood : float
        Log-likelihood at best parameters
    n_evaluations : int
        Number of model evaluations performed
    """

    @property
    def best_params(self) -> list[F]:
        """Best parameter values."""
    @property
    def best_log_likelihood(self) -> F:
        """Best log-likelihood found."""
    @property
    def n_evaluations(self) -> int:
        """Number of evaluations performed."""

class PointEstimator:
    """
    Point estimation optimizer for finding best-fit parameters.

    Parameters
    ----------
    params : ParameterSet
        Parameter set with prior distributions
    runner : ModelRunner
        Model runner
    likelihood : GaussianLikelihood
        Likelihood function
    target : Target
        Observation targets

    Examples
    --------
    >>> estimator = PointEstimator(params, runner, likelihood, target)
    >>> result = estimator.optimize(Optimizer.random_search(), n_samples=1000)
    >>> print(f"Best params: {result.best_params}")
    >>> print(f"Best log-likelihood: {result.best_log_likelihood}")
    """

    def __init__(
        self,
        params: ParameterSet,
        runner: ModelRunner,
        likelihood: GaussianLikelihood,
        target: Target,
    ) -> None: ...
    def optimize(self, optimizer: Optimizer, n_samples: int) -> OptimizationResult:
        """
        Run optimization to find best-fit parameters.

        Parameters
        ----------
        optimizer : Optimizer
            Optimization algorithm
        n_samples : int
            Number of samples to evaluate (for random search)

        Returns
        -------
        OptimizationResult
            Optimization result with best parameters
        """
    def clear_history(self) -> None:
        """Clear evaluation history."""
    def best(self) -> tuple[list[F], F] | None:
        """
        Get best parameters and log-likelihood from history.

        Returns
        -------
        tuple[list[float], float] | None
            (best_params, best_log_likelihood) or None if no evaluations
        """
    def evaluated_params(self) -> list[list[F]]:
        """Get all evaluated parameter vectors."""
    def evaluated_log_likelihoods(self) -> list[F]:
        """Get all evaluated log-likelihoods."""
    @property
    def n_params(self) -> int:
        """Number of parameters."""
    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
    @property
    def n_evaluations(self) -> int:
        """Number of evaluations performed."""

# Capability: Calibration

## Purpose

Enable calibration of model parameters against observations and uncertainty quantification via MCMC posterior sampling.

## ADDED Requirements

### Requirement: Distribution Trait

The system SHALL provide a Distribution trait for specifying prior distributions used in Bayesian calibration.

#### Scenario: Sample from uniform distribution

Given a Uniform(0, 1) distribution
When I sample 1000 values
Then all values are between 0 and 1
And the mean is approximately 0.5

#### Scenario: Sample from bounded normal distribution

Given a Bound(Normal(3.0, 0.5), lower=1.5, upper=6.0) distribution
When I sample 1000 values
Then all values are between 1.5 and 6.0
And the mean is approximately 3.0

#### Scenario: Log probability density

Given a Normal(0, 1) distribution
When I compute ln_pdf(0)
Then the result is approximately -0.919 (ln(1/sqrt(2*pi)))

### Requirement: ParameterSet Type

The system SHALL provide a ParameterSet type for defining calibration parameters with their prior distributions. Construction supports both dict-based and fluent builder patterns.

#### Scenario: Dict-based construction

Given a dict {"x": Uniform(0, 1), "y": Normal(0, 1)}
When I create ParameterSet(dict)
Then the ParameterSet has 2 parameters
And names() returns ["x", "y"]

#### Scenario: Fluent builder construction

Given an empty ParameterSet
When I call .add("sensitivity", Uniform(1.5, 6.0)).add("feedback", Normal(0, 0.5))
Then the ParameterSet has 2 parameters
And names() returns ["sensitivity", "feedback"]
And the builder returns self for chaining

#### Scenario: Sample parameter vectors

Given a ParameterSet with 2 parameters
When I sample 100 vectors
Then the result is a 100x2 array
And all values respect parameter bounds

#### Scenario: Compute log prior

Given a ParameterSet with Uniform(0, 1) for "x"
When I compute ln_prior([0.5])
Then the result is 0.0 (uniform density = 1)
When I compute ln_prior([1.5])
Then the result is -infinity (out of bounds)

### Requirement: Target Type

The system SHALL provide a Target type for representing observational data used in likelihood calculations. Supports fluent construction and DataFrame/scmdata import.

#### Scenario: Fluent construction with chaining

Given an empty Target
When I call .add_variable("Temperature", times, values, uncertainties).with_reference_period(1850, 1900)
Then the Target contains 1 variable with reference period set
And the builder returns self for chaining

#### Scenario: Construction from pandas DataFrame

Given a DataFrame with columns ["year", "gmst", "uncertainty"]
When I call Target.from_dataframe(df, time_col="year", variables={"Temperature": "gmst"}, uncertainty_col="uncertainty")
Then the Target contains 1 variable with times and uncertainties from DataFrame

#### Scenario: Construction with relative error

Given a DataFrame without explicit uncertainties
When I call Target.from_dataframe(df, ..., relative_error=0.05)
Then uncertainties are computed as 5% of observation values

#### Scenario: Reference period normalization

Given a Target with "Surface Temperature" from 1850-2020
When I set reference_period(1850, 1900)
Then values are converted to anomalies relative to 1850-1900 mean

### Requirement: GaussianLikelihood

The system SHALL provide a GaussianLikelihood type for computing log-likelihood of model outputs against observations.

#### Scenario: Compute likelihood with uncertainties

Given observations with values [1.0, 2.0] and uncertainties [0.1, 0.2]
And model output [1.05, 1.95]
When I compute ln_likelihood
Then the result reflects Gaussian probability given the uncertainties

#### Scenario: Relative error specification

Given a GaussianLikelihood with relative_error=0.05
And observations with values [10.0, 20.0]
When uncertainties are computed
Then they are [0.5, 1.0] (5% of values)

### Requirement: ModelRunner Trait

The system SHALL provide a ModelRunner trait as the interface for running models with different parameter values. Uses indexed arrays (not HashMap) for performance in hot loops.

#### Scenario: Run model with indexed parameters

Given a ModelRunner with param_names() = ["sensitivity", "feedback"]
When I call run([3.0, 0.1])
Then the model executes with sensitivity=3.0, feedback=0.1
And returns ModelOutput with requested variables

#### Scenario: Batch execution in parallel

Given a ModelRunner
When I call run_batch with 100 parameter vectors
Then all 100 models run in parallel via rayon
And I receive 100 Result<ModelOutput> (handling failures gracefully)

#### Scenario: Model failure handling

Given a ModelRunner and parameters that cause solver divergence
When I call run(invalid_params)
Then the result is Err(RSCMError)
And the likelihood function returns -INFINITY for this run

### Requirement: EnsembleSampler

The system SHALL provide an EnsembleSampler implementing the affine-invariant ensemble MCMC algorithm (Goodman & Weare 2010). Default walker count is max(2 * n_params, 32).

#### Scenario: Initialize from prior

Given an EnsembleSampler with 3 parameters (default 32 walkers)
When I call initialize(Prior)
Then I get SamplerState with 32 walker positions
And all positions are sampled from the prior

#### Scenario: Default walker count scales with dimension

Given an EnsembleSampler with 50 parameters
When constructed without explicit n_walkers
Then n_walkers defaults to 100 (2 * 50)

#### Scenario: Run MCMC steps with thinning

Given an initialized EnsembleSampler
When I run 10000 steps with thin=100
Then I get a Chain with 100 stored iterations
And each iteration has n_walkers positions

#### Scenario: Progress callback

Given an initialized EnsembleSampler
When I run with a progress callback
Then the callback is called each iteration
And receives ProgressInfo with iteration, acceptance_rate, mean_log_prob

#### Scenario: Checkpoint during run

Given an EnsembleSampler running with checkpoint_every=1000, checkpoint_path="chain.bin"
When 1000 iterations complete
Then the current SamplerState is saved to "chain.bin"
And sampling continues without interruption

#### Scenario: Resume from checkpoint

Given a checkpoint file from a previous run
When I call sampler.resume("chain.bin", n_steps=5000)
Then sampling continues from the saved state
And the returned Chain includes both old and new samples

#### Scenario: Stretch move proposal

Given walker at position x and complement walker at c
When z is sampled from g(z) = ((a-1)*U+1)^2/a with a=2.0
Then proposal y = c + z*(x - c)
And acceptance probability includes factor z^(n-1)

#### Scenario: Parallel evaluation

Given 32 walkers split into two groups of 16
When proposing moves for group 1 using group 2 as complement
Then all 16 proposals are evaluated in parallel via rayon

### Requirement: Chain Type

The system SHALL provide a Chain type for storing MCMC samples with support for thinning, burn-in removal, diagnostics, and data export.

#### Scenario: Access samples

Given a Chain from 1000 iterations with 32 walkers and 5 parameters
When I access samples
Then shape is (1000, 32, 5)

#### Scenario: Flat samples with burn-in

Given a Chain from 1000 iterations
When I call flat_samples(discard=200)
Then I get (800 * n_walkers, n_params) array
And the first 200 iterations are excluded

#### Scenario: Log probabilities

Given a Chain
When I access log_probs
Then shape is (n_iterations, n_walkers)

#### Scenario: Export to DataFrame

Given a Chain with param_names ["sensitivity", "feedback"]
When I call to_dataframe(discard=200)
Then I get a pandas DataFrame with columns ["sensitivity", "feedback"]
And rows correspond to flat_samples(discard=200)

#### Scenario: Diagnostics as chain methods

Given a Chain from 1000 iterations
When I call chain.r_hat(discard=200)
Then I get dict mapping parameter names to R-hat values
And I do not need a separate Diagnostics class

#### Scenario: Convergence check

Given a Chain where all parameters have R-hat < 1.1
When I call chain.is_converged(discard=200)
Then the result is True

Given a Chain where one parameter has R-hat = 1.25
When I call chain.is_converged(discard=200)
Then the result is False

### Requirement: PointEstimator

The system SHALL provide a PointEstimator for finding best-fit parameter values via optimization.

#### Scenario: Optimize with L-BFGS

Given a PointEstimator with parameters and target
When I call optimize(LBFGS)
Then the optimizer finds parameters maximizing likelihood
And returns PointEstimateResult with best_params

#### Scenario: Random search

Given a PointEstimator
When I call optimize(RandomSearch(n_samples=1000))
Then 1000 random parameter sets are evaluated
And the best one is returned

### Requirement: Convergence Diagnostics

The system SHALL provide convergence diagnostics as methods on the Chain type for discoverability and ergonomics.

#### Scenario: Gelman-Rubin R-hat

Given a Chain with multiple walkers
When I call chain.r_hat(discard=500)
Then I get dict mapping parameter names to R-hat values
And converged chains have R-hat < 1.1

#### Scenario: Effective sample size

Given a Chain with autocorrelated samples
When I call chain.ess(discard=500)
Then I get dict mapping parameter names to ESS values
And ESS < total_samples due to autocorrelation

#### Scenario: Acceptance fraction

Given a completed Chain
When I call chain.acceptance_fraction()
Then I get per-walker acceptance rates as list
And reasonable values are between 0.2 and 0.5

#### Scenario: Mean acceptance rate

Given a completed Chain
When I call chain.mean_acceptance_rate()
Then I get a single float representing overall acceptance
And this is useful for quick health checks

#### Scenario: Autocorrelation time

Given a Chain with correlated samples
When I call chain.autocorr_time(discard=500)
Then I get dict mapping parameter names to autocorrelation times
And thin interval should be >= max autocorrelation time for independent samples

### Requirement: Chain Persistence

The system SHALL support saving and loading MCMC chains for resumption and analysis.

#### Scenario: Save chain to file

Given a Chain with 10000 iterations
When I call storage.save(chain)
Then the chain is written to a binary file
And the file can be memory-mapped for large chains

#### Scenario: Resume from checkpoint

Given a saved SamplerState
When I create a new EnsembleSampler and load the state
Then I can continue sampling from where I left off

### Requirement: Python Bindings

The system MUST provide PyO3 bindings exposing a modern, Pythonic calibration API.

#### Scenario: Create ParameterSet with dict constructor

```python
params = ParameterSet({
    "x": Uniform(0, 1),
    "y": Bound(Normal(0, 1), lower=-2, upper=2),
})
```

Then the ParameterSet is usable with EnsembleSampler

#### Scenario: Create ParameterSet with fluent builder

```python
params = (ParameterSet()
    .add("x", Uniform(0, 1))
    .add("y", Bound(Normal(0, 1), lower=-2, upper=2)))
```

Then .add() returns self for method chaining

#### Scenario: Run sampler with progress bar

```python
sampler = EnsembleSampler(params, target, runner, likelihood)
state = sampler.initialize("prior")
chain = sampler.run(state, n_steps=10000, thin=100, progress=True)
```

Then a progress bar is displayed during sampling
And chain.flat_samples(discard=1000) returns numpy array

#### Scenario: Run sampler with checkpointing

```python
chain = sampler.run(state, n_steps=10000, checkpoint_every=1000, checkpoint_path="chain.bin")
```

Then checkpoint is saved every 1000 steps
And sampler.resume("chain.bin", n_steps=5000) continues from checkpoint

#### Scenario: Access diagnostics as chain methods

```python
r_hat = chain.r_hat(discard=1000)
ess = chain.ess(discard=1000)
converged = chain.is_converged(discard=1000)
```

Then r_hat and ess are dicts mapping parameter names to values
And converged is a boolean

#### Scenario: Export chain to DataFrame

```python
df = chain.to_dataframe(discard=1000)
```

Then df is a pandas DataFrame with parameter names as columns
And df.describe() shows posterior statistics

### Requirement: Python Visualization

The system SHALL provide visualization utilities via a `plot` namespace on Chain for discoverability.

#### Scenario: Trace plot via plot namespace

Given a Chain from Python
When I call chain.plot.trace(discard=1000)
Then a matplotlib figure is created
And shows parameter values vs iteration for each parameter

#### Scenario: Corner plot via plot namespace

Given a Chain from Python
When I call chain.plot.corner(discard=1000)
Then a corner plot is created using the corner library
And shows 1D marginals and 2D correlations

#### Scenario: Autocorrelation plot

Given a Chain from Python
When I call chain.plot.autocorr()
Then a matplotlib figure shows autocorrelation vs lag for each parameter

#### Scenario: Acceptance rate plot

Given a Chain from Python
When I call chain.plot.acceptance()
Then a matplotlib figure shows acceptance rate over iterations

#### Scenario: Posterior predictive check

Given a Chain and ModelRunner
When I call chain.plot.posterior_predictive(runner, target, n_samples=100, discard=1000)
Then model is run with 100 posterior samples
And output is plotted against target observations with uncertainty bands

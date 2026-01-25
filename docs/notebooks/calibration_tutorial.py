# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RSCM Calibration Framework Tutorial
#
# This tutorial demonstrates how to calibrate climate model parameters using the RSCM calibration framework.
#
# ## Overview
#
# The calibration framework provides two main workflows:
#
# 1. **Point Estimation**: Find the single best-fit parameter set that maximizes likelihood
# 2. **MCMC Sampling**: Explore the full posterior distribution to quantify uncertainty
#
# This tutorial covers:
# - **Beginner**: Basic calibration workflow with a simple model
# - **Intermediate**: Two-layer climate model calibration with diagnostics
# - **Advanced**: Checkpoint/resume, thinning, and DataFrame integration

# %% [markdown]
# ## Setup
#
# First, import the required libraries:

# %%
import matplotlib.pyplot as plt
import numpy as np

from rscm.calibrate import (
    EnsembleSampler,
    # Likelihood computation
    GaussianLikelihood,
    # Model runner interface
    ModelRunner,
    Optimizer,
    # Parameter set specification
    ParameterSet,
    # Point estimation
    PointEstimator,
    # Observations and targets
    Target,
    # Distributions for parameter priors
    Uniform,
    WalkerInit,
    # Progress tracking
    progress,
)

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ---
#
# # Part 1: Beginner - Quadratic Model Calibration
#
# We'll start with a simple quadratic model: `y = a*x² + b*x + c`
#
# This demonstrates the complete workflow without climate model complexity.

# %% [markdown]
# ## Step 1: Generate Synthetic Observations
#
# First, we create synthetic observations from a "true" model with known parameters.

# %%
# True parameters (what we're trying to recover)
true_a = -0.3
true_b = 3.0
true_c = 1.5


def _model(x: np.ndarray, a: float, b: float, c: float):
    return a * x**2 + b * x + c


# Generate observations: y = a*x^2 + b*x + c + noise
x_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = _model(x_obs, true_a, true_b, true_c)
y_obs = y_true + np.random.normal(0, 0.1, size=len(x_obs))
uncertainty = 0.1  # Standard deviation of noise

# Plot observations
plt.figure(figsize=(8, 5))
plt.errorbar(x_obs, y_obs, yerr=uncertainty, fmt="o", label="Observations", capsize=5)
plt.plot(x_obs, y_true, "--", label="True model", alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Synthetic Observations")
plt.grid(True, alpha=0.3)
plt.show()

print(f"True parameters: a={true_a}, b={true_b}, c={true_c}")

# %% [markdown]
# ## Step 2: Define Parameter Priors
#
# We specify what we know about the parameters before seeing data (the prior distribution).
# This might be an initial guess

# %%
# Create parameter set with uniform priors
params = ParameterSet()
params.add("a", Uniform(-2.0, 2.0))
params.add("b", Uniform(0.0, 5.0))
params.add("c", Uniform(0.0, 5.0))

print(f"Parameter names: {params.param_names}")
print(f"Number of parameters: {len(params)}")

# %% [markdown]
# ## Step 3: Create Observation Target
#
# The target specifies what observations we want to match.

# %%
# Create target with observations
target = Target()
for x, y in zip(x_obs, y_obs):
    target.add_observation("y", float(x), float(y), uncertainty)

print(f"Target has {len(x_obs)} observations for variable 'y'")


# %% [markdown]
# ## Step 4: Define the Model Runner
#
# The model runner takes parameters and produces model outputs.


# %%
def model_factory(param_dict):
    """
    Run the linear model: y = a*x^2 + b*x + c

    Parameters
    ----------
    param_dict : dict
        Dictionary with keys 'a', 'b' and 'c'

    Returns
    -------
    dict
        Model outputs as {variable_name: {time: value}}
    """
    a = param_dict["a"]
    b = param_dict["b"]
    c = param_dict["c"]

    # Compute model outputs for each observation time
    return {"y": {float(x): float(a * x**2 + b * x + c) for x in x_obs}}


# Create model runner
runner = ModelRunner(
    model_factory=model_factory,
    param_names=["a", "b", "c"],
    output_variables=["y"],
)

print("Model runner created")

# %% [markdown]
# ## Step 5: Point Estimation - Find Best Fit
#
# Use random search to find the parameter set that best fits the observations.

# %%
# Create likelihood function
likelihood = GaussianLikelihood()

# Create point estimator
estimator = PointEstimator(params, runner, likelihood, target)

# Run random search with 500 samples
result = estimator.optimize(Optimizer.random_search(), n_samples=50000)

# Extract best parameters
best_params_vec = result.best_params
param_names = estimator.param_names
best_params = dict(zip(param_names, best_params_vec))

print("\n=== Point Estimation Results ===")
print(
    f"Best parameters: a={best_params['a']:.3f}, b={best_params['b']:.3f}, c={best_params['c']:.3f}"
)
print(f"True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
print(f"Best log likelihood: {result.best_log_likelihood:.3f}")
print(f"Total evaluations: {estimator.n_evaluations}")

# %% [markdown]
# ## Step 6: Visualise Point Estimation Results

# %%
# Plot best fit vs observations
x_plot = np.linspace(0, 6, 100)
y_best = _model(x_plot, **best_params)
y_true_plot = _model(x_plot, true_a, true_b, true_c)

plt.figure(figsize=(10, 6))
plt.errorbar(
    x_obs,
    y_obs,
    yerr=uncertainty,
    fmt="o",
    label="Observations",
    capsize=5,
    markersize=8,
)
plt.plot(x_plot, y_true_plot, "--", label="True model", alpha=0.7, linewidth=2)
plt.plot(x_plot, y_best, "-", label="Best fit", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Point Estimation: Best Fit vs Observations")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Step 7: MCMC Sampling - Quantify Uncertainty
#
# Point estimation gives us the best fit, but doesn't tell us how certain we are.
# MCMC sampling explores the full posterior distribution.

# %%
# Create ensemble sampler
sampler = EnsembleSampler(params, runner, likelihood, target)

# Run MCMC with progress bar
print("Running MCMC sampling...")
chain = sampler.run_with_progress(
    n_iterations=1000,
    init=WalkerInit.from_prior(),
    thin=1,
    progress_callback=progress.create_simple_callback(),
)

print("\nMCMC complete!")
print(f"Total iterations: {chain.total_iterations}")
print(f"Number of walkers: {sampler.default_n_walkers()}")
print(f"Parameter names: {chain.param_names}")

# %% [markdown]
# ## Step 8: Check Convergence Diagnostics
#
# Before trusting the results, we need to verify the MCMC chain has converged.

# %%
# Discard first 200 iterations as burn-in
discard = 200

# Compute diagnostics
r_hat = chain.r_hat(discard)
ess = chain.ess(discard)
autocorr_time = chain.autocorr_time(discard)
is_converged = chain.is_converged(discard, threshold=1.2)

print("\n=== Convergence Diagnostics ===")
print("\nR-hat (should be < 1.1 for convergence):")
for param, value in r_hat.items():
    print(f"  {param}: {value:.4f}")

print("\nEffective Sample Size:")
for param, value in ess.items():
    print(f"  {param}: {value:.0f}")

print("\nAutocorrelation Time:")
for param, value in autocorr_time.items():
    print(f"  {param}: {value:.1f} steps")

print(f"\nConverged: {is_converged}")

# %% [markdown]
# ## Step 9: Analyse Posterior Distribution

# %%
# Get samples as dictionary
samples_dict = chain.to_param_dict(discard)

# Compute posterior statistics
a_samples = samples_dict["a"]
b_samples = samples_dict["b"]
c_samples = samples_dict["c"]

print("\n=== Posterior Statistics ===")
print("\nParameter 'a' (quadratic coefficient):")
print(f"  Mean: {np.mean(a_samples):.3f}")
print(f"  Std:  {np.std(a_samples):.3f}")
print(
    f"  95% credible interval: [{np.percentile(a_samples, 2.5):.3f}, {np.percentile(a_samples, 97.5):.3f}]"
)
print(f"  True value: {true_a:.3f}")

print("\nParameter 'b' (linear coefficient):")
print(f"  Mean: {np.mean(b_samples):.3f}")
print(f"  Std:  {np.std(b_samples):.3f}")
print(
    f"  95% credible interval: [{np.percentile(b_samples, 2.5):.3f}, {np.percentile(b_samples, 97.5):.3f}]"
)
print(f"  True value: {true_b:.3f}")

print("\nParameter 'c' (constant):")
print(f"  Mean: {np.mean(c_samples):.3f}")
print(f"  Std:  {np.std(c_samples):.3f}")
print(
    f"  95% credible interval: [{np.percentile(c_samples, 2.5):.3f}, {np.percentile(c_samples, 97.5):.3f}]"
)
print(f"  True value: {true_c:.3f}")

# %% [markdown]
# ## Step 10: Visualise MCMC Results

# %%
# Create trace plots and posteriors for all 3 parameters
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Parameter a
axes[0, 0].plot(a_samples, alpha=0.5)
axes[0, 0].axhline(true_a, color="red", linestyle="--", label="True value")
axes[0, 0].set_ylabel("a")
axes[0, 0].set_title("Trace: Parameter a (quadratic)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(a_samples, bins=30, density=True, alpha=0.7, edgecolor="black")
axes[0, 1].axvline(true_a, color="red", linestyle="--", linewidth=2, label="True value")
axes[0, 1].axvline(
    np.mean(a_samples), color="blue", linestyle="-", linewidth=2, label="Posterior mean"
)
axes[0, 1].set_xlabel("a")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_title("Posterior: Parameter a")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Parameter b
axes[1, 0].plot(b_samples, alpha=0.5)
axes[1, 0].axhline(true_b, color="red", linestyle="--", label="True value")
axes[1, 0].set_ylabel("b")
axes[1, 0].set_title("Trace: Parameter b (linear)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(b_samples, bins=30, density=True, alpha=0.7, edgecolor="black")
axes[1, 1].axvline(true_b, color="red", linestyle="--", linewidth=2, label="True value")
axes[1, 1].axvline(
    np.mean(b_samples), color="blue", linestyle="-", linewidth=2, label="Posterior mean"
)
axes[1, 1].set_xlabel("b")
axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title("Posterior: Parameter b")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Parameter c
axes[2, 0].plot(c_samples, alpha=0.5)
axes[2, 0].axhline(true_c, color="red", linestyle="--", label="True value")
axes[2, 0].set_xlabel("Sample")
axes[2, 0].set_ylabel("c")
axes[2, 0].set_title("Trace: Parameter c (constant)")
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].hist(c_samples, bins=30, density=True, alpha=0.7, edgecolor="black")
axes[2, 1].axvline(true_c, color="red", linestyle="--", linewidth=2, label="True value")
axes[2, 1].axvline(
    np.mean(c_samples), color="blue", linestyle="-", linewidth=2, label="Posterior mean"
)
axes[2, 1].set_xlabel("c")
axes[2, 1].set_ylabel("Density")
axes[2, 1].set_title("Posterior: Parameter c")
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Joint posteriors (pairwise 2D histograms for 3 parameters)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# a vs b
axes[0].hist2d(a_samples, b_samples, bins=30, cmap="Blues")
axes[0].plot(
    true_a,
    true_b,
    "r*",
    markersize=20,
    label="True values",
    markeredgecolor="white",
    markeredgewidth=1.5,
)
axes[0].set_xlabel("a")
axes[0].set_ylabel("b")
axes[0].set_title("Joint Posterior: a vs b")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# a vs c
axes[1].hist2d(a_samples, c_samples, bins=30, cmap="Blues")
axes[1].plot(
    true_a,
    true_c,
    "r*",
    markersize=20,
    label="True values",
    markeredgecolor="white",
    markeredgewidth=1.5,
)
axes[1].set_xlabel("a")
axes[1].set_ylabel("c")
axes[1].set_title("Joint Posterior: a vs c")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# b vs c
h = axes[2].hist2d(b_samples, c_samples, bins=30, cmap="Blues")
axes[2].plot(
    true_b,
    true_c,
    "r*",
    markersize=20,
    label="True values",
    markeredgecolor="white",
    markeredgewidth=1.5,
)
axes[2].set_xlabel("b")
axes[2].set_ylabel("c")
axes[2].set_title("Joint Posterior: b vs c")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.colorbar(h[3], ax=axes[2], label="Sample density")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
#
# # Part 2: Intermediate - Two-Layer Climate Model
#
# Now let's calibrate a real climate model - the two-layer energy balance model.
#
# This model simulates surface and deep ocean temperatures in response to radiative forcing.

# %%
# Import RSCM components
from rscm.core import (
    InterpolationStrategy,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)
from rscm.two_layer import TwoLayerBuilder

# %% [markdown]
# ## Step 1: Generate Synthetic Observations
#
# We'll create synthetic temperature observations from a "true" model.

# %%
# True parameter values (physically realistic)
true_climate_params = {
    "lambda0": 1.1,  # W/(m² K) - climate feedback parameter
    "efficacy": 1.3,  # dimensionless - ocean heat uptake efficacy
    "a": 0.05,  # W/(m² K²) - nonlinear feedback
    "eta": 0.7,  # W/(m² K) - heat exchange coefficient
    "heat_capacity_deep": 100.0,  # W yr/(m² K) - deep ocean
    "heat_capacity_surface": 8.0,  # W yr/(m² K) - surface layer
}

# Create true model and run it
true_component = TwoLayerBuilder.from_parameters(true_climate_params).build()

# Simple forcing scenario: constant 3 W/m² forcing
forcing_collection = TimeseriesCollection()
forcing_collection.add_timeseries(
    "Effective Radiative Forcing",
    Timeseries(
        np.array([3.0]),
        TimeAxis.from_bounds(np.array([2000.0, 2010.0])),
        "W/m^2",
        InterpolationStrategy.Previous,
    ),
)

# Run model
result = true_component.solve(2000, 2010, forcing_collection)
true_temp = result.get("Surface Temperature").as_scalar()

# Add observational noise
obs_temp = true_temp + np.random.normal(0, 0.05)  # Small noise
obs_uncertainty = 0.1  # K

print(f"True temperature: {true_temp:.3f} K")
print(f"Observed temperature: {obs_temp:.3f} K")
print(f"Uncertainty: ±{obs_uncertainty} K")

# %% [markdown]
# ## Step 2: Set Up Calibration
#
# We'll calibrate two key parameters: `lambda0` (climate feedback) and `a` (nonlinear feedback).
# Other parameters will be fixed at their true values.

# %%
# Define parameter priors (wide enough to include true values)
climate_params = ParameterSet()
climate_params.add("lambda0", Uniform(0.8, 1.5))  # Typical range for climate feedback
climate_params.add("a", Uniform(0.0, 0.1))  # Typical range for nonlinear feedback

# Create target from observations
climate_target = Target()
climate_target.add_observation("Surface Temperature", 2010.0, obs_temp, obs_uncertainty)

print(f"Calibrating {len(climate_params)} parameters: {climate_params.param_names}")

# %% [markdown]
# ## Step 3: Create Climate Model Runner

# %%
# Fixed parameters (not being calibrated)
fixed_climate_params = {
    "efficacy": true_climate_params["efficacy"],
    "eta": true_climate_params["eta"],
    "heat_capacity_deep": true_climate_params["heat_capacity_deep"],
    "heat_capacity_surface": true_climate_params["heat_capacity_surface"],
}


def climate_model_factory(param_values):
    """
    Create and run the two-layer climate model.

    Parameters
    ----------
    param_values : dict
        Dictionary with keys 'lambda0' and 'a'

    Returns
    -------
    dict
        Model outputs as {variable_name: {time: value}}
    """
    # Combine calibrated and fixed parameters
    all_params = fixed_climate_params.copy()
    all_params.update(param_values)

    # Create and run model
    component = TwoLayerBuilder.from_parameters(all_params).build()

    forcing = TimeseriesCollection()
    forcing.add_timeseries(
        "Effective Radiative Forcing",
        Timeseries(
            np.array([3.0]),
            TimeAxis.from_bounds(np.array([2000.0, 2010.0])),
            "W/m^2",
            InterpolationStrategy.Previous,
        ),
    )

    result = component.solve(2000, 2010, forcing)
    temp = result.get("Surface Temperature").as_scalar()

    return {"Surface Temperature": {2010.0: float(temp)}}


# Create runner
climate_runner = ModelRunner(
    model_factory=climate_model_factory,
    param_names=["lambda0", "a"],
    output_variables=["Surface Temperature"],
)

print("Climate model runner created")

# %% [markdown]
# ## Step 4: Point Estimation with Climate Model

# %%
# Create estimator
climate_estimator = PointEstimator(
    climate_params, climate_runner, GaussianLikelihood(), climate_target
)

# Run random search
print("Running point estimation (1000 samples)...")
climate_result = climate_estimator.optimize(Optimizer.random_search(), n_samples=1000)

# Extract results
best_climate_params_vec = climate_result.best_params
best_climate_params = dict(zip(climate_estimator.param_names, best_climate_params_vec))

print("\n=== Climate Model Point Estimation ===")
print("Best parameters:")
print(
    f"  lambda0 = {best_climate_params['lambda0']:.4f} (true: {true_climate_params['lambda0']:.4f})"
)
print(
    f"  a       = {best_climate_params['a']:.4f} (true: {true_climate_params['a']:.4f})"
)
print(f"Best log likelihood: {climate_result.best_log_likelihood:.3f}")

# %% [markdown]
# ## Step 5: MCMC Sampling for Climate Model
#
# Now we'll use MCMC to fully characterise the posterior distribution.

# %%
# Create ensemble sampler
climate_sampler = EnsembleSampler(
    climate_params, climate_runner, GaussianLikelihood(), climate_target
)

# Run MCMC with progress tracking
print("Running MCMC for climate model (1000 iterations)...")
climate_chain = climate_sampler.run_with_progress(
    n_iterations=1000,
    init=WalkerInit.from_prior(),
    thin=1,
    progress_callback=progress.create_simple_callback(),
)

print(f"\nMCMC complete! Total iterations: {climate_chain.total_iterations}")

# %% [markdown]
# ## Step 6: Analyse Climate Model Results

# %%
# Check convergence
discard_climate = 200

r_hat_climate = climate_chain.r_hat(discard_climate)
ess_climate = climate_chain.ess(discard_climate)
is_converged_climate = climate_chain.is_converged(discard_climate, threshold=1.2)

print("=== Climate Model Diagnostics ===")
print("\nR-hat:")
for param, value in r_hat_climate.items():
    status = "✓" if value < 1.2 else "✗"
    print(f"  {param}: {value:.4f} {status}")

print("\nEffective Sample Size:")
for param, value in ess_climate.items():
    print(f"  {param}: {value:.0f}")

print(f"\nConverged: {is_converged_climate}")

# %%
# Get posterior samples
climate_samples_dict = climate_chain.to_param_dict(discard_climate)
lambda0_samples = climate_samples_dict["lambda0"]
a_samples_climate = climate_samples_dict["a"]

print("\n=== Climate Posterior Statistics ===")
print("\nlambda0 (climate feedback):")
print(f"  Posterior mean: {np.mean(lambda0_samples):.4f}")
print(f"  Posterior std:  {np.std(lambda0_samples):.4f}")
print(f"  True value:     {true_climate_params['lambda0']:.4f}")
print(
    f"  95% CI: [{np.percentile(lambda0_samples, 2.5):.4f}, {np.percentile(lambda0_samples, 97.5):.4f}]"
)

print("\na (nonlinear feedback):")
print(f"  Posterior mean: {np.mean(a_samples_climate):.4f}")
print(f"  Posterior std:  {np.std(a_samples_climate):.4f}")
print(f"  True value:     {true_climate_params['a']:.4f}")
print(
    f"  95% CI: [{np.percentile(a_samples_climate, 2.5):.4f}, {np.percentile(a_samples_climate, 97.5):.4f}]"
)

# %%
# Visualise climate model posteriors
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# lambda0 posterior
axes[0].hist(lambda0_samples, bins=40, density=True, alpha=0.7, edgecolor="black")
axes[0].axvline(
    true_climate_params["lambda0"],
    color="red",
    linestyle="--",
    linewidth=2,
    label="True value",
)
axes[0].axvline(
    np.mean(lambda0_samples),
    color="blue",
    linestyle="-",
    linewidth=2,
    label="Posterior mean",
)
axes[0].set_xlabel("lambda0 [W/(m² K)]")
axes[0].set_ylabel("Density")
axes[0].set_title("Posterior: Climate Feedback Parameter")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# a posterior
axes[1].hist(a_samples_climate, bins=40, density=True, alpha=0.7, edgecolor="black")
axes[1].axvline(
    true_climate_params["a"],
    color="red",
    linestyle="--",
    linewidth=2,
    label="True value",
)
axes[1].axvline(
    np.mean(a_samples_climate),
    color="blue",
    linestyle="-",
    linewidth=2,
    label="Posterior mean",
)
axes[1].set_xlabel("a [W/(m² K²)]")
axes[1].set_ylabel("Density")
axes[1].set_title("Posterior: Nonlinear Feedback")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
#
# # Part 3: Advanced - Checkpointing and DataFrame Integration
#
# For long-running calibrations, we need:
# - **Checkpointing**: Save progress and resume if interrupted
# - **Thinning**: Reduce memory usage for long chains
# - **DataFrame integration**: Export to pandas for analysis

# %% [markdown]
# ## Checkpoint/Resume Workflow

# %%
import os
import tempfile

# Create temporary directory for checkpoints
temp_dir = tempfile.mkdtemp()
checkpoint_path = os.path.join(temp_dir, "chain_checkpoint")

print(f"Checkpoint path: {checkpoint_path}")

# %%
# First run: 500 iterations with checkpointing every 100 steps
print("Running first batch (500 iterations)...")
chain_1 = sampler.run_with_checkpoint(
    n_iterations=500,
    init=WalkerInit.from_prior(),
    thin=1,
    checkpoint_every=100,
    checkpoint_path=checkpoint_path,
)

print(f"First batch complete: {chain_1.total_iterations} iterations")
print(f"Checkpoint files created: {checkpoint_path}.chain, {checkpoint_path}.state")

# %%
# Simulate interruption and resume
print("\nResuming from checkpoint (500 -> 1000 total iterations)...")
chain_2 = sampler.resume_from_checkpoint(
    n_iterations=1000,  # Total iterations, not additional!
    checkpoint_path=checkpoint_path,
    checkpoint_every=100,
    thin=1,
)

print(f"Resumed chain complete: {chain_2.total_iterations} iterations")
print("Chain contains both original and new samples")

# %% [markdown]
# ## Thinning for Memory Efficiency

# %%
# Run with heavy thinning
print("Running with thinning (thin=10)...")
chain_thinned = sampler.run(
    n_iterations=1000,
    init=WalkerInit.from_prior(),
    thin=10,  # Store only every 10th sample
)

# Compare memory usage
samples_no_thin = chain.flat_samples(discard=0)
samples_thinned = chain_thinned.flat_samples(discard=0)

print("\n=== Thinning Comparison ===")
print(f"No thinning (thin=1):  {samples_no_thin.shape[0]:,} samples")
print(f"With thinning (thin=10): {samples_thinned.shape[0]:,} samples")
print(f"Memory reduction: {samples_no_thin.shape[0] / samples_thinned.shape[0]:.1f}x")

# %% [markdown]
# ## Chain Persistence - Save and Load

# %%
# Save chain to file
chain_file = os.path.join(temp_dir, "saved_chain.bin")
chain.save(chain_file)

print(f"Chain saved to: {chain_file}")
print(f"File size: {os.path.getsize(chain_file) / 1024:.1f} KB")

# Load chain from file
from rscm.calibrate import Chain

loaded_chain = Chain.load(chain_file)

print("\nChain loaded successfully")
print(f"Total iterations: {loaded_chain.total_iterations}")
print(f"Parameter names: {loaded_chain.param_names}")

# %% [markdown]
# ## Chain Merging

# %%
# Merge two chains (e.g., from parallel runs)
chain_copy = Chain.load(chain_file)  # Make a copy
original_iterations = chain_copy.total_iterations

# Merge in place
chain_copy.merge(loaded_chain)

print("=== Chain Merging ===")
print(f"Original chain: {original_iterations} iterations")
print(f"Merged chain:   {chain_copy.total_iterations} iterations")
print("Samples are concatenated in order")

# %% [markdown]
# ## DataFrame Integration (requires pandas)

# %%
# Convert chain to DataFrame
df = chain.to_dataframe(discard=200)

print("=== Chain as DataFrame ===")
print(f"\nShape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())


# %% [markdown]
# ## Progress Tracking with ProgressTracker

# %%
# Create progress tracker to record metrics history
tracker = progress.ProgressTracker()

# Run with tracker
print("Running with progress tracker...")
tracked_chain = sampler.run_with_progress(
    n_iterations=500,
    init=WalkerInit.from_prior(),
    thin=1,
    progress_callback=tracker,
)

# Plot convergence metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Acceptance rate over time
axes[0].plot(tracker.iterations, tracker.acceptance_rates)
axes[0].axhline(0.25, color="red", linestyle="--", alpha=0.5, label="Target (25%)")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Acceptance Rate")
axes[0].set_title("Acceptance Rate vs Iteration")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Mean log probability over time
axes[1].plot(tracker.iterations, tracker.mean_log_probs)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Mean Log Probability")
axes[1].set_title("Mean Log Probability vs Iteration")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nTracked {len(tracker.iterations)} iterations")
print(f"Final acceptance rate: {tracker.acceptance_rates[-1]:.2%}")

# %% [markdown]
# ## Clean Up

# %%
# Clean up temporary files
import shutil

shutil.rmtree(temp_dir)
print("Temporary files cleaned up")

# %% [markdown]
# ---
#
# # Summary
#
# This tutorial covered:
#
# **Beginner:**
# - Defining parameter priors with distributions (Uniform, Normal, Bound)
# - Creating observation targets
# - Setting up a model runner
# - Point estimation to find best-fit parameters
# - MCMC sampling to quantify uncertainty
# - Convergence diagnostics (R-hat, ESS, acceptance rate)
#
# **Intermediate:**
# - Calibrating a real climate model (two-layer energy balance)
# - Interpreting posterior distributions
# - Visualising results (trace plots, posteriors, joint distributions)
#
# **Advanced:**
# - Checkpoint/resume for long-running calibrations
# - Thinning to reduce memory usage
# - Chain persistence (save/load/merge)
# - DataFrame integration for analysis
# - Progress tracking with custom callbacks
#
# ## Next Steps
#
# - Try calibrating with multiple observation variables
# - Experiment with different prior distributions (Normal, Bound, LogNormal)
# - Use warm-starting: initialise MCMC walkers near the point estimate
# - Implement custom likelihood functions for specific applications
# - Explore posterior predictive checks to validate model fit
#
# ## Further Reading
#
# - **Algorithm reference**: Goodman & Weare (2010), "Ensemble samplers with affine invariance" - https://doi.org/10.2140/camcos.2010.5.65
# - **RSCM documentation**: See rustdoc for detailed API reference
# - **Convergence diagnostics**: Gelman et al. (2013), "Bayesian Data Analysis", Chapter 11

# %%

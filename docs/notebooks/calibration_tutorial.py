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
# 1. **Point Estimation**: Find the single best-fit parameter set that maximises the likelihood.
# 2. **MCMC Sampling**: Explore the full posterior distribution to quantify uncertainty.
#
# This tutorial covers:
#
# - **Part 1**: The basic calibration workflow with a simple quadratic model.
# - **Part 2**: Calibrating a two-layer climate model.
# - **Part 3**: Checkpoint and resume, thinning, persistence and DataFrame integration.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd

from rscm.calibrate import (
    Bound,
    EnsembleSampler,
    GaussianLikelihood,
    LogNormal,
    ModelRunner,
    Normal,
    Optimiser,
    ParameterSet,
    PointEstimator,
    Target,
    Uniform,
    WalkerInit,
    progress,
)

# Seeds the observation noise only: the Rust samplers use their own unseeded RNG.
np.random.seed(42)

# %% [markdown]
# Two helpers defined in a hidden cell, `posterior_table` and `plot_posteriors`,
# summarise and plot posterior samples in both parts.

# %% tags=["remove_input"]
import matplotlib.pyplot as plt


def posterior_table(
    samples: dict[str, np.ndarray], truth: dict[str, float]
) -> pd.DataFrame:
    """Tabulate the mean, spread and 95% credible interval of each parameter."""
    return pd.DataFrame(
        {
            name: {
                "mean": np.mean(values),
                "std": np.std(values),
                "2.5%": np.percentile(values, 2.5),
                "97.5%": np.percentile(values, 97.5),
                "true": truth[name],
            }
            for name, values in samples.items()
        }
    ).T.round(4)


def plot_posteriors(
    samples: dict[str, np.ndarray], truth: dict[str, float], trace: bool = False
) -> None:
    """Plot a histogram per parameter, with an optional sample trace beside it."""
    n_cols = 2 if trace else 1
    _, axes = plt.subplots(
        len(samples), n_cols, figsize=(7 * n_cols, 3 * len(samples)), squeeze=False
    )
    for row, (name, values) in zip(axes, samples.items()):
        if trace:
            row[0].plot(values, alpha=0.5)
            row[0].axhline(truth[name], color="red", linestyle="--", label="True value")
            row[0].set_xlabel("Sample")
            row[0].set_ylabel(name)
            row[0].set_title(f"Trace: {name}")
            row[0].legend()
            row[0].grid(True, alpha=0.3)
        ax = row[-1]
        ax.hist(values, bins=30, density=True, alpha=0.7, edgecolor="black")
        ax.axvline(
            truth[name], color="red", linestyle="--", linewidth=2, label="True value"
        )
        ax.axvline(np.mean(values), color="blue", linewidth=2, label="Posterior mean")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior: {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Part 1: Quadratic Model Calibration
#
# We start with a simple quadratic model, `y = a*x² + b*x + c`.
# This demonstrates the complete workflow without climate model complexity.
#
# ### Synthetic Observations
#
# We create synthetic observations from a "true" model with known parameters.

# %%
true_params = {"a": -0.3, "b": 3.0, "c": 1.5}


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Evaluate a*x² + b*x + c."""
    return a * x**2 + b * x + c


x_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
uncertainty = 0.1
y_obs = quadratic(x_obs, **true_params) + np.random.normal(
    0, uncertainty, size=len(x_obs)
)

# %% [markdown]
# ### Parameter Priors
#
# The prior distribution says what we know about the parameters before seeing data.
# This might be an initial guess or a physically plausible range.

# %%
parameters = ParameterSet()
parameters.add("a", Uniform(-2.0, 2.0))
parameters.add("b", Uniform(0.0, 5.0))
parameters.add("c", Uniform(0.0, 5.0))

parameters.parameter_names

# %% [markdown]
# Uniform priors are the simplest choice.
# `Normal`, `LogNormal` and `Bound` express more informative priors.
# A `ParameterSet` can also be built from a dict.
# This example is only for illustration and is not used below.

# %%
informative_params = ParameterSet(
    {
        "sensitivity": LogNormal(mean=3.0, std_dev=1.0),
        "offset": Normal(0.0, 0.5),
        "deep_ocean_efficacy": Bound(Normal(1.0, 0.3), 0.5, 2.0),
    }
)

print(f"Bounds: {informative_params.bounds()}")
print("Random samples:")
print(informative_params.sample_random(4))
print("Latin hypercube samples:")
print(informative_params.sample_lhs(4))

# %% [markdown]
# ### Observation Target
#
# The target holds the observations the model should match, each with its uncertainty.

# %%
target = Target()
for x, y in zip(x_obs, y_obs):
    target.add_observation("y", float(x), float(y), uncertainty)


# %% [markdown]
# ### Model Runner
#
# The model factory takes a dict of parameter values and returns the model outputs
# as `{variable: {time: value}}`, with one entry per observation.
# `ModelRunner` wraps it for the estimators.


# %%
def model_factory(param_dict):
    """Run the quadratic model at the observation points."""
    a, b, c = param_dict["a"], param_dict["b"], param_dict["c"]
    return {"y": {float(x): float(quadratic(x, a, b, c)) for x in x_obs}}


runner = ModelRunner(
    model_factory=model_factory,
    parameter_names=["a", "b", "c"],
    output_variables=["y"],
)

# %% [markdown]
# ### Point Estimation
#
# Random search over the prior finds the parameter set that best fits the observations.
# 1000 samples are enough for three parameters.

# %%
likelihood = GaussianLikelihood()
estimator = PointEstimator(parameters, runner, likelihood, target)
result = estimator.optimise(Optimiser.random_search(), n_samples=1000)

best_parameters = dict(zip(estimator.parameter_names, result.best_parameters))
print(f"Best log likelihood: {result.best_log_likelihood:.3f}")
pd.DataFrame({"best": best_parameters, "true": true_params}).round(3)

# %% tags=["remove_input"]
x_plot = np.linspace(0, 6, 100)
plt.figure(figsize=(8, 5))
plt.errorbar(x_obs, y_obs, yerr=uncertainty, fmt="o", label="Observations", capsize=5)
plt.plot(x_plot, quadratic(x_plot, **true_params), "--", label="True model", alpha=0.7)
plt.plot(x_plot, quadratic(x_plot, **best_parameters), "-", label="Best fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Point Estimation: Best Fit vs Observations")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### MCMC Sampling
#
# Point estimation gives the best fit, but not how certain it is.
# MCMC sampling explores the full posterior distribution.
# The progress callback prints a line every 500 iterations.

# %%
sampler = EnsembleSampler(parameters, runner, likelihood, target)
chain = sampler.run_with_progress(
    n_iterations=1000,
    init=WalkerInit.from_prior(),
    thin=1,
    progress_callback=progress.create_simple_callback(print_every=500),
)

print(f"{sampler.default_n_walkers()} walkers, {chain.n_iterations} iterations")

# %% [markdown]
# ### Convergence Diagnostics
#
# Before trusting the results, check that the chain has converged.
# The first 200 iterations are discarded as burn-in.
# R-hat should be close to 1, and `is_converged` checks it against a threshold (1.1 by default).
# The effective sample size (ESS) and autocorrelation time show how many independent samples the chain holds.

# %%
discard = 200

print(f"Converged: {chain.is_converged(discard)}")
pd.DataFrame(
    {
        "r_hat": chain.r_hat(discard),
        "ess": chain.ess(discard),
        "autocorr_time": chain.autocorr_time(discard),
    }
).round(2)

# %% [markdown]
# ### Posterior Distribution
#
# `to_parameter_dict` returns the samples after burn-in as one array per parameter.
# The hidden `posterior_table` helper summarises each one.

# %%
samples = chain.to_parameter_dict(discard)
posterior_table(samples, true_params)

# %% tags=["remove_input"]
plot_posteriors(samples, true_params, trace=True)

# %% [markdown]
# The joint posteriors show how the parameters trade off against each other.

# %% tags=["remove_input"]
pairs = [("a", "b"), ("a", "c"), ("b", "c")]
fig, axes = plt.subplots(1, len(pairs), figsize=(16, 5))
for ax, (x_name, y_name) in zip(axes, pairs):
    h = ax.hist2d(samples[x_name], samples[y_name], bins=30, cmap="Blues")
    ax.plot(
        true_params[x_name],
        true_params[y_name],
        "r*",
        markersize=20,
        markeredgecolor="white",
        label="True values",
    )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"Joint Posterior: {x_name} vs {y_name}")
    ax.legend()
plt.colorbar(h[3], ax=axes[-1], label="Sample density")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 2: Two-Layer Climate Model
#
# Now we calibrate a climate model: the two-layer energy balance model.
# It simulates surface and deep ocean temperatures in response to radiative forcing.
#
# ### Running the Model
#
# `run_two_layer` builds the component from a full parameter dict, applies a constant 3 W/m² forcing
# from 2000 to 2010 and returns the surface temperature in 2010.
# The forcing collection also holds the starting temperatures, so it is built once and reused.

# %%
from rscm.core import InterpolationStrategy, TimeAxis, Timeseries, TimeseriesCollection
from rscm.two_layer import TwoLayerBuilder

time_axis = TimeAxis.from_values(np.array([2000.0, 2010.0]))
forcing = TimeseriesCollection()
for name, value, unit in [
    ("Effective Radiative Forcing", 3.0, "W/m^2"),
    ("Surface Temperature", 0.0, "K"),
    ("Deep Ocean Temperature", 0.0, "K"),
]:
    forcing.add_timeseries(
        name,
        Timeseries(
            np.array([value, value]), time_axis, unit, InterpolationStrategy.Previous
        ),
    )


def run_two_layer(parameters: dict) -> float:
    """Return the 2010 surface temperature for a full parameter set."""
    component = TwoLayerBuilder.from_parameters(parameters).build()
    result = component.solve(2000, 2010, forcing)
    return float(result.get("Surface Temperature").as_scalar())


# %% [markdown]
# ### Synthetic Temperature Observation
#
# The "true" model gives one observation of the 2010 temperature, with added noise.

# %%
true_climate_params = {
    "lambda0": 1.1,  # W/(m² K)
    "deep_ocean_efficacy": 1.3,
    "a": 0.05,  # W/(m² K²)
    "ocean_heat_transfer": 0.7,  # W/(m² K)
    "heat_capacity_deep": 100.0,  # W yr/(m² K)
    "heat_capacity_surface": 8.0,  # W yr/(m² K)
}

true_temp = run_two_layer(true_climate_params)
obs_temp = true_temp + np.random.normal(0, 0.05)
obs_uncertainty = 0.1  # K

print(f"True: {true_temp:.3f} K, observed: {obs_temp:.3f} ± {obs_uncertainty} K")

# %% [markdown]
# ### Calibration Setup
#
# We calibrate two parameters: `lambda0` (climate feedback) and `a` (nonlinear feedback).
# The others stay fixed at their true values,
# so the model factory merges the sampled values into `fixed_climate_params`.

# %%
climate_params = ParameterSet()
climate_params.add("lambda0", Uniform(0.8, 1.5))
climate_params.add("a", Uniform(0.0, 0.1))

climate_target = Target()
climate_target.add_observation("Surface Temperature", 2010.0, obs_temp, obs_uncertainty)

fixed_climate_params = {
    name: value
    for name, value in true_climate_params.items()
    if name not in climate_params.parameter_names
}


def climate_model_factory(param_values):
    """Run the two-layer model with the sampled parameters."""
    temperature = run_two_layer({**fixed_climate_params, **param_values})
    return {"Surface Temperature": {2010.0: temperature}}


climate_runner = ModelRunner(
    model_factory=climate_model_factory,
    parameter_names=climate_params.parameter_names,
    output_variables=["Surface Temperature"],
)

# %% [markdown]
# ### Climate Point Estimate

# %%
climate_estimator = PointEstimator(
    climate_params, climate_runner, GaussianLikelihood(), climate_target
)
climate_result = climate_estimator.optimise(Optimiser.random_search(), n_samples=1000)

best_climate_params = dict(
    zip(climate_estimator.parameter_names, climate_result.best_parameters)
)
print(f"Best log likelihood: {climate_result.best_log_likelihood:.3f}")
pd.DataFrame(
    {
        "best": best_climate_params,
        "true": {name: true_climate_params[name] for name in best_climate_params},
    }
).round(4)

# %% [markdown]
# A single temperature observation cannot separate `lambda0` from `a`.
# Many combinations give the same warming, so the best fit need not match the true values.
# The posterior below stays close to the prior as a result.
# More observations, such as a full temperature record or ocean heat content, would constrain both.
#
# ### Climate MCMC Sampling
#
# The walkers start near the point estimate rather than across the whole prior.
# `WalkerInit.ball(centre, radius)` spreads them uniformly over a box of width `radius` around `centre`.
# `WalkerInit.explicit(positions)` takes one row of parameter values per walker instead.

# %%
climate_sampler = EnsembleSampler(
    climate_params, climate_runner, GaussianLikelihood(), climate_target
)
climate_chain = climate_sampler.run_with_progress(
    n_iterations=1000,
    init=WalkerInit.ball(climate_result.best_parameters, 0.01),
    thin=1,
    progress_callback=progress.create_simple_callback(print_every=500),
)

print(f"Converged: {climate_chain.is_converged(discard)}")
pd.DataFrame(
    {"r_hat": climate_chain.r_hat(discard), "ess": climate_chain.ess(discard)}
).round(2)

# %% [markdown]
# ### Climate Posterior

# %%
climate_samples = climate_chain.to_parameter_dict(discard)
posterior_table(climate_samples, true_climate_params)

# %% tags=["remove_input"]
plot_posteriors(climate_samples, true_climate_params)

# %% [markdown]
# ## Part 3: Long Runs and DataFrames
#
# Long-running calibrations need:
#
# - **Checkpointing**: Save progress and resume if interrupted.
# - **Thinning**: Reduce memory usage for long chains.
# - **DataFrame integration**: Export to pandas for analysis.
#
# These examples reuse the quadratic `sampler` from Part 1.
#
# ### Checkpoint and Resume
#
# `run_with_checkpoint` writes `<path>.chain` and `<path>.state` every `checkpoint_every` iterations.

# %%
import os
import tempfile

temp_dir = tempfile.mkdtemp()
checkpoint_path = os.path.join(temp_dir, "chain_checkpoint")

chain_1 = sampler.run_with_checkpoint(
    n_iterations=500,
    init=WalkerInit.from_prior(),
    thin=1,
    checkpoint_every=100,
    checkpoint_path=checkpoint_path,
)
print(f"First run: {chain_1.n_iterations} iterations")

# %% [markdown]
# `resume_from_checkpoint` continues an interrupted run.
# Its `n_iterations` is the total, not the number of extra iterations,
# and the returned chain holds both the original and the new samples.

# %%
chain_2 = sampler.resume_from_checkpoint(
    n_iterations=1000,
    checkpoint_path=checkpoint_path,
    checkpoint_every=100,
    thin=1,
)
print(f"Resumed run: {chain_2.n_iterations} iterations")

# %% [markdown]
# ### Thinning
#
# `thin=10` stores only every tenth sample.

# %%
chain_thinned = sampler.run(
    n_iterations=1000,
    init=WalkerInit.from_prior(),
    thin=10,
)

print(f"thin=1:  {chain.flat_samples(discard=0).shape[0]:,} samples")
print(f"thin=10: {chain_thinned.flat_samples(discard=0).shape[0]:,} samples")

# %% [markdown]
# ### Saving, Loading and Merging Chains
#
# `merge` appends another chain's samples in place, for example from parallel runs.

# %%
from rscm.calibrate import Chain

chain_file = os.path.join(temp_dir, "saved_chain.bin")
chain.save(chain_file)
loaded_chain = Chain.load(chain_file)
print(
    f"Loaded: {loaded_chain.n_iterations} iterations of {loaded_chain.parameter_names}"
)

merged_chain = Chain.load(chain_file)
merged_chain.merge(loaded_chain)
print(f"Merged: {merged_chain.n_iterations} iterations")

# %% [markdown]
# ### DataFrame Integration
#
# `to_dataframe` returns one row per stored sample, with a column per parameter and `log_prob`.
# It needs pandas.

# %%
chain.to_dataframe(discard=200).head()

# %% [markdown]
# Observations can also come from a DataFrame.
# `Target.from_dataframe` needs a `variable` column, even for a single variable.

# %%
obs_df = pd.DataFrame(
    {
        "variable": "y",
        "time": x_obs,
        "value": y_obs,
        "uncertainty": uncertainty,
    }
)
df_target = Target.from_dataframe(obs_df)

print(f"Variables: {df_target.variable_names()}")
print(f"Observations: {df_target.n_observations()}")

# %% [markdown]
# ### Relative Uncertainties and Reference Periods
#
# `add_observation_relative` sets each uncertainty as a fraction of the observed value.
# `set_reference_period` records that a variable's observations are anomalies relative to a baseline.
#
# The likelihood does not apply the reference period yet.
# It compares model outputs with the observations as given,
# so the model factory must return anomalies over the same baseline.

# %%
anomaly_target = Target()
for year, anomaly in [(1950.0, 0.3), (1980.0, 0.5), (2010.0, 1.0)]:
    anomaly_target.add_observation_relative("Surface Temperature", year, anomaly, 0.1)
anomaly_target.set_reference_period("Surface Temperature", 1850.0, 1900.0)

surface = anomaly_target.get_variable("Surface Temperature")
print(f"Reference period: {surface.reference_period}")
for obs in surface.observations:
    print(f"  {obs.time:.0f}: {obs.value:.2f} ± {obs.uncertainty:.3f} K")

# %% [markdown]
# ### Progress Tracking
#
# A `ProgressTracker` passed as the progress callback records the acceptance rate
# and mean log probability at each iteration.

# %%
tracker = progress.ProgressTracker()
tracked_chain = sampler.run_with_progress(
    n_iterations=500,
    init=WalkerInit.from_prior(),
    thin=1,
    progress_callback=tracker,
)

print(f"Final acceptance rate: {tracker.acceptance_rates[-1]:.2%}")

# %% [markdown]
# The first iterations start from the prior and have very low log probabilities,
# so the plot starts at iteration 50.

# %% tags=["remove_input"]
start = 50
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(tracker.iterations[start:], tracker.acceptance_rates[start:])
axes[0].set_ylabel("Acceptance Rate")
axes[0].set_title("Acceptance Rate vs Iteration")
axes[1].plot(tracker.iterations[start:], tracker.mean_log_probs[start:])
axes[1].set_ylabel("Mean Log Probability")
axes[1].set_title("Mean Log Probability vs Iteration")
for ax in axes:
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% tags=["remove_input"]
import shutil

shutil.rmtree(temp_dir)

# %% [markdown]
# ## Summary
#
# This tutorial covered:
#
# **Part 1:**
#
# - Defining parameter priors with distributions (`Uniform`, `Normal`, `LogNormal`, `Bound`).
# - Creating observation targets.
# - Setting up a model runner.
# - Point estimation to find best-fit parameters.
# - MCMC sampling to quantify uncertainty.
# - Convergence diagnostics (R-hat, ESS, autocorrelation time).
#
# **Part 2:**
#
# - Calibrating a climate model (two-layer energy balance).
# - Warm-starting MCMC walkers near the point estimate.
# - Interpreting posterior distributions.
#
# **Part 3:**
#
# - Checkpoint and resume for long-running calibrations.
# - Thinning to reduce memory usage.
# - Chain persistence (save, load and merge).
# - DataFrame integration for chains and observation targets.
# - Progress tracking with callbacks (`progress.create_tqdm_callback` gives a progress bar).
#
# ## Next Steps
#
# - Try calibrating with multiple observation variables.
# - Tune the proposal with `EnsembleSampler.with_stretch_parameter`.
# - Use `GaussianLikelihood(normalise=True)` when comparing models by likelihood.
# - Explore posterior predictive checks to validate model fit.
#
# ## Further Reading
#
# - **Algorithm**: Goodman and Weare (2010),
#   [Ensemble samplers with affine invariance](https://doi.org/10.2140/camcos.2010.5.65).
# - **API reference**: [`rscm.calibrate`](../../api/rscm/calibrate/).
# - **Convergence diagnostics**: Gelman et al. (2013), "Bayesian Data Analysis", Chapter 11.

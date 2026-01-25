"""
Simple integration test for the calibration framework using a mock model.

This test validates the complete calibration workflow with a simple analytical
model that doesn't require complex component setup.
"""

import numpy as np
import pytest

from rscm.calibrate import (
    EnsembleSampler,
    GaussianLikelihood,
    ModelRunner,
    ParameterSet,
    PointEstimator,
    Target,
    Uniform,
    WalkerInit,
)


def test_point_estimation_simple():
    """Test point estimation with a simple linear model."""
    # True parameters
    true_a = 2.0
    true_b = 1.5

    # Generate synthetic observations: y = a*x + b + noise
    rng = np.random.default_rng(42)
    x_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = true_a * x_obs + true_b
    y_obs = y_true + rng.normal(0, 0.1, size=len(x_obs))
    uncertainty = 0.1

    # Define parameter priors
    params = ParameterSet()
    params.add("a", Uniform(0.0, 5.0))
    params.add("b", Uniform(0.0, 5.0))

    # Create target observations
    target = Target()
    for x, y in zip(x_obs, y_obs):
        target.add_observation("y", float(x), float(y), uncertainty)

    # Create model runner
    def model_factory(param_dict):
        a = param_dict["a"]
        b = param_dict["b"]
        # Return model outputs for each observation time
        return {"y": {float(x): float(a * x + b) for x in x_obs}}

    runner = ModelRunner(
        model_factory=model_factory,
        param_names=["a", "b"],
        output_variables=["y"],
    )

    # Create likelihood
    likelihood = GaussianLikelihood()

    # Create point estimator
    estimator = PointEstimator(params, runner, likelihood, target)

    # Run random search
    from rscm.calibrate import Optimizer  # noqa: PLC0415

    result = estimator.optimize(Optimizer.random_search(), n_samples=1000)

    # Check results
    best_params_vec = result.best_params
    param_names = estimator.param_names
    best_params = dict(zip(param_names, best_params_vec))

    assert "a" in best_params
    assert "b" in best_params

    # Parameters should be within prior bounds
    assert 0.0 <= best_params["a"] <= 5.0
    assert 0.0 <= best_params["b"] <= 5.0

    # Should have explored multiple points
    assert estimator.n_evaluations == 1000

    # Log likelihood should be finite
    assert np.isfinite(result.best_log_likelihood)

    # With enough samples, should be reasonably close to true values
    # (within ~1.0 for this simple problem with random search)
    assert abs(best_params["a"] - true_a) < 1.0
    assert abs(best_params["b"] - true_b) < 1.0

    print("\nPoint estimation results:")
    print(f"  Best parameters: a={best_params['a']:.3f}, b={best_params['b']:.3f}")
    print(f"  True parameters: a={true_a}, b={true_b}")
    print(f"  Best log likelihood: {result.best_log_likelihood:.3f}")
    print(f"  Evaluations: {estimator.n_evaluations}")


def test_mcmc_sampling_simple():
    """Test MCMC sampling with a simple linear model."""
    # True parameters
    true_a = 2.0
    true_b = 1.5

    # Generate synthetic observations
    rng = np.random.default_rng(42)
    x_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = true_a * x_obs + true_b
    y_obs = y_true + rng.normal(0, 0.1, size=len(x_obs))
    uncertainty = 0.1

    # Define parameter priors
    params = ParameterSet()
    params.add("a", Uniform(0.0, 5.0))
    params.add("b", Uniform(0.0, 5.0))

    # Create target
    target = Target()
    for x, y in zip(x_obs, y_obs):
        target.add_observation("y", float(x), float(y), uncertainty)

    # Create model runner
    def model_factory(param_dict):
        a = param_dict["a"]
        b = param_dict["b"]
        return {"y": {float(x): float(a * x + b) for x in x_obs}}

    runner = ModelRunner(
        model_factory=model_factory,
        param_names=["a", "b"],
        output_variables=["y"],
    )

    # Create likelihood
    likelihood = GaussianLikelihood()

    # Create ensemble sampler
    sampler = EnsembleSampler(params, runner, likelihood, target)

    # Run MCMC
    n_steps = 200
    chain = sampler.run(n_steps, WalkerInit.from_prior(), thin=1)

    # Check chain structure
    assert chain.total_iterations == n_steps
    assert len(chain.param_names) == 2
    assert "a" in chain.param_names
    assert "b" in chain.param_names

    # Check we have samples
    discard = 100  # Discard first half as burn-in
    samples = chain.flat_samples(discard=discard)
    assert samples.shape[1] == 2

    # Check parameter ranges (should be within prior bounds)
    a_samples = samples[:, 0]
    b_samples = samples[:, 1]
    assert np.all((a_samples >= 0.0) & (a_samples <= 5.0))
    assert np.all((b_samples >= 0.0) & (b_samples <= 5.0))

    # Compute diagnostics
    r_hat = chain.r_hat(discard=discard)
    ess = chain.ess(discard=discard)

    # With a simple problem, should converge reasonably well
    # (but we won't enforce strict convergence for a short run)
    assert len(r_hat) == 2
    assert len(ess) == 2

    # Mean should be close to true values (within a few sigma)
    mean_a = np.mean(a_samples)
    mean_b = np.mean(b_samples)
    assert abs(mean_a - true_a) < 0.5  # Should be quite close
    assert abs(mean_b - true_b) < 0.5

    print("\nMCMC results:")
    print(f"  Chain length: {len(chain)} samples")
    print(f"  R-hat: {r_hat}")
    print(f"  ESS: {ess}")
    print(f"  Mean a: {mean_a:.3f} (true: {true_a})")
    print(f"  Mean b: {mean_b:.3f} (true: {true_b})")


def test_chain_diagnostics():
    """Test that chain diagnostic methods work correctly."""
    # Simple model - no random seed needed, using fixed observations
    x_obs = np.array([1.0, 2.0, 3.0])
    y_obs = np.array([2.5, 4.0, 5.5])

    params = ParameterSet()
    params.add("a", Uniform(0.0, 5.0))
    params.add("b", Uniform(0.0, 5.0))

    target = Target()
    for x, y in zip(x_obs, y_obs):
        target.add_observation("y", float(x), float(y), 0.1)

    def model_factory(param_dict):
        a = param_dict["a"]
        b = param_dict["b"]
        return {"y": {float(x): float(a * x + b) for x in x_obs}}

    runner = ModelRunner(
        model_factory=model_factory,
        param_names=["a", "b"],
        output_variables=["y"],
    )

    likelihood = GaussianLikelihood()
    sampler = EnsembleSampler(params, runner, likelihood, target)

    # Run short chain
    chain = sampler.run(100, WalkerInit.from_prior(), thin=2)

    # Test to_param_dict
    param_dict = chain.to_param_dict(discard=20)
    assert "a" in param_dict
    assert "b" in param_dict
    assert len(param_dict["a"]) > 0
    assert len(param_dict["b"]) > 0

    # Test ESS
    ess = chain.ess(discard=20)
    assert "a" in ess
    assert "b" in ess
    assert all(v > 0 for v in ess.values())

    # Test autocorr_time
    tau = chain.autocorr_time(discard=20)
    assert "a" in tau
    assert "b" in tau
    assert all(v > 0 for v in tau.values())

    # Test convergence check
    is_converged = chain.is_converged(discard=20, threshold=1.2)
    assert isinstance(is_converged, bool)

    print("\nChain diagnostics:")
    print(f"  Thin: {chain.thin}")
    print(f"  ESS: {ess}")
    print(f"  Autocorrelation time: {tau}")
    print(f"  Converged (R-hat < 1.2): {is_converged}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

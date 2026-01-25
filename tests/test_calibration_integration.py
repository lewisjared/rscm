"""
Integration test for calibrating a two-layer climate model.

This test demonstrates a complete calibration workflow:
1. Generate synthetic observations from a "true" model
2. Define parameter priors and target observations
3. Perform point estimation to find best-fit parameters
4. Run MCMC to sample the posterior distribution
5. Check convergence diagnostics
"""

import numpy as np
import pytest

from rscm._lib.core import (
    InterpolationStrategy,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)
from rscm._lib.two_layer import TwoLayerBuilder
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


@pytest.fixture
def synthetic_observations():
    """Generate synthetic temperature observations from a "true" model."""
    # True parameter values (physically realistic)
    true_params = {
        "lambda0": 1.1,
        "efficacy": 1.3,
        "a": 0.05,
        "eta": 0.7,
        "heat_capacity_deep": 100.0,
        "heat_capacity_surface": 8.0,
    }

    # Create true model and run a single timestep
    component = TwoLayerBuilder.from_parameters(true_params).build()

    # Simple forcing scenario
    collection = TimeseriesCollection()
    collection.add_timeseries(
        "Effective Radiative Forcing",
        Timeseries(
            np.array([3.0]),
            TimeAxis.from_bounds(np.array([2000.0, 2010.0])),
            "W/m^2",
            InterpolationStrategy.Previous,
        ),
    )

    result = component.solve(2000, 2010, collection)
    temp = result.get("Surface Temperature").as_scalar()

    # Return a single observation with uncertainty
    return {
        "times": [2010.0],
        "temperatures": [temp],
        "uncertainty": 0.1,  # 0.1 K uncertainty
        "true_params": true_params,
    }


@pytest.fixture
def calibration_setup(synthetic_observations):
    """Set up parameter priors and target observations."""
    # Define parameter priors (wide enough to include true values)
    params = ParameterSet()
    params.add(
        "lambda0", Uniform(0.8, 1.5)
    )  # Climate feedback parameter: typical range
    params.add("a", Uniform(0.0, 0.1))  # Nonlinear feedback: typical range

    # For this test, fix other parameters at their true values
    # In a real calibration, you might calibrate more parameters

    # Create target from synthetic observations
    target = Target()
    for t, temp in zip(
        synthetic_observations["times"],
        synthetic_observations["temperatures"],
    ):
        target.add_observation(
            "Surface Temperature", t, temp, synthetic_observations["uncertainty"]
        )

    return {
        "params": params,
        "target": target,
        "true_params": synthetic_observations["true_params"],
    }


def create_model_runner(fixed_params):
    """Create a ModelRunner for the two-layer model."""

    def model_factory(param_values):
        """
        Create and run a two-layer model.

        Parameters
        ----------
        param_values : dict
            Dictionary mapping parameter names to values

        Returns
        -------
        dict
            Model outputs as {variable_name: {time: value}}
        """
        # Combine calibrated and fixed parameters
        all_params = fixed_params.copy()
        all_params.update(param_values)

        # Create component
        component = TwoLayerBuilder.from_parameters(all_params).build()

        # Run model with simple forcing scenario (single timestep)
        collection = TimeseriesCollection()
        collection.add_timeseries(
            "Effective Radiative Forcing",
            Timeseries(
                np.array([3.0]),
                TimeAxis.from_bounds(np.array([2000.0, 2010.0])),
                "W/m^2",
                InterpolationStrategy.Previous,
            ),
        )

        result = component.solve(2000, 2010, collection)
        temp = result.get("Surface Temperature").as_scalar()

        # Return output
        return {"Surface Temperature": {2010.0: float(temp)}}

    # Get parameter names from the factory
    return ModelRunner(
        model_factory=model_factory,
        param_names=["lambda0", "a"],
        output_variables=["Surface Temperature"],
    )


def test_point_estimation(calibration_setup):
    """Test point estimation finds reasonable parameters."""
    params = calibration_setup["params"]
    target = calibration_setup["target"]
    true_params = calibration_setup["true_params"]

    # Create fixed parameters
    fixed_params = {
        "efficacy": true_params["efficacy"],
        "eta": true_params["eta"],
        "heat_capacity_deep": true_params["heat_capacity_deep"],
        "heat_capacity_surface": true_params["heat_capacity_surface"],
    }

    # Create model runner
    runner = create_model_runner(fixed_params)

    # Create likelihood
    likelihood = GaussianLikelihood()

    # Create point estimator
    estimator = PointEstimator(params, runner, likelihood, target)

    # Run random search (simple baseline)
    from rscm.calibrate import Optimizer  # noqa: PLC0415

    result = estimator.optimize(Optimizer.random_search(), n_samples=50)

    # Check that we found reasonable parameters
    best_params_vec = result.best_params
    param_names = estimator.param_names
    best_params = dict(zip(param_names, best_params_vec))

    assert "lambda0" in best_params
    assert "a" in best_params

    # Parameters should be within prior bounds
    assert 0.8 <= best_params["lambda0"] <= 1.5
    assert 0.0 <= best_params["a"] <= 0.1

    # Should have explored multiple points
    assert estimator.n_evaluations == 50

    # Log likelihood should be finite (model succeeded)
    assert np.isfinite(result.best_log_likelihood)

    print("\nPoint estimation results:")
    print(f"  Best parameters: {best_params}")
    print(f"  True parameters: lambda0={true_params['lambda0']}, a={true_params['a']}")
    print(f"  Best log likelihood: {result.best_log_likelihood:.3f}")
    print(f"  Evaluations: {estimator.n_evaluations}")


def test_mcmc_sampling(calibration_setup):
    """Test MCMC sampling converges to reasonable posterior."""
    params = calibration_setup["params"]
    target = calibration_setup["target"]
    true_params = calibration_setup["true_params"]

    # Create fixed parameters
    fixed_params = {
        "efficacy": true_params["efficacy"],
        "eta": true_params["eta"],
        "heat_capacity_deep": true_params["heat_capacity_deep"],
        "heat_capacity_surface": true_params["heat_capacity_surface"],
    }

    # Create model runner
    runner = create_model_runner(fixed_params)

    # Create likelihood
    likelihood = GaussianLikelihood()

    # Create ensemble sampler
    sampler = EnsembleSampler(params, runner, likelihood, target)

    # Run MCMC (short run for testing)
    n_steps = 100
    thin = 1
    chain = sampler.run(n_steps, WalkerInit.from_prior(), thin=thin)

    # Check chain structure
    assert chain.total_iterations == n_steps
    assert chain.thin == thin
    assert len(chain.param_names) == 2
    assert "lambda0" in chain.param_names
    assert "a" in chain.param_names

    # Check we have samples
    samples = chain.flat_samples(discard=50)  # Discard first 50 as burn-in
    assert samples.shape[1] == 2  # Two parameters

    # Check parameter ranges (should be within prior bounds)
    lambda0_samples = samples[:, 0]
    a_samples = samples[:, 1]
    assert np.all((lambda0_samples >= 0.8) & (lambda0_samples <= 1.5))
    assert np.all((a_samples >= 0.0) & (a_samples <= 0.1))

    # Compute diagnostics
    r_hat = chain.r_hat(discard=50)
    print("\nMCMC results:")
    print(f"  Chain length: {len(chain)} samples")
    print(f"  Total iterations: {chain.total_iterations}")
    print(f"  R-hat: {r_hat}")
    mean_l0 = np.mean(lambda0_samples)
    print(f"  Mean lambda0: {mean_l0:.3f} (true: {true_params['lambda0']})")
    print(f"  Mean a: {np.mean(a_samples):.3f} (true: {true_params['a']})")

    # For a short run, we don't expect perfect convergence, but check basics
    assert len(r_hat) == 2  # Should have R-hat for both parameters


def test_chain_diagnostics(calibration_setup):
    """Test chain diagnostic methods work correctly."""
    params = calibration_setup["params"]
    target = calibration_setup["target"]
    true_params = calibration_setup["true_params"]

    # Create fixed parameters
    fixed_params = {
        "efficacy": true_params["efficacy"],
        "eta": true_params["eta"],
        "heat_capacity_deep": true_params["heat_capacity_deep"],
        "heat_capacity_surface": true_params["heat_capacity_surface"],
    }

    # Create model runner
    runner = create_model_runner(fixed_params)
    likelihood = GaussianLikelihood()
    sampler = EnsembleSampler(params, runner, likelihood, target)

    # Run short chain
    chain = sampler.run(50, WalkerInit.from_prior(), thin=1)

    # Test to_param_dict
    param_dict = chain.to_param_dict(discard=10)
    assert "lambda0" in param_dict
    assert "a" in param_dict
    assert len(param_dict["lambda0"]) > 0

    # Test ESS
    ess = chain.ess(discard=10)
    assert "lambda0" in ess
    assert "a" in ess
    assert all(v > 0 for v in ess.values())

    # Test autocorr_time
    tau = chain.autocorr_time(discard=10)
    assert "lambda0" in tau
    assert "a" in tau
    assert all(v > 0 for v in tau.values())

    print("\nChain diagnostics:")
    print(f"  ESS: {ess}")
    print(f"  Autocorrelation time: {tau}")


if __name__ == "__main__":
    # Allow running this test standalone
    pytest.main([__file__, "-v", "-s"])

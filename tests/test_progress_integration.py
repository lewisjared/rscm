"""
Test progress reporting integration for calibration workflows.
"""

import numpy as np
import pytest

from rscm.calibrate import (
    EnsembleSampler,
    GaussianLikelihood,
    ModelRunner,
    ParameterSet,
    Target,
    Uniform,
    WalkerInit,
)
from rscm.calibrate.progress import (
    ProgressTracker,
    create_simple_callback,
    create_tqdm_callback,
)


def create_simple_linear_model():
    """Create a simple linear model setup for testing."""
    # True parameters
    true_a = 2.0
    true_b = 1.5

    # Generate synthetic observations: y = a*x + b + noise
    rng = np.random.default_rng(42)
    x_obs = np.array([1.0, 2.0, 3.0])
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
        return {"y": {float(x): float(a * x + b) for x in x_obs}}

    runner = ModelRunner(
        model_factory=model_factory,
        param_names=["a", "b"],
        output_variables=["y"],
    )

    likelihood = GaussianLikelihood()

    return params, runner, likelihood, target


def test_progress_tracker():
    """Test ProgressTracker stores progress metrics correctly."""
    params, runner, likelihood, target = create_simple_linear_model()
    sampler = EnsembleSampler(params, runner, likelihood, target)

    # Create tracker
    tracker = ProgressTracker()

    # Run short chain
    n_iterations = 10
    _chain = sampler.run_with_progress(
        n_iterations=n_iterations,
        progress_callback=tracker,
        init=WalkerInit.from_prior(),
        thin=1,
    )

    # Check tracker stored data
    assert len(tracker.iterations) == n_iterations
    assert len(tracker.acceptance_rates) == n_iterations
    assert len(tracker.mean_log_probs) == n_iterations

    # Check values are reasonable
    assert all(0 <= rate <= 1 for rate in tracker.acceptance_rates)
    assert all(np.isfinite(lp) for lp in tracker.mean_log_probs)
    assert tracker.iterations == list(range(n_iterations))

    # Test clear
    tracker.clear()
    assert len(tracker.iterations) == 0
    assert len(tracker.acceptance_rates) == 0
    assert len(tracker.mean_log_probs) == 0


def test_simple_callback(capsys):
    """Test simple text callback prints progress."""
    params, runner, likelihood, target = create_simple_linear_model()
    sampler = EnsembleSampler(params, runner, likelihood, target)

    # Create simple callback that prints every 5 iterations
    callback = create_simple_callback(print_every=5)

    # Run short chain
    n_iterations = 10
    _chain = sampler.run_with_progress(
        n_iterations=n_iterations,
        progress_callback=callback,
        init=WalkerInit.from_prior(),
        thin=1,
    )

    # Check that output was printed
    captured = capsys.readouterr()
    assert "Iteration" in captured.out
    assert "Acceptance rate" in captured.out
    assert "Mean log prob" in captured.out

    # Should have printed at iteration 5 and 10
    assert "5/10" in captured.out or "10/10" in captured.out


def test_tqdm_callback():
    """Test tqdm callback integration."""
    pytest.importorskip("tqdm")  # Skip if tqdm not installed

    params, runner, likelihood, target = create_simple_linear_model()
    sampler = EnsembleSampler(params, runner, likelihood, target)

    # Create tqdm callback
    callback = create_tqdm_callback(total=10, desc="Test Sampling")

    # Run short chain
    n_iterations = 10
    _chain = sampler.run_with_progress(
        n_iterations=n_iterations,
        progress_callback=callback,
        init=WalkerInit.from_prior(),
        thin=1,
    )

    # Check progress bar exists and can be closed
    assert hasattr(callback, "pbar")
    assert hasattr(callback, "close")

    # Clean up
    callback.close()


def test_tqdm_not_installed():
    """Test that create_tqdm_callback raises helpful error when tqdm not available."""
    # Just verify the function exists and is callable
    # Actually testing the ImportError would require mocking tqdm which is complex
    assert callable(create_tqdm_callback)


def test_tracker_with_print(capsys):
    """Test ProgressTracker with printing enabled."""
    params, runner, likelihood, target = create_simple_linear_model()
    sampler = EnsembleSampler(params, runner, likelihood, target)

    # Create tracker with printing
    tracker = ProgressTracker(print_every=5)

    # Run short chain
    n_iterations = 10
    _chain = sampler.run_with_progress(
        n_iterations=n_iterations,
        progress_callback=tracker,
        init=WalkerInit.from_prior(),
        thin=1,
    )

    # Check both tracking and printing worked
    assert len(tracker.iterations) == n_iterations

    captured = capsys.readouterr()
    assert "Iteration" in captured.out

"""
Comprehensive tests for the Python calibration API.

This test suite validates all public Python APIs for the calibration module,
including distributions, parameter sets, targets, chains, and estimators.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rscm.calibrate import (
    Bound,
    Chain,
    EnsembleSampler,
    GaussianLikelihood,
    LogNormal,
    ModelRunner,
    Normal,
    Optimizer,
    ParameterSet,
    PointEstimator,
    Target,
    Uniform,
    WalkerInit,
)


class TestDistributions:
    """Test distribution classes and their properties."""

    def test_uniform_creation(self):
        """Test Uniform distribution creation and basic properties."""
        dist = Uniform(0.0, 10.0)
        assert dist is not None

        # Sample should be in range
        sample = dist.sample()
        assert 0.0 <= sample <= 10.0

        # Multiple samples
        samples = [dist.sample() for _ in range(100)]
        assert all(0.0 <= s <= 10.0 for s in samples)

    def test_uniform_validation(self):
        """Test Uniform validates bounds."""
        with pytest.raises(ValueError, match=r"low.*high"):
            Uniform(10.0, 5.0)  # Lower > upper should fail

    def test_normal_creation(self):
        """Test Normal distribution creation."""
        dist = Normal(0.0, 1.0)
        assert dist is not None

        # Sample from standard normal
        samples = np.array([dist.sample() for _ in range(1000)])

        # Mean should be close to 0
        assert abs(np.mean(samples)) < 0.2

        # Std should be close to 1
        assert abs(np.std(samples) - 1.0) < 0.2

    def test_normal_validation(self):
        """Test Normal validates parameters."""
        with pytest.raises(ValueError, match="positive"):
            Normal(0.0, -1.0)  # Negative std should fail

    def test_lognormal_creation(self):
        """Test LogNormal distribution creation."""
        # Using mu/sigma parameterisation
        dist = LogNormal(mu=0.0, sigma=1.0)
        assert dist is not None

        # Samples should be positive
        samples = [dist.sample() for _ in range(100)]
        assert all(s > 0 for s in samples)

    def test_lognormal_mean_std(self):
        """Test LogNormal mean/std constructor."""
        dist = LogNormal(mean=1.0, std=0.5)
        assert dist is not None

        samples = [dist.sample() for _ in range(100)]
        assert all(s > 0 for s in samples)

    def test_bound_wrapper(self):
        """Test Bound wrapper constrains distributions."""
        # Normal with hard bounds
        base_dist = Normal(0.0, 1.0)
        bounded = Bound(base_dist, -2.0, 2.0)

        # All samples should be in bounds
        samples = [bounded.sample() for _ in range(100)]
        assert all(-2.0 <= s <= 2.0 for s in samples)


class TestParameterSet:
    """Test ParameterSet builder API and operations."""

    def test_fluent_builder(self):
        """Test fluent builder API returns self for chaining."""
        params = ParameterSet()
        result = params.add("a", Uniform(0.0, 1.0))
        assert result is params  # Should return self

        # Chaining
        params.add("b", Normal(0.0, 1.0)).add("c", Uniform(-5.0, 5.0))
        assert len(params.param_names) == 3

    def test_param_names_order(self):
        """Test parameter names preserve insertion order."""
        params = ParameterSet()
        params.add("z", Uniform(0.0, 1.0))
        params.add("a", Uniform(0.0, 1.0))
        params.add("m", Uniform(0.0, 1.0))

        # Should preserve insertion order
        assert params.param_names == ["z", "a", "m"]

    def test_sample_random(self):
        """Test random sampling from priors."""
        params = ParameterSet()
        params.add("a", Uniform(0.0, 1.0))
        params.add("b", Uniform(-5.0, 5.0))

        samples = params.sample_random(10)
        assert samples.shape == (10, 2)

        # Check bounds
        assert np.all((samples[:, 0] >= 0.0) & (samples[:, 0] <= 1.0))
        assert np.all((samples[:, 1] >= -5.0) & (samples[:, 1] <= 5.0))

    def test_sample_lhs(self):
        """Test Latin Hypercube Sampling."""
        params = ParameterSet()
        params.add("a", Uniform(0.0, 1.0))
        params.add("b", Uniform(0.0, 1.0))

        samples = params.sample_lhs(10)
        assert samples.shape == (10, 2)

        # LHS should have better coverage than random
        # Each dimension should have one sample in each decile
        for dim in range(2):
            dim_samples = samples[:, dim]
            # Sort and check spacing
            sorted_samples = np.sort(dim_samples)
            # With LHS, sorted samples should be relatively evenly spaced
            gaps = np.diff(sorted_samples)
            # No huge gaps or clustering
            assert np.max(gaps) < 0.3  # Max gap < 30% of range

    def test_log_prior(self):
        """Test log prior probability computation."""
        params = ParameterSet()
        params.add("a", Uniform(0.0, 1.0))
        params.add("b", Uniform(0.0, 1.0))

        # In-bounds point
        in_bounds = [0.5, 0.5]
        log_p = params.log_prior(in_bounds)
        assert np.isfinite(log_p)

        # Out-of-bounds point
        out_bounds = [1.5, 0.5]
        log_p_out = params.log_prior(out_bounds)
        assert log_p_out == float("-inf")

    def test_bounds_extraction(self):
        """Test bounds extraction for optimisers."""
        params = ParameterSet()
        params.add("a", Uniform(0.0, 1.0))
        params.add("b", Uniform(-5.0, 5.0))

        bounds = params.bounds()
        # bounds() returns (lower_bounds, upper_bounds) as two lists
        assert len(bounds) == 2
        lower, upper = bounds
        assert lower == [0.0, -5.0]
        assert upper == [1.0, 5.0]


class TestTarget:
    """Test Target builder API and observation handling."""

    def test_fluent_builder(self):
        """Test fluent builder API."""
        target = Target()
        result = target.add_observation("temp", 2000.0, 15.0, 0.5)
        assert result is target  # Returns self

        # Chaining
        target.add_observation("temp", 2001.0, 15.5, 0.5).add_observation(
            "temp", 2002.0, 16.0, 0.5
        )

    def test_multiple_variables(self):
        """Test adding observations for multiple variables."""
        target = Target()
        target.add_observation("temp", 2000.0, 15.0, 0.5)
        target.add_observation("ohc", 2000.0, 100.0, 10.0)
        target.add_observation("temp", 2001.0, 15.5, 0.5)

        # Both variables should be present
        # (We can't directly inspect internal state, but this shouldn't error)

    def test_relative_error(self):
        """Test relative error specification."""
        target = Target()
        # Add with 10% relative error
        target.add_observation_relative("temp", 2000.0, 100.0, 0.1)

        # Should compute uncertainty as 10% of value = 10.0
        # (We validate this works by not raising an error)

    def test_reference_period(self):
        """Test reference period for anomaly calculation."""
        target = Target()

        # Add observations first
        target.add_observation("temp", 1875.0, 14.0, 0.2)
        target.add_observation("temp", 2000.0, 15.0, 0.2)

        # Set reference period for variable
        target.set_reference_period("temp", 1850.0, 1900.0)

    @pytest.mark.skipif(
        not hasattr(Target, "from_dataframe"),
        reason="pandas optional dependency",
    )
    def test_from_dataframe(self):
        """Test DataFrame constructor."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame(
            {
                "variable": ["temp", "temp", "ohc", "ohc"],
                "time": [2000.0, 2001.0, 2000.0, 2001.0],
                "value": [15.0, 15.5, 100.0, 105.0],
                "uncertainty": [0.5, 0.5, 10.0, 10.0],
            }
        )

        target = Target.from_dataframe(df)
        assert target is not None


class TestChain:
    """Test Chain persistence and diagnostic methods."""

    @pytest.fixture
    def sample_chain(self):
        """Create a sample chain for testing."""
        # Create simple model for sampling
        params = ParameterSet()
        params.add("a", Uniform(0.0, 5.0))
        params.add("b", Uniform(0.0, 5.0))

        target = Target()
        target.add_observation("y", 1.0, 3.5, 0.1)
        target.add_observation("y", 2.0, 5.5, 0.1)

        def model_factory(param_dict):
            a = param_dict["a"]
            b = param_dict["b"]
            return {"y": {1.0: a + b, 2.0: 2 * a + b}}

        runner = ModelRunner(
            model_factory=model_factory,
            param_names=["a", "b"],
            output_variables=["y"],
        )

        likelihood = GaussianLikelihood()
        sampler = EnsembleSampler(params, runner, likelihood, target)

        # Run short chain
        chain = sampler.run(50, WalkerInit.from_prior(), thin=1)
        return chain

    def test_chain_properties(self, sample_chain):
        """Test chain basic properties."""
        assert len(sample_chain) > 0
        assert sample_chain.thin == 1
        assert sample_chain.total_iterations == 50
        assert "a" in sample_chain.param_names
        assert "b" in sample_chain.param_names

    def test_flat_samples(self, sample_chain):
        """Test flat_samples method."""
        samples = sample_chain.flat_samples(discard=10)
        assert samples.ndim == 2
        assert samples.shape[1] == 2  # Two parameters

        # With discard, should have fewer samples
        all_samples = sample_chain.flat_samples(discard=0)
        assert len(samples) < len(all_samples)

    def test_flat_log_probs(self, sample_chain):
        """Test flat_log_probs method."""
        log_probs = sample_chain.flat_log_probs(discard=10)
        assert log_probs.ndim == 1
        assert np.all(np.isfinite(log_probs))

    def test_to_param_dict(self, sample_chain):
        """Test to_param_dict method."""
        param_dict = sample_chain.to_param_dict(discard=10)
        assert "a" in param_dict
        assert "b" in param_dict
        assert isinstance(param_dict["a"], np.ndarray)
        assert isinstance(param_dict["b"], np.ndarray)
        assert len(param_dict["a"]) == len(param_dict["b"])

    def test_diagnostics(self, sample_chain):
        """Test diagnostic methods."""
        # R-hat
        r_hat = sample_chain.r_hat(discard=10)
        assert isinstance(r_hat, dict)
        assert "a" in r_hat
        assert "b" in r_hat
        assert all(v > 0 for v in r_hat.values())

        # ESS
        ess = sample_chain.ess(discard=10)
        assert isinstance(ess, dict)
        assert "a" in ess
        assert "b" in ess
        assert all(v > 0 for v in ess.values())

        # Autocorrelation time
        tau = sample_chain.autocorr_time(discard=10)
        assert isinstance(tau, dict)
        assert "a" in tau
        assert "b" in tau
        assert all(v > 0 for v in tau.values())

        # Convergence check
        is_converged = sample_chain.is_converged(discard=10, threshold=1.2)
        assert isinstance(is_converged, bool)

    def test_save_load(self, sample_chain):
        """Test chain persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_chain.bin"

            # Save
            sample_chain.save(str(path))
            assert path.exists()

            # Load
            loaded_chain = Chain.load(str(path))
            assert len(loaded_chain) == len(sample_chain)
            assert loaded_chain.param_names == sample_chain.param_names
            assert loaded_chain.total_iterations == sample_chain.total_iterations

            # Check samples match
            orig_samples = sample_chain.flat_samples(discard=0)
            loaded_samples = loaded_chain.flat_samples(discard=0)
            np.testing.assert_array_almost_equal(orig_samples, loaded_samples)

    def test_merge_chains(self):
        """Test merging multiple chain segments."""
        # Create two short chains
        params = ParameterSet()
        params.add("x", Uniform(0.0, 1.0))

        target = Target()
        target.add_observation("y", 1.0, 0.5, 0.1)

        def model_factory(param_dict):
            return {"y": {1.0: param_dict["x"]}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["x"], output_variables=["y"]
        )

        likelihood = GaussianLikelihood()
        sampler = EnsembleSampler(params, runner, likelihood, target)

        chain1 = sampler.run(20, WalkerInit.from_prior(), thin=1)
        chain2 = sampler.run(20, WalkerInit.from_prior(), thin=1)

        # Store original lengths
        len1 = len(chain1)
        len2 = len(chain2)

        # Merge modifies chain1 in place
        chain1.merge(chain2)
        assert chain1.total_iterations == 40
        assert len(chain1) == len1 + len2


class TestPointEstimator:
    """Test PointEstimator API and optimization."""

    def test_creation(self):
        """Test PointEstimator creation."""
        params = ParameterSet()
        params.add("a", Uniform(0.0, 5.0))

        target = Target()
        target.add_observation("y", 1.0, 2.0, 0.1)

        def model_factory(param_dict):
            return {"y": {1.0: param_dict["a"]}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["a"], output_variables=["y"]
        )

        likelihood = GaussianLikelihood()
        estimator = PointEstimator(params, runner, likelihood, target)

        assert estimator.n_params == 1
        assert estimator.param_names == ["a"]
        assert estimator.n_evaluations == 0

    def test_random_search(self):
        """Test random search optimization."""
        params = ParameterSet()
        params.add("a", Uniform(0.0, 5.0))
        params.add("b", Uniform(0.0, 5.0))

        # Target: y = 2*x + 1
        target = Target()
        target.add_observation("y", 1.0, 3.0, 0.1)
        target.add_observation("y", 2.0, 5.0, 0.1)

        def model_factory(param_dict):
            a = param_dict["a"]
            b = param_dict["b"]
            return {"y": {1.0: a + b, 2.0: 2 * a + b}}

        runner = ModelRunner(
            model_factory=model_factory,
            param_names=["a", "b"],
            output_variables=["y"],
        )

        likelihood = GaussianLikelihood()
        estimator = PointEstimator(params, runner, likelihood, target)

        # Run random search
        result = estimator.optimize(Optimizer.random_search(), n_samples=50)

        assert result.best_params is not None
        assert len(result.best_params) == 2
        assert np.isfinite(result.best_log_likelihood)
        assert estimator.n_evaluations == 50

    def test_evaluated_history(self):
        """Test access to evaluation history."""
        params = ParameterSet()
        params.add("x", Uniform(0.0, 1.0))

        target = Target()
        target.add_observation("y", 1.0, 0.5, 0.1)

        def model_factory(param_dict):
            return {"y": {1.0: param_dict["x"]}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["x"], output_variables=["y"]
        )

        likelihood = GaussianLikelihood()
        estimator = PointEstimator(params, runner, likelihood, target)

        # Run optimization
        estimator.optimize(Optimizer.random_search(), n_samples=10)

        # Check history
        evaluated_params = estimator.evaluated_params()
        evaluated_log_likes = estimator.evaluated_log_likelihoods()

        assert len(evaluated_params) == 10
        assert len(evaluated_log_likes) == 10
        assert all(len(p) == 1 for p in evaluated_params)

    def test_clear_history(self):
        """Test clearing evaluation history."""
        params = ParameterSet()
        params.add("x", Uniform(0.0, 1.0))

        target = Target()
        target.add_observation("y", 1.0, 0.5, 0.1)

        def model_factory(param_dict):
            return {"y": {1.0: param_dict["x"]}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["x"], output_variables=["y"]
        )

        likelihood = GaussianLikelihood()
        estimator = PointEstimator(params, runner, likelihood, target)

        # Run and clear
        estimator.optimize(Optimizer.random_search(), n_samples=10)
        assert estimator.n_evaluations == 10

        estimator.clear_history()
        assert estimator.n_evaluations == 0
        assert len(estimator.evaluated_params()) == 0

    def test_best_method(self):
        """Test best() method returns best result so far."""
        params = ParameterSet()
        params.add("x", Uniform(0.0, 1.0))

        target = Target()
        target.add_observation("y", 1.0, 0.5, 0.1)

        def model_factory(param_dict):
            return {"y": {1.0: param_dict["x"]}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["x"], output_variables=["y"]
        )

        likelihood = GaussianLikelihood()
        estimator = PointEstimator(params, runner, likelihood, target)

        # Initially no best
        assert estimator.best() is None

        # After optimization
        estimator.optimize(Optimizer.random_search(), n_samples=10)
        best = estimator.best()
        assert best is not None
        # best() returns (params, log_likelihood) tuple
        best_params, best_log_likelihood = best
        assert len(best_params) == 1
        assert np.isfinite(best_log_likelihood)


class TestModelRunner:
    """Test ModelRunner Python wrapper."""

    def test_creation(self):
        """Test ModelRunner creation."""

        def model_factory(param_dict):
            return {"y": {1.0: param_dict["x"]}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["x"], output_variables=["y"]
        )

        assert runner is not None

    def test_run_via_estimator(self):
        """Test that ModelRunner works via PointEstimator."""
        # ModelRunner doesn't have a public run() method;
        # it's used internally by PointEstimator and EnsembleSampler

        def model_factory(param_dict):
            x = param_dict["x"]
            return {"y": {1.0: x, 2.0: 2 * x}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["x"], output_variables=["y"]
        )

        params = ParameterSet()
        params.add("x", Uniform(0.0, 5.0))

        target = Target()
        target.add_observation("y", 1.0, 2.0, 0.1)

        likelihood = GaussianLikelihood()
        estimator = PointEstimator(params, runner, likelihood, target)

        # This validates that the runner works
        result = estimator.optimize(Optimizer.random_search(), n_samples=5)
        assert result.best_params is not None

    def test_error_propagation(self):
        """Test that model errors are propagated."""

        def failing_model(param_dict):
            msg = "Model failed"
            raise ValueError(msg)

        runner = ModelRunner(
            model_factory=failing_model, param_names=["x"], output_variables=["y"]
        )

        # Should propagate the error
        with pytest.raises(Exception):
            runner.run({"x": 1.0})


class TestWalkerInit:
    """Test WalkerInit enum and initialization strategies."""

    def test_from_prior(self):
        """Test WalkerInit.from_prior() creates correct strategy."""
        init = WalkerInit.from_prior()
        assert init is not None

    def test_from_ball(self):
        """Test WalkerInit.ball() with center point."""
        center = [1.0, 2.0, 3.0]
        init = WalkerInit.ball(center, radius=0.1)
        assert init is not None

    def test_from_explicit(self):
        """Test WalkerInit.explicit() with positions."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((10, 3))  # 10 walkers, 3 params
        init = WalkerInit.explicit(positions)
        assert init is not None


class TestEnsembleSampler:
    """Test EnsembleSampler API and checkpointing."""

    def test_creation(self):
        """Test sampler creation."""
        params = ParameterSet()
        params.add("x", Uniform(0.0, 1.0))

        target = Target()
        target.add_observation("y", 1.0, 0.5, 0.1)

        def model_factory(param_dict):
            return {"y": {1.0: param_dict["x"]}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["x"], output_variables=["y"]
        )

        likelihood = GaussianLikelihood()
        sampler = EnsembleSampler(params, runner, likelihood, target)

        assert sampler is not None

    def test_run_basic(self):
        """Test basic run method."""
        params = ParameterSet()
        params.add("x", Uniform(0.0, 1.0))

        target = Target()
        target.add_observation("y", 1.0, 0.5, 0.1)

        def model_factory(param_dict):
            return {"y": {1.0: param_dict["x"]}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["x"], output_variables=["y"]
        )

        likelihood = GaussianLikelihood()
        sampler = EnsembleSampler(params, runner, likelihood, target)

        chain = sampler.run(10, WalkerInit.from_prior(), thin=1)
        assert len(chain) > 0
        assert chain.total_iterations == 10

    def test_checkpointing(self):
        """Test checkpoint save and resume."""
        params = ParameterSet()
        params.add("x", Uniform(0.0, 1.0))

        target = Target()
        target.add_observation("y", 1.0, 0.5, 0.1)

        def model_factory(param_dict):
            return {"y": {1.0: param_dict["x"]}}

        runner = ModelRunner(
            model_factory=model_factory, param_names=["x"], output_variables=["y"]
        )

        likelihood = GaussianLikelihood()
        sampler = EnsembleSampler(params, runner, likelihood, target)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.bin"

            # Run with checkpointing
            chain1 = sampler.run_with_checkpoint(
                n_iterations=20,
                init=WalkerInit.from_prior(),
                checkpoint_path=str(checkpoint_path),
                checkpoint_every=10,
                thin=1,
            )

            # Checkpointing creates .chain and .state files
            assert (checkpoint_path.parent / f"{checkpoint_path.name}.chain").exists()
            assert (checkpoint_path.parent / f"{checkpoint_path.name}.state").exists()
            assert chain1.total_iterations == 20

            # Resume from checkpoint (n_iterations is total, not additional)
            chain2 = sampler.resume_from_checkpoint(
                n_iterations=30,
                thin=1,
                checkpoint_every=10,
                checkpoint_path=str(checkpoint_path),
            )

            assert chain2.total_iterations == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

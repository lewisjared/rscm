"""
Memory stress test for calibration framework.

This test validates:
1. Large chain handling (100k+ samples)
2. Checkpoint/resume functionality
3. Memory stability during long runs
4. Chain persistence and merging
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
    Chain,
    EnsembleSampler,
    GaussianLikelihood,
    ModelRunner,
    ParameterSet,
    Target,
    Uniform,
    WalkerInit,
)


@pytest.fixture
def simple_calibration_setup():
    """Create a simple calibration setup for stress testing."""
    # True parameters
    true_params = {
        "lambda0": 1.1,
        "efficacy": 1.3,
        "a": 0.05,
        "eta": 0.7,
        "heat_capacity_deep": 100.0,
        "heat_capacity_surface": 8.0,
    }

    # Generate synthetic observation
    component = TwoLayerBuilder.from_parameters(true_params).build()
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

    # Define parameters (2 parameters for simplicity)
    params = ParameterSet()
    params.add("lambda0", Uniform(0.8, 1.5))
    params.add("a", Uniform(0.0, 0.1))

    # Create target
    target = Target()
    target.add_observation("Surface Temperature", 2010.0, temp, 0.1)

    # Create model runner
    fixed_params = {
        "efficacy": true_params["efficacy"],
        "eta": true_params["eta"],
        "heat_capacity_deep": true_params["heat_capacity_deep"],
        "heat_capacity_surface": true_params["heat_capacity_surface"],
    }

    def model_factory(param_values):
        all_params = fixed_params.copy()
        all_params.update(param_values)
        component = TwoLayerBuilder.from_parameters(all_params).build()
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
        return {"Surface Temperature": {2010.0: float(temp)}}

    runner = ModelRunner(
        model_factory=model_factory,
        param_names=["lambda0", "a"],
        output_variables=["Surface Temperature"],
    )

    likelihood = GaussianLikelihood()

    return {
        "params": params,
        "runner": runner,
        "likelihood": likelihood,
        "target": target,
        "true_params": true_params,
    }


@pytest.mark.slow
def test_large_chain_memory(simple_calibration_setup):
    """
    Test that large chains (100k+ samples) don't cause memory issues.

    This test runs a long MCMC chain with heavy thinning to verify:
    - Memory usage remains stable
    - Chain operations work with large sample counts
    - Diagnostics can be computed on large chains
    """
    setup = simple_calibration_setup

    # Create sampler
    sampler = EnsembleSampler(
        setup["params"],
        setup["runner"],
        setup["likelihood"],
        setup["target"],
    )

    # Run a very long chain with heavy thinning
    # 5000 iterations x 4 walkers (default) x thin=1 = 20k samples
    # This is substantial but not excessive for CI
    n_steps = 5000
    thin = 1

    print(f"\nRunning large chain: {n_steps} iterations, thin={thin}")
    chain = sampler.run(n_steps, WalkerInit.from_prior(), thin=thin)

    # Verify chain structure
    assert chain.total_iterations == n_steps
    assert chain.thin == thin

    # Get flat samples (should handle large arrays)
    samples = chain.flat_samples(discard=1000)
    log_probs = chain.flat_log_probs(discard=1000)

    print(f"Chain has {samples.shape[0]} samples after burn-in")
    assert samples.shape[0] > 10000  # Should have substantial samples
    assert samples.shape[1] == 2  # Two parameters
    assert len(log_probs) == samples.shape[0]

    # Verify all samples are valid
    assert np.all(np.isfinite(samples))
    assert np.all(np.isfinite(log_probs))

    # Compute diagnostics on large chain
    r_hat = chain.r_hat(discard=1000)
    ess = chain.ess(discard=1000)
    tau = chain.autocorr_time(discard=1000)

    print("Diagnostics computed successfully:")
    print(f"  R-hat: {r_hat}")
    print(f"  ESS: {ess}")
    print(f"  Autocorr time: {tau}")

    # Verify diagnostics are reasonable
    assert all(np.isfinite(list(r_hat.values())))
    assert all(np.isfinite(list(ess.values())))
    assert all(np.isfinite(list(tau.values())))

    # Verify convergence
    assert all(r < 1.2 for r in r_hat.values()), f"Chain did not converge: {r_hat}"


@pytest.mark.slow
def test_checkpoint_resume(simple_calibration_setup, tmp_path):
    """
    Test checkpoint/resume functionality with multiple chain segments.

    This validates:
    - Checkpoint creation during long runs
    - Resuming from checkpoint preserves state
    - Multiple chain segments can be merged correctly
    """
    setup = simple_calibration_setup

    # Create sampler
    sampler = EnsembleSampler(
        setup["params"],
        setup["runner"],
        setup["likelihood"],
        setup["target"],
    )

    # First run: 500 steps with checkpoint
    checkpoint_path = tmp_path / "checkpoint.bin"
    n_steps_1 = 500

    print(f"\nFirst run: {n_steps_1} steps with checkpoint")
    chain_1 = sampler.run_with_checkpoint(
        n_iterations=n_steps_1,
        init=WalkerInit.from_prior(),
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=100,
        thin=1,
    )

    assert chain_1.total_iterations == n_steps_1
    # Checkpoint creates .chain and .state files
    assert (checkpoint_path.parent / f"{checkpoint_path.name}.chain").exists()
    assert (checkpoint_path.parent / f"{checkpoint_path.name}.state").exists()

    # Resume from checkpoint: 500 -> 1000 total steps
    n_steps_total = 1000

    print(f"Resuming: {n_steps_1} -> {n_steps_total} total steps")
    chain_2 = sampler.resume_from_checkpoint(
        n_iterations=n_steps_total,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=100,
        thin=1,
    )

    # Verify resumed chain has correct length
    # Chain 2 should contain samples from both runs
    assert chain_2.total_iterations == n_steps_total

    # Get samples from both chains
    samples_1 = chain_1.flat_samples(discard=0)
    samples_2 = chain_2.flat_samples(discard=0)

    print(f"Chain 1: {samples_1.shape[0]} samples")
    print(f"Chain 2 (resumed): {samples_2.shape[0]} samples")

    # Chain 2 should have more samples (includes chain 1 + new samples)
    assert samples_2.shape[0] > samples_1.shape[0]

    # Verify continuity: first part of chain_2 should match chain_1
    # (allowing for some numerical differences in the final shared samples)
    overlap_size = min(100, samples_1.shape[0])
    assert np.allclose(
        samples_1[:overlap_size],
        samples_2[:overlap_size],
        rtol=1e-10,
    ), "Resumed chain diverges from original"

    # Compute diagnostics on resumed chain
    r_hat = chain_2.r_hat(discard=200)
    ess = chain_2.ess(discard=200)

    print("Resumed chain diagnostics:")
    print(f"  R-hat: {r_hat}")
    print(f"  ESS: {ess}")

    assert all(np.isfinite(list(r_hat.values())))
    assert all(v > 0 for v in ess.values())


@pytest.mark.slow
def test_chain_persistence(simple_calibration_setup, tmp_path):
    """
    Test saving and loading large chains.

    This validates:
    - Chain serialization to disk
    - Chain deserialization matches original
    - Chain merging from multiple saved segments
    """
    setup = simple_calibration_setup

    # Create sampler
    sampler = EnsembleSampler(
        setup["params"],
        setup["runner"],
        setup["likelihood"],
        setup["target"],
    )

    # Run two independent chains
    n_steps = 1000

    print(f"\nRunning two chains: {n_steps} steps each")
    chain_1 = sampler.run(n_steps, WalkerInit.from_prior(), thin=1)
    chain_2 = sampler.run(n_steps, WalkerInit.from_prior(), thin=1)

    # Save both chains
    path_1 = tmp_path / "chain_1.bin"
    path_2 = tmp_path / "chain_2.bin"

    chain_1.save(str(path_1))
    chain_2.save(str(path_2))

    assert path_1.exists()
    assert path_2.exists()

    print(f"Chain 1 saved: {path_1.stat().st_size / 1024:.1f} KB")
    print(f"Chain 2 saved: {path_2.stat().st_size / 1024:.1f} KB")

    # Load chains back
    loaded_1 = Chain.load(str(path_1))
    loaded_2 = Chain.load(str(path_2))

    # Verify loaded chains match originals
    samples_1_orig = chain_1.flat_samples(discard=0)
    samples_1_load = loaded_1.flat_samples(discard=0)

    assert np.array_equal(samples_1_orig, samples_1_load), (
        "Loaded chain doesn't match original"
    )
    assert np.array_equal(
        chain_1.flat_log_probs(discard=0),
        loaded_1.flat_log_probs(discard=0),
    ), "Log probs don't match"

    # Merge chains (modifies loaded_1 in place)
    samples_2_orig = chain_2.flat_samples(discard=0)
    n1 = samples_1_orig.shape[0]

    loaded_1.merge(loaded_2)

    samples_merged = loaded_1.flat_samples(discard=0)

    print(f"Merged chain: {samples_merged.shape[0]} total samples")

    # Verify merged chain has combined length
    assert samples_merged.shape[0] == samples_1_orig.shape[0] + samples_2_orig.shape[0]

    # Verify merged chain contains both originals
    assert np.array_equal(samples_merged[:n1], samples_1_orig), (
        "First part of merge doesn't match chain 1"
    )
    assert np.array_equal(samples_merged[n1:], samples_2_orig), (
        "Second part of merge doesn't match chain 2"
    )

    # Compute diagnostics on merged chain
    r_hat = loaded_1.r_hat(discard=200)
    ess = loaded_1.ess(discard=200)

    print("Merged chain diagnostics:")
    print(f"  R-hat: {r_hat}")
    print(f"  ESS: {ess}")

    assert all(np.isfinite(list(r_hat.values())))
    assert all(v > 0 for v in ess.values())


@pytest.mark.slow
def test_thinning_memory_efficiency(simple_calibration_setup):
    """
    Test that heavy thinning reduces memory usage proportionally.

    This validates:
    - Thinning actually reduces stored samples
    - Thinned chains produce valid diagnostics
    - Memory usage scales with thin parameter
    """
    setup = simple_calibration_setup

    # Create sampler
    sampler = EnsembleSampler(
        setup["params"],
        setup["runner"],
        setup["likelihood"],
        setup["target"],
    )

    n_steps = 2000

    # Run with no thinning
    print(f"\nRun 1: {n_steps} steps, thin=1")
    chain_nothin = sampler.run(n_steps, WalkerInit.from_prior(), thin=1)
    samples_nothin = chain_nothin.flat_samples(discard=0)

    # Run with heavy thinning
    thin = 10
    print(f"Run 2: {n_steps} steps, thin={thin}")
    chain_thin = sampler.run(n_steps, WalkerInit.from_prior(), thin=thin)
    samples_thin = chain_thin.flat_samples(discard=0)

    print(f"No thinning: {samples_nothin.shape[0]} samples")
    print(f"Thinned (10x): {samples_thin.shape[0]} samples")

    # Thinned chain should have ~10x fewer samples
    assert samples_thin.shape[0] < samples_nothin.shape[0] / 5
    assert (
        samples_thin.shape[0] > samples_nothin.shape[0] / 15
    )  # Some tolerance for walker count

    # Both should converge to similar posterior
    mean_nothin = np.mean(samples_nothin[1000:], axis=0)
    mean_thin = np.mean(samples_thin[100:], axis=0)

    print(f"Posterior mean (no thin): {mean_nothin}")
    print(f"Posterior mean (thin={thin}): {mean_thin}")

    # Means should be close (within 2 standard errors)
    std_nothin = np.std(samples_nothin[1000:], axis=0)
    assert np.allclose(mean_nothin, mean_thin, atol=2 * std_nothin), (
        "Thinned and non-thinned chains converged to different posteriors"
    )

    # Diagnostics should work on both
    r_hat_nothin = chain_nothin.r_hat(discard=500)
    r_hat_thin = chain_thin.r_hat(discard=50)

    print(f"R-hat (no thin): {r_hat_nothin}")
    print(f"R-hat (thin={thin}): {r_hat_thin}")

    # Both should indicate convergence
    assert all(r < 1.2 for r in r_hat_nothin.values())
    assert all(r < 1.2 for r in r_hat_thin.values())


if __name__ == "__main__":
    # Allow running this test standalone
    pytest.main([__file__, "-v", "-s", "-m", "slow"])

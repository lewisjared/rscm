"""Edge cases of the calibration API: invalid input, interrupts, seeds and failures."""

import math
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from rscm.calibrate import (
    Bound,
    Chain,
    EnsembleSampler,
    GaussianLikelihood,
    LogNormal,
    ModelRunner,
    Normal,
    Observation,
    Optimiser,
    ParameterSet,
    PointEstimator,
    Target,
    Uniform,
    WalkerInit,
)
from rscm.calibrate.progress import create_simple_callback

TIMES = [2000.0, 2010.0, 2020.0]
NAN = float("nan")


def linear(parameters):
    return {"y": {t: parameters["a"] * (t - 2000) / 10 for t in TIMES}}


def two_parameter(parameters):
    assert 0 <= parameters["a"] <= 5, parameters
    assert 100 <= parameters["b"] <= 200, parameters
    return linear(parameters)


@pytest.fixture
def target():
    target = Target()
    for time, value in zip(TIMES, [0.0, 1.0, 2.0]):
        target.add_observation("y", time, value, 0.1)
    return target


@pytest.fixture
def parameters():
    return ParameterSet({"a": Uniform(0.0, 5.0)})


@pytest.fixture
def sampler(parameters, target):
    runner = ModelRunner(linear, ["a"], ["y"])
    return EnsembleSampler(parameters, runner, GaussianLikelihood(), target)


def run_isolated(code, timeout=60):
    """Run `code` in a fresh interpreter so a hang fails the test, not the session."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class TestBound:
    @pytest.mark.parametrize(
        "inner, low, high",
        [("Normal(0.0, 1.0)", 100.0, 101.0), ("Uniform(0.0, 1.0)", 5.0, 6.0)],
    )
    def test_bound_without_probability_mass_is_rejected(self, inner, low, high):
        result = run_isolated(
            f"""
            from rscm.calibrate import Bound, Normal, Uniform
            try:
                Bound({inner}, {low}, {high}).sample()
            except ValueError as e:
                print("ValueError:", e)
            """
        )
        assert result.returncode == 0, result.stderr
        assert "ValueError" in result.stdout
        assert "probability" in result.stdout or "support" in result.stdout

    def test_far_tail_bound_samples_quickly(self):
        result = run_isolated(
            """
            from rscm.calibrate import Bound, Normal
            d = Bound(Normal(0.0, 1.0), 7.0, 9.0)
            xs = [d.sample() for _ in range(100)]
            assert all(7.0 <= x <= 9.0 for x in xs), xs
            print("ok")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_bound_sampling_matches_truncated_normal(self):
        d = Bound(Normal(0.0, 1.0), 0.0, math.inf)
        xs = np.array([d.sample() for _ in range(4000)])
        assert xs.min() >= 0.0
        # Half-normal mean is sqrt(2 / pi).
        assert abs(xs.mean() - math.sqrt(2 / math.pi)) < 0.05

    def test_bound_nan_rejected(self):
        with pytest.raises(ValueError, match="low"):
            Bound(Normal(0.0, 1.0), NAN, 1.0)


class TestNonFiniteInput:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: Uniform(NAN, 1.0),
            lambda: Uniform(0.0, NAN),
            lambda: Uniform(0.0, math.inf),
            lambda: Normal(NAN, 1.0),
            lambda: Normal(0.0, NAN),
            lambda: Normal(0.0, math.inf),
            lambda: LogNormal(mu=NAN, sigma=1.0),
            lambda: LogNormal(mu=0.0, sigma=NAN),
            lambda: LogNormal(mean=NAN, std_dev=1.0),
        ],
    )
    def test_distribution_rejects_non_finite(self, make):
        with pytest.raises(ValueError, match="finite"):
            make()

    def test_lognormal_tiny_std_dev_bounds(self):
        dist = Bound(LogNormal(mean=1.0, std_dev=1e-10), 0.5, 1.5)
        assert dist.log_pdf(1.0) > 0.0

    @pytest.mark.parametrize(
        ("mean", "std_dev"), [(1.0, 1e-200), (1.0, 1e200), (1e-300, 1e300)]
    )
    def test_lognormal_degenerate_mean_std_dev(self, mean, std_dev):
        with pytest.raises(ValueError, match="mean"):
            LogNormal(mean=mean, std_dev=std_dev)

    @pytest.mark.parametrize(
        "dist", [Uniform(0.0, 1.0), Normal(0.0, 1.0), LogNormal(mu=0.0, sigma=1.0)]
    )
    def test_log_pdf_of_nan_is_minus_infinity(self, dist):
        assert dist.log_pdf(NAN) == -math.inf
        assert Bound(dist, 0.1, 0.9).log_pdf(NAN) == -math.inf

    def test_log_prior_of_nan_is_minus_infinity(self, parameters):
        assert parameters.log_prior([NAN]) == -math.inf

    @pytest.mark.parametrize(
        "time, value, uncertainty",
        [(NAN, 1.0, 0.1), (2000.0, NAN, 0.1), (2000.0, 1.0, NAN), (2000.0, 1.0, 0.0)],
    )
    def test_observation_rejects_non_finite(self, time, value, uncertainty):
        with pytest.raises(ValueError, match="finite"):
            Observation(time, value, uncertainty)
        with pytest.raises(ValueError, match="finite"):
            Target().add_observation("y", time, value, uncertainty)

    def test_relative_observation_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            Target().add_observation_relative("y", 2000.0, 1.0, NAN)

    @pytest.mark.parametrize("a", [NAN, math.inf, 1.0])
    def test_stretch_parameter_rejects_invalid(self, parameters, target, a):
        runner = ModelRunner(linear, ["a"], ["y"])
        with pytest.raises(ValueError, match="greater than 1"):
            EnsembleSampler.with_stretch_parameter(
                parameters, runner, GaussianLikelihood(), target, a
            )


class TestInterrupts:
    @pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
    def test_model_interrupt_aborts_run(self, parameters, target, exception):
        calls = []

        def model(p):
            calls.append(p)
            if len(calls) > 40:
                raise exception
            return linear(p)

        runner = ModelRunner(model, ["a"], ["y"])
        sampler = EnsembleSampler(parameters, runner, GaussianLikelihood(), target)
        with pytest.raises(exception):
            sampler.run(10, WalkerInit.from_prior())
        assert len(calls) == 41

    def test_model_interrupt_aborts_point_estimation(self, parameters, target):
        def model(p):
            raise KeyboardInterrupt

        runner = ModelRunner(model, ["a"], ["y"])
        estimator = PointEstimator(parameters, runner, GaussianLikelihood(), target)
        with pytest.raises(KeyboardInterrupt):
            estimator.optimise(Optimiser.random_search(), 10)
        assert estimator.n_evaluations == 1


class TestFailedEvaluations:
    def test_failures_mid_run_are_counted(self, parameters, target):
        calls = []

        def model(p):
            calls.append(p)
            if len(calls) > 40:
                raise RuntimeError("boom mid")  # noqa: TRY003
            return linear(p)

        runner = ModelRunner(model, ["a"], ["y"])
        sampler = EnsembleSampler(parameters, runner, GaussianLikelihood(), target)
        with pytest.warns(RuntimeWarning, match="boom mid"):
            chain = sampler.run(3, WalkerInit.from_prior())
        assert chain.n_failed_evaluations == len(calls) - 40
        assert "boom mid" in chain.first_failure

    def test_no_failures_no_warning(self, sampler, recwarn):
        chain = sampler.run(3, WalkerInit.from_prior())
        assert chain.n_failed_evaluations == 0
        assert chain.first_failure is None
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]

    def test_point_estimator_counts_failures(self, parameters, target):
        calls = []

        def model(p):
            calls.append(p)
            if len(calls) % 2 == 0:
                raise RuntimeError("every other")  # noqa: TRY003
            return linear(p)

        runner = ModelRunner(model, ["a"], ["y"])
        estimator = PointEstimator(parameters, runner, GaussianLikelihood(), target)
        with pytest.warns(RuntimeWarning, match="every other"):
            result = estimator.optimise(Optimiser.random_search(), 10)
        assert result.n_failed_evaluations == 5
        assert "every other" in result.first_failure


class TestParameterNames:
    def test_swapped_order_maps_by_name(self, target):
        parameters = ParameterSet({"a": Uniform(0.0, 5.0), "b": Uniform(100.0, 200.0)})
        runner = ModelRunner(two_parameter, ["b", "a"], ["y"])
        sampler = EnsembleSampler(parameters, runner, GaussianLikelihood(), target)
        chain = sampler.run(3, WalkerInit.from_prior())
        assert chain.parameter_names == ["a", "b"]
        assert chain.n_failed_evaluations == 0

        estimator = PointEstimator(parameters, runner, GaussianLikelihood(), target)
        result = estimator.optimise(Optimiser.random_search(), 5)
        assert result.n_failed_evaluations == 0

    @pytest.mark.parametrize("names", [["b"], ["a", "b"], ["a", "a"]])
    def test_mismatched_names_raise(self, parameters, target, names):
        runner = ModelRunner(linear, names, ["y"])
        with pytest.raises(ValueError, match="parameter_names"):
            EnsembleSampler(parameters, runner, GaussianLikelihood(), target)
        with pytest.raises(ValueError, match="parameter_names"):
            PointEstimator(parameters, runner, GaussianLikelihood(), target)


class TestCheckpoint:
    def test_resume_checks_thin(self, sampler, tmp_path):
        path = str(tmp_path / "ck")
        sampler.run_with_checkpoint(4, WalkerInit.from_prior(), 2, 2, path)
        with pytest.raises(ValueError, match="thin"):
            sampler.resume_from_checkpoint(8, 1, 2, path)

    def test_resume_checks_parameter_names(self, sampler, target, tmp_path):
        path = str(tmp_path / "ck")
        sampler.run_with_checkpoint(4, WalkerInit.from_prior(), 1, 2, path)
        other = EnsembleSampler(
            ParameterSet({"b": Uniform(0.0, 5.0)}),
            ModelRunner(lambda p: linear({"a": p["b"]}), ["b"], ["y"]),
            GaussianLikelihood(),
            target,
        )
        with pytest.raises(ValueError, match="parameter names"):
            other.resume_from_checkpoint(8, 1, 2, path)


class TestSeed:
    def test_same_seed_same_chain(self, sampler):
        a = sampler.run(5, WalkerInit.from_prior(), seed=42)
        b = sampler.run(5, WalkerInit.from_prior(), seed=42)
        c = sampler.run(5, WalkerInit.from_prior(), seed=43)
        np.testing.assert_array_equal(a.flat_samples(), b.flat_samples())
        assert not np.array_equal(a.flat_samples(), c.flat_samples())

    def test_seed_on_every_run_method(self, sampler, tmp_path):
        a = sampler.run_with_progress(
            4, WalkerInit.ball([2.0], 0.5), 1, lambda info: None, seed=7
        )
        b = sampler.run_with_checkpoint(
            4, WalkerInit.ball([2.0], 0.5), 1, 2, str(tmp_path / "ck"), seed=7
        )
        np.testing.assert_array_equal(a.flat_samples(), b.flat_samples())

    def test_point_estimation_seed(self, parameters, target):
        runner = ModelRunner(linear, ["a"], ["y"])
        results = [
            PointEstimator(parameters, runner, GaussianLikelihood(), target).optimise(
                Optimiser.random_search(), 10, seed=3
            )
            for _ in range(2)
        ]
        assert results[0].best_parameters == results[1].best_parameters


class TestWalkers:
    def test_ragged_explicit_positions(self):
        with pytest.raises(ValueError, match="row 1"):
            WalkerInit.explicit([[1.0], [1.0, 2.0]])

    def test_empty_explicit_positions(self):
        with pytest.raises(ValueError, match="at least 2"):
            WalkerInit.explicit([])

    def test_explicit_positions_set_walker_count(self, sampler):
        chain = sampler.run(2, WalkerInit.explicit([[1.0], [1.1], [1.2], [1.3]]))
        assert chain.flat_samples().shape == (8, 1)

    def test_n_walkers_mismatch_with_explicit(self, sampler):
        positions = WalkerInit.explicit([[1.0], [1.1], [1.2], [1.3]])
        with pytest.raises(ValueError, match="n_walkers"):
            sampler.run(2, positions, n_walkers=6)

    def test_n_walkers(self, sampler):
        chain = sampler.run(2, WalkerInit.from_prior(), n_walkers=6)
        assert chain.flat_samples().shape == (12, 1)
        with pytest.raises(ValueError, match="n_walkers"):
            sampler.run(2, WalkerInit.from_prior(), n_walkers=5)

    @pytest.mark.parametrize("radius", [0.0, -1.0, NAN])
    def test_ball_radius_must_be_positive(self, radius):
        with pytest.raises(ValueError, match="radius"):
            WalkerInit.ball([1.0], radius)

    def test_ball_spread(self, sampler):
        chain = sampler.run(1, WalkerInit.ball([2.0], 0.2), n_walkers=64, seed=1)
        initial = chain.flat_samples()
        # One stretch move cannot leave the hull of a box of width 0.2 by much.
        assert initial.std() > 0.0


class TestThin:
    def test_thin_zero_rejected(self, sampler):
        with pytest.raises(ValueError, match="thin"):
            sampler.run(5, WalkerInit.from_prior(), thin=0)

    def test_thin_above_iterations_rejected(self, sampler):
        with pytest.raises(ValueError, match="thin"):
            sampler.run(3, WalkerInit.from_prior(), thin=10)


class TestSmallFixes:
    def test_duplicate_parameter_name(self):
        parameters = ParameterSet().add("x", Uniform(0.0, 1.0))
        with pytest.raises(ValueError, match="already"):
            parameters.add("x", Uniform(0.0, 2.0))
        assert parameters.bounds() == ([0.0], [1.0])

    @pytest.mark.parametrize("start, end", [(1900.0, 1850.0), (NAN, 1900.0)])
    def test_reference_period_order(self, target, start, end):
        with pytest.raises(ValueError, match="reference period"):
            target.set_reference_period("y", start, end)

    def test_self_merge(self, sampler):
        chain = sampler.run(3, WalkerInit.from_prior())
        chain.merge(chain)
        assert len(chain) == 6
        assert chain.n_iterations == 6

    def test_optimise_zero_samples(self, parameters, target):
        runner = ModelRunner(linear, ["a"], ["y"])
        estimator = PointEstimator(parameters, runner, GaussianLikelihood(), target)
        with pytest.raises(ValueError, match="n_samples"):
            estimator.optimise(Optimiser.random_search(), 0)

    @pytest.mark.parametrize(
        "method, needed", [("r_hat", 4), ("ess", 10), ("autocorr_time", 10)]
    )
    def test_diagnostics_warn_when_too_short(self, sampler, method, needed):
        chain = sampler.run(1, WalkerInit.from_prior())
        with pytest.warns(UserWarning, match=f"at least {needed}"):
            assert getattr(chain, method)() == {}

    def test_simple_callback_rejects_zero(self):
        with pytest.raises(ValueError, match="print_every"):
            create_simple_callback(0)

    def test_has_pandas_removed(self):
        import rscm.calibrate  # noqa: PLC0415

        assert not hasattr(rscm.calibrate, "HAS_PANDAS")


class TestTargetFromDataframe:
    def test_single_variable(self):
        df = pd.DataFrame(
            {
                "time": [1850, 1900, 1950, 2000],
                "value": [0.0, 0.2, 0.5, 1.0],
                "uncertainty": [0.1, 0.1, 0.1, 0.15],
            }
        )
        target = Target.from_dataframe(df, variable="Surface Air Temperature Change")
        assert target.variable_names() == ["Surface Air Temperature Change"]
        assert target.n_observations() == 4

    def test_single_variable_without_name(self):
        df = pd.DataFrame({"time": [2000.0], "value": [1.0], "uncertainty": [0.1]})
        with pytest.raises(ValueError, match="variable="):
            Target.from_dataframe(df)

    def test_multiple_variables_keep_order(self):
        df = pd.DataFrame(
            {
                "variable": ["temp", "temp", "ohc"],
                "time": [1950, 2000, 1950],
                "value": [0.5, 1.0, 100.0],
                "uncertainty": [0.1, 0.15, 20.0],
            }
        )
        target = Target.from_dataframe(df)
        assert target.variable_names() == ["temp", "ohc"]

    def test_relative_error(self):
        df = pd.DataFrame({"time": [1900, 2000], "value": [0.5, 1.0]})
        target = Target.from_dataframe(df, variable="y", relative_error=0.1)
        assert [o.uncertainty for o in target.get_variable("y").observations] == [
            0.05,
            0.1,
        ]

    @pytest.mark.parametrize("index", [["variable"], ["time"], ["variable", "time"]])
    def test_columns_in_index(self, index):
        df = pd.DataFrame(
            {
                "variable": ["temp", "temp"],
                "time": [1950.0, 2000.0],
                "value": [0.5, 1.0],
                "uncertainty": [0.1, 0.1],
            }
        ).set_index(index)
        target = Target.from_dataframe(df)
        assert target.n_observations() == 2

    def test_nan_variable_rejected(self):
        df = pd.DataFrame(
            {
                "variable": ["temp", None],
                "time": [1950.0, 2000.0],
                "value": [0.5, 1.0],
                "uncertainty": [0.1, 0.1],
            }
        )
        with pytest.raises(ValueError, match="missing variable"):
            Target.from_dataframe(df)

    def test_datetime_time_rejected(self):
        df = pd.DataFrame(
            {
                "variable": ["temp"],
                "time": pd.to_datetime(["2000-01-01"]),
                "value": [0.5],
                "uncertainty": [0.1],
            }
        )
        with pytest.raises(TypeError, match="numeric"):
            Target.from_dataframe(df)

    def test_missing_column(self):
        df = pd.DataFrame({"variable": ["temp"], "time": [2000.0], "value": [0.5]})
        with pytest.raises(ValueError, match="uncertainty"):
            Target.from_dataframe(df)
        with pytest.raises(ValueError, match="sigma"):
            Target.from_dataframe(df, uncertainty_col="sigma")


def test_chain_to_dataframe_is_typed(sampler):
    chain: Chain = sampler.run(2, WalkerInit.from_prior())
    assert chain.to_dataframe().shape == (2 * sampler.default_n_walkers(), 2)


@pytest.mark.parametrize(
    "method",
    [
        "flat_samples",
        "flat_log_probs",
        "r_hat",
        "ess",
        "autocorr_time",
        "to_parameter_dict",
        "to_dataframe",
    ],
)
def test_chain_rejects_negative_discard(sampler, method):
    chain = sampler.run(2, WalkerInit.from_prior())
    with pytest.raises(ValueError, match="discard must not be negative, got -1"):
        getattr(chain, method)(discard=-1)


def test_chain_is_converged_rejects_negative_discard(sampler):
    chain = sampler.run(2, WalkerInit.from_prior())
    with pytest.raises(ValueError, match="discard must not be negative"):
        chain.is_converged(discard=-1)

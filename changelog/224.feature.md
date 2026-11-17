Added reproducible calibration runs and failure reporting to `rscm.calibrate`:

- The `EnsembleSampler` run methods and `PointEstimator.optimise` take a `seed` keyword.
- The `EnsembleSampler` run methods take an `n_walkers` keyword, which defaults to the number of `WalkerInit.explicit` rows.
- Failed model evaluations are counted in `n_failed_evaluations` and `first_failure` on `Chain` and `OptimisationResult`, with a `RuntimeWarning`.
- `Chain.to_dataframe` and `Target.from_dataframe` are in the type stubs, and `HAS_PANDAS` is removed.

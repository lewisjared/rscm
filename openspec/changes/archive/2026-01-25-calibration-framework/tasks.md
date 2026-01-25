# Implementation Tasks: Calibration Framework

## 1. Core Infrastructure

- [x] 1.1 Create `rscm-calibrate` crate with workspace integration
  - Add to workspace Cargo.toml
  - Set up crate structure (lib.rs, modules)
  - Add dependencies: rayon, rand, rand_distr, ndarray, statrs

- [x] 1.2 Implement Distribution trait and core distributions
  - Uniform distribution with bounds
  - Normal distribution
  - Bound<D> wrapper for constraining any distribution
  - LogNormal distribution
  - Tests for sampling and ln_pdf correctness

- [x] 1.3 Implement ParameterSet
  - Dict-based construction: `ParameterSet::from_map(IndexMap<String, Box<dyn Distribution>>)`
  - Fluent builder: `.add()` returns `&mut Self` for chaining
  - `param_names() -> Vec<&str>` for indexed access
  - Sample n parameter vectors from priors (random, LHS)
  - Compute log prior probability for parameter vector (indexed)
  - Bounds extraction for optimisers
  - Serialisation support via typetag

## 2. Target and Likelihood

- [x] 2.1 Implement Target struct for observations
  - Fluent builder: `.add_variable()` and `.with_reference_period()` return `&mut Self`
  - Store time-indexed observations with uncertainties
  - Support multiple variables (temperature, OHC, ERF)
  - Reference period normalization (anomaly calculation)
  - Support relative_error specification (compute uncertainties as % of values)

- [x] 2.2 Implement LikelihoodFn trait and GaussianLikelihood
  - Match model output times to observation times
  - Compute Gaussian log-likelihood with observation uncertainties
  - Support relative error specification (e.g., 5%)
  - Handle missing data / NaN values

## 3. Model Integration

- [x] 3.1 Implement ModelRunner trait
  - `param_names() -> &[String]` for indexed parameter access
  - `run(&[f64]) -> RSCMResult<ModelOutput>` (indexed by param_names order)
  - `run_batch(&[Vec<f64>]) -> Vec<RSCMResult<ModelOutput>>`
  - Avoid HashMap in hot path for performance

- [x] 3.2 Implement default ModelRunner for RSCM Model
  - Model factory closure: `Fn(&[f64]) -> Model`
  - Run model and extract specified output variables
  - Return Err for model failures (solver divergence, invalid state)

- [x] 3.3 Implement parallel batch execution
  - Use rayon for parallel model runs
  - Thread-safe model construction (factory is Send + Sync)
  - Collect results preserving order

## 4. Ensemble Sampler (MCMC)

- [x] 4.1 Implement SamplerState and Chain structures
  - Walker positions array (n_walkers, n_params)
  - Log probabilities array
  - Chain storage with thinning support
  - Acceptance tracking per walker

- [x] 4.2 Implement stretch move proposal
  - Sample z from g(z) = ((a-1)*U + 1)^2 / a
  - Select complementary walker uniformly
  - Compute proposal: y = c_j + z * (x_i - c_j)
  - Tests against reference implementation

- [x] 4.3 Implement ensemble sampler main loop
  - Split walkers into two groups
  - Parallel proposal evaluation for each group
  - Accept/reject with correct probability
  - Thinning (store every Nth sample)
  - Default n_walkers = max(2 * n_params, 32)

- [x] 4.4 Implement walker initialization methods
  - Sample from prior
  - Ball around point (for warm starts from point estimate)
  - Explicit positions

- [x] 4.5 Implement progress reporting
  - ProgressInfo struct: iteration, total, acceptance_rate, mean_log_prob
  - Callback interface: `FnMut(&ProgressInfo)`

- [x] 4.6 Implement checkpointing
  - Save SamplerState at checkpoint_every intervals
  - Binary format: header + positions + log_probs
  - `resume(path) -> SamplerState` to continue from checkpoint
  - Merge checkpoint samples into final Chain

## 5. Point Estimation

- [x] 5.1 Implement PointEstimator struct
  - Wrap parameters, target, runner, likelihood
  - Track all evaluated points and likelihoods

- [x] 5.2 Integrate argmin optimizers
  - L-BFGS-B for bounded optimization
  - Nelder-Mead for derivative-free
  - Particle Swarm for global search
  - Random search baseline

- [x] 5.3 Implement PointEstimateResult (OptimizationResult)
  - Best parameters and likelihood
  - All samples explored
  - Convergence information

## 6. Diagnostics (as Chain methods)

- [x] 6.1 Implement Chain.r_hat() - Gelman-Rubin statistic
  - Split chains for within/between variance
  - Per-parameter R-hat values as HashMap<String, f64>
  - `is_converged(discard, threshold)` helper (all R-hat < threshold)

- [x] 6.2 Implement Chain.ess() - Effective sample size
  - Autocorrelation estimation
  - Per-parameter ESS as HashMap<String, f64>

- [x] 6.3 Implement Chain.autocorr_time() - Autocorrelation time
  - Per-parameter autocorrelation time
  - Useful for determining appropriate thin interval

- [x] 6.4 Implement acceptance tracking
  - `acceptance_fraction() -> Vec<f64>` (per-walker)
  - `mean_acceptance_rate() -> f64` (overall)

## 7. Persistence

- [x] 7.1 Implement ChainStorage for binary format
  - Save chain to file (bincode serialization)
  - Load chain from file
  - Chain implements save() and load() methods

- [x] 7.2 Implement checkpoint/resume
  - Save sampler state periodically (SamplerState::save_checkpoint)
  - Resume from checkpoint (SamplerState::load_checkpoint)
  - Merge multiple chain segments (Chain::merge)

## 8. Python Bindings

- [x] 8.1 Create PyO3 bindings for distributions
  - PyUniform, PyNormal, PyBound, PyLogNormal
  - Python-friendly constructors

- [x] 8.2 Create PyO3 bindings for ParameterSet
  - Dict constructor: `ParameterSet({"x": Uniform(0, 1)})`
  - Fluent builder: `.add()` returns self for chaining
  - `sample_random()` and `sample_lhs()` returning numpy arrays
  - `param_names` property for parameter names
  - `log_prior()` and `bounds()` methods

- [x] 8.3 Create PyO3 bindings for Target
  - Fluent API: `.add_observation()`, `.add_observation_relative()`, `.set_reference_period()` return self
  - PyObservation, PyVariableTarget, PyTarget classes
  - Support relative_error parameter via add_observation_relative()
  - NOTE: Deferred `from_dataframe()` and `from_scmdata()` - can be added later as Python helpers

- [x] 8.4 Create PyO3 bindings for ModelRunner
  - PyModelRunner wrapper wrapping Python callable (Session 16) ✓
  - Constructor with model_factory, param_names, output_variables ✓
  - Callable takes dict, returns dict[str, dict[float, float]] ✓
  - Implements ModelRunner trait with Python→Rust→Python calls ✓

- [x] 8.5 Create PyO3 bindings for EnsembleSampler
  - PyEnsembleSampler wrapper (Session 17) ✓
  - Constructor with params, runner, likelihood, target ✓
  - PyWalkerInit enum for initialization strategies ✓
  - run(), run_with_progress(), run_with_checkpoint(), resume_from_checkpoint() ✓
  - Progress callback integration with Python ✓
  - Full integration test validates end-to-end workflow ✓

- [x] 8.6 Create PyO3 bindings for Chain
  - `flat_samples(discard)` → numpy array ✓
  - `flat_log_probs(discard)` → numpy array ✓
  - `to_param_dict(discard)` → dict of numpy arrays ✓
  - `param_names`, `thin`, `total_iterations` properties ✓
  - Diagnostic methods: `r_hat()`, `ess()`, `autocorr_time()`, `is_converged()` ✓
  - Persistence: `save()`, `load()`, `merge()` ✓
  - NOTE: Deferred `to_dataframe()` and `plot` namespace - can be added as Python helpers

- [x] 8.7 Create PyO3 bindings for PointEstimator
  - PyPointEstimator wrapper with constructor accepting params, runner, likelihood, target
  - PyOptimizer enum with random_search() static method
  - PyOptimizationResult wrapper with property accessors
  - optimize() method with optimizer and n_samples params
  - Methods: clear_history(), best(), evaluated_params(), evaluated_log_likelihoods()
  - Properties: n_params, param_names, n_evaluations
  - Full manual testing validates all functionality ✓

## 9. Python Package

- [x] 9.1 Create python/rscm/calibrate/ package
  - __init__.py with public exports
  - Type stubs (.pyi files) for all public classes
  - pandas_helpers.py for DataFrame integration
  - Fixed type stub inconsistencies (Session 5):
    - Chain.merge() now correctly documented as instance method
    - EnsembleSampler.resume_from_checkpoint() corrected to instance method

- [ ] 9.2 Create ChainPlotter class for plot namespace
  - `trace(discard)` - parameter traces over iterations
  - `corner(discard)` - corner plot using corner library
  - `autocorr()` - autocorrelation vs lag
  - `acceptance()` - acceptance rate over time
  - `posterior_predictive(runner, target, n_samples, discard)` - model runs vs observations

- [x] 9.3 Create Target convenience constructors
  - `Target.from_dataframe()` - pandas import (implemented in pandas_helpers.py)
  - Supports relative_error and absolute uncertainties
  - Documented in docstrings

- [x] 9.4 Create progress integration
  - tqdm wrapper for progress callback (create_tqdm_callback)
  - Simple text callback (create_simple_callback)
  - ProgressTracker class for metrics history
  - All Jupyter notebook compatible

## 10. Testing and Validation

- [x] 10.1 Unit tests for all Rust components
  - Distribution sampling and pdf tests ✓
  - ParameterSet operations (both construction styles) ✓
  - Likelihood calculations (including model failure case) ✓
  - Sampler move correctness vs reference implementation ✓
  - 114 tests total, all passing

- [x] 10.2 Sampler correctness test
  - Sample from known posterior (uniform prior, single observation)
  - Verify recovered mean and std match analytical posterior
  - Uses effective sample size (ESS) for proper standard error calculation
  - Tests convergence via R-hat diagnostic
  - Implemented in test_sampler_correctness_multivariate_normal

- [x] 10.3 Parallel determinism test
  - Documents that full bit-for-bit determinism is NOT guaranteed
  - Due to thread_rng() usage in parallel execution paths
  - Tests verify both runs converge to similar posteriors (within 1 std)
  - Implemented in test_parallel_determinism
  - Users should rely on reproducible methodology, not identical samples

- [x] 10.4 Edge case tests
  - Single parameter (1D sampling) ✓ - test_edge_case_single_parameter
  - High-dimensional (50 parameters) ✓ - test_edge_case_high_dimensional
  - All walkers initialized close together ✓ - test_edge_case_all_walkers_same_init
  - Note: Walkers at exactly the same point cannot move (stretch move limitation)

- [x] 10.5 Integration test: simple model calibration
  - Linear model with known parameters
  - Verify MCMC recovers true posterior
  - Check convergence diagnostics (R-hat < 1.1)
  - Implemented in tests/test_calibration_simple.py with 3 comprehensive tests

- [x] 10.6 Integration test: two-layer model calibration
  - Calibrate against synthetic observations
  - Verify parameter recovery within credible intervals
  - Performance benchmark (samples/second)

- [x] 10.7 Memory stress test
  - Chain with 100k+ samples
  - Verify no OOM
  - Test checkpoint/resume with large chains

- [x] 10.8 Python API tests
  - Full workflow from Python (both API styles)
  - Comprehensive test suite (tests/test_calibration_python_api.py)
  - 39 tests covering all public APIs
  - Progress bar integration (tests/test_progress_integration.py)
  - Note: Visualization tests deferred (can be added as Python-only tests later)

## 11. Documentation

- [x] 11.1 Rustdoc for public API
  - All public structs and traits ✓
  - Examples in doc comments ✓
  - Algorithm references ✓
  - Enhanced lib.rs with comprehensive quick start example
  - Fixed HTML tag warnings (backticks around type names)
  - Builds cleanly with `cargo doc --no-deps`

- [x] 11.2 Python docstrings
  - All public classes and functions ✓
  - Usage examples ✓
  - Comprehensive numpy-style docstrings in .pyi files
  - Progress utilities documented (tqdm, simple callbacks, ProgressTracker)
  - Pandas helpers documented (DataFrame integration)

- [x] 11.3 Tutorial notebook
  - End-to-end calibration example ✓
  - Interpretation of diagnostics ✓
  - Visualization examples ✓
  - Created comprehensive Jupyter notebook with 59 cells
  - Covers beginner (linear model), intermediate (two-layer climate), and advanced (checkpointing/DataFrame) workflows
  - File: examples/calibration_tutorial.ipynb

# Investigation: Model Calibration and Constraining Framework

## Context

The user wants to add calibration and constraining capabilities to RSCM:

- **Calibration**: Point estimation to find a single best-fit parameter set
- **Constraining**: MCMC-based posterior distribution estimation for uncertainty quantification

These are critical for the scientific process of developing climate model components. Targets come from IPCC AR6 ranges, observational data, and other reference sources. The system must be usable from both Python and Rust, with most computation happening in Rust for performance.

## Codebase Analysis

### Relevant Existing Code in RSCM

| File | Purpose | Relevance |
|------|---------|-----------|
| `crates/rscm-core/src/model.rs:1073` | Model struct with components, collection, time_axis | Core execution target for calibration |
| `crates/rscm-core/src/model.rs:111` | ModelBuilder for constructing models | Will need parameter injection |
| `crates/rscm-core/src/component.rs:312` | Component trait with `solve()` | Parameters embedded in component structs |
| `crates/rscm-components/src/components/co2_erf.rs:18` | CO2ERFParameters example | Pattern for typed, serializable parameters |
| `crates/rscm-core/src/python/model.rs:200` | Model.to_toml()/from_toml() | Serialization for checkpointing |

**Key observations:**

- Parameters are embedded in component structs and serialize via serde
- No existing calibration, optimization, or MCMC infrastructure
- Model executes components in dependency order via BFS
- Full model serialization enables checkpoint/resume
- Python bindings exist for model building and execution

### Existing Calibration Infrastructure (scmcallib)

The scmcallib library provides clean abstractions for calibration:

**Core Classes:**

- `ParameterSet` - Defines tunable parameters with distributions (priors)
- `Distribution` - Prior specification (Uniform, Normal, Bound, etc.)
- `BaseFinder` - Abstract calibration interface with target/transform handling
- `PointEstimateFinder` - Single best-fit optimization
- `DistributionFinder` - Bayesian posterior estimation via PyMC3
- `Evaluator` - Runs model, computes metrics, tracks samples
- `BaseOptimiser` - Pluggable optimization backend (scipy, bayesopt)
- `BaseSCM` - Abstract model interface for calibration

**Key patterns from scmcallib:**

1. Clean separation: parameter specification → target → evaluation → optimization → results
2. Metrics normalized to "maximize" for consistent interface
3. Transforms applied to model output before metric calculation
4. Reference period normalization for anomaly-based comparison
5. Weights per variable for multi-objective calibration

### Existing Workflow (magicc7-calibration)

The MAGICC7 calibration uses a two-stage approach:

1. **Point Estimation**: Bayesian optimization via `scmcallib.OptimiserBayesOpt`
   - Random exploration (100+ samples) → GP surrogate → iterative refinement
   - Metrics: R², MSE, MAE, explained variance

2. **MCMC Constraining**: Ensemble sampling via `emcee`
   - Affine-invariant ensemble sampler with parallel walkers
   - Uniform priors on parameter bounds
   - Gaussian likelihood with 5% relative variance
   - HDF5 backend for chain persistence
   - Diagnostics: trace plots, corner plots, convergence metrics

3. **Parallel Execution**: ProcessPoolExecutor with worker initialization
   - Each worker gets isolated MAGICC instance
   - Shared state via Manager().dict()
   - Batch processing with progress tracking

### Patterns to Follow

1. **Component Parameter Pattern** (from RSCM):

   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct MyComponentParameters {
       pub param1: FloatValue,
       pub param2: FloatValue,
   }
   ```

2. **Calibration Problem Definition** (from scmcallib):

   ```python
   param_set = ParameterSet()
   param_set.set_tune('sensitivity', Bound(Normal(mu=3.0, sigma=0.5), lower=1.5, upper=6.0))
   finder.set_target(observations, weights={'Temperature': 2.0})
   result = finder.find_best_fit(scm, optimiser_name='bayesopt')
   ```

3. **MCMC Likelihood** (from magicc7-calibration):

   ```python
   def ln_likelihood(theta):
       if not in_bounds(theta): return -inf
       model_output = run_model(theta)
       var = (target * 0.05) ** 2  # 5% relative error
       return -0.5 * sum((target - model_output)**2 / var)
   ```

### Potential Conflicts or Concerns

1. **Parameter Mutability**: RSCM components are immutable after construction. Calibration requires running with different parameters, meaning model reconstruction or a parameter injection mechanism.

2. **Python/Rust Boundary**: If optimization logic is in Python but model execution in Rust, frequent boundary crossings add overhead. MCMC with 10,000+ samples could be bottlenecked.

3. **Parallelism Model**: RSCM models are not currently parallelized. Calibration needs embarrassingly parallel runs with different parameters.

4. **State Management**: MCMC chains need persistence (HDF5 backend in emcee). Rust would need similar infrastructure.

5. **Diagnostics**: Corner plots, trace plots use matplotlib. Pure Rust would need plotting library or export data for Python visualization.

## Approaches Considered

### Approach A: Rust Core with Python Orchestration (Hybrid)

**Description:** Implement core calibration primitives (parameter sampling, likelihood calculation, parallel model execution) in Rust with PyO3 bindings. Keep high-level workflow orchestration and visualization in Python, leveraging existing ecosystem (emcee, scipy, matplotlib).

**Architecture:**

```
Python Layer (orchestration):
├── ParameterSet definition (priors, bounds)
├── Target/observation loading
├── Optimizer selection (emcee, scipy via Python)
├── Results analysis & plotting
└── Calls Rust for batch model execution

Rust Layer (computation):
├── BatchRunner - runs N models with different parameters
├── LikelihoodEvaluator - computes log-likelihood for batch
├── ParameterSampler - generates samples from priors (LHS, random)
└── Parallel execution via rayon
```

**Pros:**

- Leverages mature Python MCMC/optimization libraries (emcee, scipy, PyMC)
- Faster time to value - less Rust code to write
- Scientists comfortable with Python workflow
- Visualization stays in matplotlib/corner

**Cons:**

- Python/Rust boundary crossing for each MCMC iteration (mitigated by batch API)
- Two languages to maintain
- MCMC sampler logic not in Rust (less control over parallelism)

**Estimated scope:** Medium

### Approach B: Pure Rust Calibration Library

**Description:** Implement complete calibration infrastructure in Rust, including MCMC samplers, optimizers, and diagnostics. Expose via PyO3 for Python users.

**Architecture:**

```
Rust Calibration Crate (rscm-calibrate):
├── distributions/ - Prior distributions (Uniform, Normal, Bound, etc.)
├── parameters/ - ParameterSet, bounds, transformations
├── samplers/ - MCMC (emcee-style ensemble, Metropolis-Hastings)
├── optimizers/ - Point estimation (L-BFGS, Bayesian opt via argmin)
├── likelihood/ - Likelihood functions, metrics
├── diagnostics/ - Convergence (R-hat, ESS), chain analysis
└── parallel/ - Rayon-based batch execution

Python Bindings:
├── PyParameterSet, PyDistribution
├── PySampler (wraps Rust sampler)
├── PyCalibrationResult
└── Export to numpy/pandas for visualization
```

**Rust Dependencies:**

- `argmin` - Optimization algorithms (L-BFGS, CMA-ES, etc.)
- `emcee` crate or custom ensemble sampler
- `rayon` - Parallel execution
- `statrs` - Statistical distributions
- `hdf5` - Chain persistence

**Pros:**

- Maximum performance - no Python boundary overhead in hot loop
- Full control over parallelism and memory
- Single language for core logic
- Consistent with RSCM's Rust-first philosophy

**Cons:**

- Larger implementation effort
- Rust MCMC libraries less mature than Python equivalents
- Need Rust plotting or export-to-Python for diagnostics
- Scientists may find pure Rust workflow less familiar

**Estimated scope:** Large

### Approach C: Thin Rust Runner + Python scmcallib Adapter

**Description:** Create minimal Rust infrastructure for fast batch model execution. Adapt scmcallib to use RSCM as its SCM backend via a Python adapter class.

**Architecture:**

```
RSCM (existing + minor additions):
├── Model.run_batch(parameters: List[Dict]) -> List[Timeseries]
└── Parallel execution via rayon internally

Python Adapter:
├── class RSCMBackend(BaseSCM):
│   def _run_single(self, params) -> ScmDataFrame:
│       return self.rscm_model.run_batch([params])[0]
└── Uses existing scmcallib PointEstimateFinder, DistributionFinder

scmcallib (mostly unchanged):
├── ParameterSet, Distribution (reuse)
├── Finder classes (reuse)
└── Optimizers (reuse)
```

**Pros:**

- Minimal new code - reuses proven scmcallib
- Fastest path to working calibration
- Scientists already familiar with scmcallib API
- All existing visualization/diagnostics work

**Cons:**

- Depends on external Python library (scmcallib)
- Limited control over optimization internals
- Python orchestration overhead remains
- Not a long-term maintainable solution if scmcallib becomes unmaintained

**Estimated scope:** Small

## Open Questions

1. **Performance requirements**: How many model evaluations are typical? (1,000? 100,000? 1,000,000?) This affects whether Python boundary overhead matters.

2. **MCMC sampler preference**: Is emcee's affine-invariant ensemble sampler the required algorithm, or are alternatives (Metropolis-Hastings, NUTS, etc.) acceptable?

3. **Parameter injection mechanism**: Should we support modifying component parameters without rebuilding the model, or is rebuild-per-evaluation acceptable?

4. **Visualization**: Are Python-based plots (matplotlib, corner) acceptable, or do you need Rust-native plotting?

5. **Persistence format**: HDF5 for chain storage (like emcee), or alternatives (Parquet, Arrow, custom binary)?

6. **scmcallib maintenance**: Is scmcallib actively maintained? Should RSCM aim to replace it or integrate with it?

7. **Multi-objective calibration**: Is weighted multi-variable calibration (like scmcallib's `weights` parameter) required from day one?

## User Decisions

Based on discussion with the user:

1. **Performance**: Millions of model evaluations expected (20M+ MCMC steps) - pure Rust is essential
2. **MCMC sampler**: Implement emcee's affine-invariant ensemble sampler in Rust (algorithm from Goodman & Weare 2010)
3. **Visualization**: Python-based (matplotlib, corner) - Rust only for computation
4. **scmcallib**: Not maintained, we're replacing it - calibration should be first-class in RSCM
5. **Posterior sub-sampling**: Deferred to later - not in initial scope
6. **API style**: Extensible design with Distribution objects, explicit Likelihood, first-class Diagnostics (not simplified emcee-clone)

## Chosen Approach: B (Pure Rust Calibration Library)

**Rationale:**

1. **Performance-critical**: Millions of evaluations require minimising overhead
2. **First-class citizen**: Calibration should be core RSCM functionality, not a Python add-on
3. **Algorithm simplicity**: The emcee stretch move algorithm is straightforward to implement
4. **Replaces scmcallib**: Provides modern, maintained alternative

## Stretch Move Algorithm (from emcee source)

The affine-invariant ensemble sampler uses the stretch move:

1. **Sample z** from proposal distribution:
   ```
   z = ((a - 1) * U + 1)² / a
   ```
   where U ~ Uniform(0,1), a = 2.0 (default stretch scale)

2. **Select complementary walker** c_j uniformly from the complementary ensemble

3. **Propose new position**:
   ```
   y = c_j + z * (x_i - c_j)
   ```

4. **Accept with probability**:
   ```
   min(1, z^(n-1) * p(y) / p(x_i))
   ```
   where n is the dimensionality, p is the target density

Reference: Goodman, J. & Weare, J. (2010). "Ensemble samplers with affine invariance." Communications in Applied Mathematics and Computational Science, 5(1), 65-80.

# Design: Calibration Framework Architecture

## Context

RSCM needs calibration infrastructure that can handle millions of model evaluations efficiently. The system must:

- Run entirely in Rust for performance (no Python in hot loop)
- Expose Python API for scientific workflows
- Support both point estimation (single best fit) and MCMC (posterior distribution)
- Enable parallel model execution
- Persist chains for analysis and resumption

**Constraints:**

- Components are immutable after construction (rebuild per parameter set)
- Model serialization already works (TOML/JSON)
- Python bindings via PyO3 exist

## Decision

Implement a new `rscm-calibrate` crate with the following architecture:

```
rscm-calibrate/
├── src/
│   ├── lib.rs
│   ├── distribution.rs      # Prior distributions
│   ├── parameter.rs         # ParameterSet, bounds
│   ├── sampler/
│   │   ├── mod.rs
│   │   ├── ensemble.rs      # Affine-invariant ensemble sampler
│   │   └── chain.rs         # Chain storage and management
│   ├── optimizer/
│   │   ├── mod.rs
│   │   └── point.rs         # Point estimation via argmin
│   ├── likelihood.rs        # Likelihood functions, metrics
│   ├── target.rs            # Observational target handling
│   ├── runner.rs            # Parallel model execution
│   ├── diagnostics.rs       # Convergence metrics (R-hat, ESS)
│   └── python/
│       └── mod.rs           # PyO3 bindings
```

### Core Abstractions

**1. Distribution Trait**

```rust
pub trait Distribution: Send + Sync {
    /// Sample n values from the distribution
    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<f64>;

    /// Log probability density at x
    fn ln_pdf(&self, x: f64) -> f64;

    /// Bounds (if any)
    fn bounds(&self) -> (Option<f64>, Option<f64>);
}
```

Implementations: `Uniform`, `Normal`, `Bound<D>`, `LogNormal`, `TruncatedNormal`

**2. ParameterSet**

```rust
pub struct ParameterSet {
    parameters: IndexMap<String, ParameterDef>,
}

pub struct ParameterDef {
    pub name: String,
    pub prior: Arc<dyn Distribution>,
    pub transform: Option<Transform>,  // e.g., log transform
}

impl ParameterSet {
    pub fn add(&mut self, name: &str, prior: impl Distribution);
    pub fn sample(&self, n: usize, rng: &mut impl Rng) -> Array2<f64>;
    pub fn ln_prior(&self, theta: &[f64]) -> f64;
    pub fn names(&self) -> Vec<&str>;
    pub fn bounds(&self) -> Vec<(Option<f64>, Option<f64>)>;
}
```

**3. Target (Observations)**

```rust
pub struct Target {
    /// Variable name -> (times, values, uncertainties)
    observations: HashMap<String, Observation>,
    reference_period: Option<(f64, f64)>,
}

pub struct Observation {
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub uncertainties: Vec<f64>,  // Standard deviations
}

impl Target {
    pub fn add_variable(&mut self, name: &str, times: &[f64], values: &[f64], uncertainties: &[f64]);
    pub fn with_reference_period(self, start: f64, end: f64) -> Self;
}
```

**4. ModelRunner Trait**

```rust
pub trait ModelRunner: Send + Sync {
    /// Parameter names in order (defines indexing for run/run_batch)
    fn param_names(&self) -> &[String];

    /// Run model with parameters indexed by param_names() order
    /// Returns Err for model failures (solver divergence, invalid state)
    fn run(&self, params: &[f64]) -> RSCMResult<ModelOutput>;

    /// Run multiple parameter sets in parallel
    fn run_batch(&self, params: &[Vec<f64>]) -> Vec<RSCMResult<ModelOutput>>;
}

pub struct ModelOutput {
    pub variables: HashMap<String, Timeseries>,
}
```

Default implementation wraps Model with rayon parallelism. Using `&[f64]` indexed by parameter order avoids HashMap allocation and string hashing overhead in the hot loop (millions of evaluations).

**5. LikelihoodFn Trait**

```rust
pub trait LikelihoodFn: Send + Sync {
    /// Compute log-likelihood of model output against target observations.
    /// Receives Result to handle model failures gracefully (returns -INFINITY).
    fn ln_likelihood(&self, model_output: &RSCMResult<ModelOutput>, target: &Target) -> f64;
}

/// Gaussian likelihood with per-observation uncertainties
pub struct GaussianLikelihood {
    pub relative_error: Option<f64>,  // e.g., 0.05 for 5%
}

impl LikelihoodFn for GaussianLikelihood {
    fn ln_likelihood(&self, model_output: &RSCMResult<ModelOutput>, target: &Target) -> f64 {
        match model_output {
            Ok(output) => self.compute_gaussian_ll(output, target),
            Err(_) => f64::NEG_INFINITY,  // Failed runs have zero probability
        }
    }
}
```

**6. EnsembleSampler**

Implements Goodman & Weare (2010) affine-invariant ensemble sampler.

```rust
pub struct EnsembleSampler<R: ModelRunner, L: LikelihoodFn> {
    parameters: ParameterSet,
    target: Target,
    runner: R,
    likelihood: L,
    n_walkers: usize,        // Default: max(2 * n_params, 32)
    stretch_scale: f64,      // 'a' parameter, default 2.0
}

pub struct SamplerState {
    pub positions: Array2<f64>,  // (n_walkers, n_params)
    pub log_probs: Array1<f64>,  // (n_walkers,)
    pub iteration: usize,        // Current iteration count
}

pub struct Chain {
    pub samples: Array3<f64>,    // (n_iterations, n_walkers, n_params)
    pub log_probs: Array2<f64>,  // (n_iterations, n_walkers)
    pub acceptance_fractions: Vec<f64>,
    pub param_names: Vec<String>,
}

/// Options for sampler execution
pub struct RunOptions {
    pub thin: usize,                      // Store every Nth sample (default: 1)
    pub checkpoint_every: Option<usize>,  // Save state every N steps
    pub checkpoint_path: Option<PathBuf>, // Path for checkpoint files
}

impl<R, L> EnsembleSampler<R, L> {
    /// Create sampler with default n_walkers = max(2 * n_params, 32)
    pub fn new(params: ParameterSet, target: Target, runner: R, likelihood: L) -> Self;

    /// Create sampler with explicit walker count
    pub fn with_n_walkers(self, n_walkers: usize) -> Self;

    /// Initialize walker positions
    pub fn initialize(&mut self, method: InitMethod) -> SamplerState;

    /// Run n_steps iterations, returning the chain
    pub fn run(&mut self, state: SamplerState, n_steps: usize, options: RunOptions) -> Chain;

    /// Run with progress callback (called each iteration)
    pub fn run_with_progress<F>(
        &mut self,
        state: SamplerState,
        n_steps: usize,
        options: RunOptions,
        callback: F,
    ) -> Chain
    where
        F: FnMut(&ProgressInfo);

    /// Resume from checkpoint file
    pub fn resume(&mut self, checkpoint_path: &Path, n_steps: usize, options: RunOptions) -> Chain;
}

/// Progress information passed to callback
pub struct ProgressInfo {
    pub iteration: usize,
    pub total_iterations: usize,
    pub acceptance_rate: f64,
    pub mean_log_prob: f64,
}

pub enum InitMethod {
    /// Sample from prior
    Prior,
    /// Ball around a point (useful for warm starts from point estimate)
    Ball { center: Vec<f64>, radius: f64 },
    /// Explicit positions (n_walkers x n_params)
    Explicit(Array2<f64>),
}
```

**7. PointEstimator**

```rust
pub struct PointEstimator<R: ModelRunner, L: LikelihoodFn> {
    parameters: ParameterSet,
    target: Target,
    runner: R,
    likelihood: L,
}

pub struct PointEstimateResult {
    pub best_params: HashMap<String, f64>,
    pub best_likelihood: f64,
    pub samples: Array2<f64>,       // All evaluated points
    pub likelihoods: Array1<f64>,   // Corresponding likelihoods
}

impl<R, L> PointEstimator<R, L> {
    /// Find best fit using specified optimizer
    pub fn optimize(&self, method: OptimizeMethod, options: OptimizeOptions) -> PointEstimateResult;
}

pub enum OptimizeMethod {
    LBFGS,
    NelderMead,
    ParticleSwarm,
    RandomSearch { n_samples: usize },
}
```

**8. Chain Methods (including Diagnostics)**

Diagnostics are methods on Chain for better discoverability and ergonomics:

```rust
impl Chain {
    /// Get flat samples array with burn-in removed
    /// Returns (n_iterations - discard) * n_walkers rows
    pub fn flat_samples(&self, discard: usize) -> Array2<f64>;

    /// Gelman-Rubin potential scale reduction factor (R-hat < 1.1 indicates convergence)
    pub fn r_hat(&self, discard: usize) -> HashMap<String, f64>;

    /// Effective sample size (accounts for autocorrelation)
    pub fn ess(&self, discard: usize) -> HashMap<String, f64>;

    /// Acceptance fraction per walker (healthy range: 0.2-0.5)
    pub fn acceptance_fraction(&self) -> Vec<f64>;

    /// Mean acceptance rate across all walkers
    pub fn mean_acceptance_rate(&self) -> f64;

    /// Autocorrelation time estimate per parameter
    pub fn autocorr_time(&self, discard: usize) -> HashMap<String, f64>;

    /// Check if chain has converged (all R-hat < threshold, default 1.1)
    pub fn is_converged(&self, discard: usize, r_hat_threshold: Option<f64>) -> bool;
}
```

### Parallel Execution Strategy

The ensemble sampler naturally parallelizes:

1. Split walkers into two groups (even/odd indices)
2. For each group, propose moves using complementary group
3. Evaluate all proposals in parallel via rayon
4. Accept/reject independently

```rust
fn stretch_move(&self, state: &mut SamplerState, rng: &mut impl Rng) {
    let (s0, s1) = split_walkers(&state.positions);

    // Update first half using second half as complement
    let proposals_0 = self.propose_stretch(&s0, &s1, rng);
    let log_probs_0: Vec<f64> = proposals_0.par_iter()
        .map(|p| self.evaluate_log_prob(p))
        .collect();
    self.accept_reject(&mut s0, proposals_0, log_probs_0, rng);

    // Update second half using first half as complement
    let proposals_1 = self.propose_stretch(&s1, &s0, rng);
    let log_probs_1: Vec<f64> = proposals_1.par_iter()
        .map(|p| self.evaluate_log_prob(p))
        .collect();
    self.accept_reject(&mut s1, proposals_1, log_probs_1, rng);
}
```

### Python API

Modern, fluent API with multiple construction styles:

```python
from rscm.calibrate import (
    ParameterSet, Uniform, Normal, Bound, LogNormal,
    Target, GaussianLikelihood,
    EnsembleSampler, PointEstimator, ModelRunner,
)

# === Parameter Definition ===

# Option A: Dict-based construction (recommended for most cases)
params = ParameterSet({
    "climate_sensitivity": Bound(Normal(3.0, 0.5), lower=1.5, upper=6.0),
    "ocean_heat_uptake": Uniform(0.5, 2.0),
    "aerosol_forcing": Normal(-1.0, 0.5),
})

# Option B: Fluent builder (for dynamic construction)
params = (ParameterSet()
    .add("climate_sensitivity", Bound(Normal(3.0, 0.5), lower=1.5, upper=6.0))
    .add("ocean_heat_uptake", Uniform(0.5, 2.0)))

# === Target Definition ===

# From arrays
target = (Target()
    .add_variable("Surface Temperature", times, values, uncertainties)
    .with_reference_period(1850, 1900))

# From pandas DataFrame
target = Target.from_dataframe(
    df,
    time_col="year",
    variables={"Temperature": "gmst", "OHC": "ocean_heat"},
    uncertainty_col="uncertainty",  # Or use relative_error=0.05
    reference_period=(1850, 1900),
)

# From scmdata (for compatibility with existing workflows)
target = Target.from_scmdata(obs_run, reference_period=(1850, 1900))

# === Model Runner ===

# Define how parameters map to model construction
def make_model(params: dict[str, float]) -> Model:
    return (ModelBuilder()
        .with_component(TwoLayerComponent(
            climate_sensitivity=params["climate_sensitivity"],
            ocean_heat_uptake=params["ocean_heat_uptake"],
        ))
        .build())

runner = ModelRunner(
    model_factory=make_model,
    output_variables=["Surface Temperature", "Ocean Heat Content"],
)

# === Point Estimation ===

estimator = PointEstimator(params, target, runner, GaussianLikelihood(relative_error=0.05))
result = estimator.optimize(method="L-BFGS", n_restarts=10)
print(f"Best fit: {result.best_params}")
print(f"Log-likelihood: {result.best_likelihood}")

# === MCMC Sampling ===

sampler = EnsembleSampler(
    parameters=params,
    target=target,
    runner=runner,
    likelihood=GaussianLikelihood(relative_error=0.05),
    # n_walkers defaults to max(2 * n_params, 32)
)

# Initialize and run with progress bar
state = sampler.initialize(method="prior")
chain = sampler.run(
    state,
    n_steps=10_000,
    thin=100,
    progress=True,  # Shows tqdm progress bar
    checkpoint_every=1000,
    checkpoint_path="chain_checkpoint.bin",
)

# Resume from checkpoint if interrupted
chain = sampler.resume("chain_checkpoint.bin", n_steps=5000)

# === Diagnostics (methods on Chain) ===

print(f"R-hat: {chain.r_hat(discard=1000)}")
print(f"ESS: {chain.ess(discard=1000)}")
print(f"Acceptance rate: {chain.mean_acceptance_rate():.2%}")
print(f"Converged: {chain.is_converged(discard=1000)}")

# === Data Export ===

# Flat numpy array
samples = chain.flat_samples(discard=1000)  # (n_samples, n_params)

# Pandas DataFrame with parameter names as columns
df = chain.to_dataframe(discard=1000)
print(df.describe())

# === Visualization (via plot namespace) ===

chain.plot.trace(discard=1000)           # Trace plots for all parameters
chain.plot.corner(discard=1000)          # Corner plot (requires corner library)
chain.plot.autocorr()                    # Autocorrelation plots
chain.plot.acceptance()                  # Acceptance rate over time

# Posterior predictive check
chain.plot.posterior_predictive(
    runner,
    target,
    n_samples=100,
    discard=1000,
)
```

### Chain Persistence

Use a simple binary format for chain storage:

```rust
pub struct ChainStorage {
    path: PathBuf,
}

impl ChainStorage {
    pub fn save(&self, chain: &Chain) -> io::Result<()>;
    pub fn load(&self) -> io::Result<Chain>;
    pub fn append(&self, chain: &Chain) -> io::Result<()>;  // For resumption
}
```

Format: Header (n_iterations, n_walkers, n_params, param_names) + raw f64 arrays.

Python can read via numpy memory mapping for large chains.

## Alternatives Rejected

**A. Python orchestration with Rust batch execution**

- Rejected because millions of evaluations would still cross Python/Rust boundary thousands of times
- MCMC state management in Python adds complexity

**C. scmcallib adapter**

- Rejected because scmcallib is unmaintained
- Doesn't provide long-term maintainability

**Using existing Rust MCMC crates (emcee, rmcmc)**

- Evaluated but found less mature than implementing ourselves
- Our specific needs (tight model integration, parallel batch evaluation) are easier to optimise with custom implementation

## Trade-offs Accepted

1. **Implementation effort**: Implementing MCMC from scratch requires careful testing, but the algorithm is well-documented and straightforward.

2. **No built-in plotting**: Rust doesn't have matplotlib-quality plotting. Accepted: export to Python for visualization via plot namespace.

3. **Model rebuild per evaluation**: Components are immutable, so we rebuild models for each parameter set. **Action required**: Benchmark model construction overhead early in implementation to validate this assumption. If construction exceeds ~10μs, consider parameter injection mechanism.

4. **Parallel determinism**: Thread-local RNGs with deterministic seeding (e.g., seed + thread_id) provide reproducibility for debugging. Document that exact reproducibility requires single-threaded mode.

## Performance Considerations

1. **Indexed parameters over HashMap**: The `ModelRunner::run(&[f64])` interface uses positional indexing rather than `HashMap<String, f64>`. This avoids string hashing and allocation in the hot loop. With 20M+ evaluations, even 100ns overhead per call = 2 seconds total.

2. **Walker count scaling**: Default `n_walkers = max(2 * n_params, 32)` ensures adequate exploration for high-dimensional posteriors while avoiding excessive overhead for simple cases.

3. **Checkpoint format**: Binary with raw f64 arrays for fast I/O. Memory-mapped reading enables analysis of large chains without loading entirely into RAM.

## Dependencies

New Rust dependencies for `rscm-calibrate`:

- `rayon` - Parallel iteration (already in workspace)
- `rand` / `rand_distr` - Random sampling
- `ndarray` - Array operations (already in workspace)
- `argmin` - Optimization algorithms
- `statrs` - Statistical distributions (for ln_pdf)

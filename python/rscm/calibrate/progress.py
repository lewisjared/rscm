"""
Progress reporting utilities for calibration workflows.

Provides integration with tqdm and other progress bar libraries for
tracking long-running MCMC sampling and optimization tasks.
"""


def create_tqdm_callback(total: int, desc: str = "Sampling", **tqdm_kwargs):
    """
    Create a progress callback that updates a tqdm progress bar.

    This function creates a callback compatible with
    `EnsembleSampler.run_with_progress()` that displays a tqdm progress bar
    during MCMC sampling.

    Parameters
    ----------
    total : int
        Total number of iterations expected
    desc : str, optional
        Description to display in the progress bar (default: "Sampling")
    **tqdm_kwargs
        Additional keyword arguments to pass to tqdm constructor
        (e.g., position=0, leave=True, unit="iter")

    Returns
    -------
    callable
        A callback function that can be passed to run_with_progress()

    Examples
    --------
    >>> from rscm.calibrate import EnsembleSampler, WalkerInit
    >>> from rscm.calibrate.progress import create_tqdm_callback
    >>>
    >>> # Create sampler (assumes params, runner, likelihood, target defined)
    >>> sampler = EnsembleSampler(params, runner, likelihood, target)
    >>>
    >>> # Create progress callback
    >>> callback = create_tqdm_callback(total=1000, desc="MCMC Sampling")
    >>>
    >>> # Run with progress bar
    >>> chain = sampler.run_with_progress(
    ...     n_iterations=1000, init=WalkerInit.from_prior(), callback=callback
    ... )

    Notes
    -----
    The callback displays:
    - Current iteration / total iterations
    - Percentage complete
    - Estimated time remaining
    - Mean acceptance rate
    - Mean log probability

    For Jupyter notebooks, tqdm will automatically use the notebook widget
    interface for a richer display.
    """
    try:
        from tqdm.auto import tqdm  # noqa: PLC0415
    except ImportError:
        msg = (
            "tqdm is required for progress bar display. Install with: pip install tqdm"
        )
        raise ImportError(msg) from None

    # Set default tqdm kwargs
    defaults = {"desc": desc, "total": total, "unit": "iter"}
    defaults.update(tqdm_kwargs)

    pbar = tqdm(**defaults)

    def callback(progress_info):
        """Update tqdm progress bar with current progress info."""
        # Update to current iteration
        pbar.n = progress_info.iteration + 1  # +1 because iteration is 0-indexed

        # Update postfix with diagnostic info
        pbar.set_postfix(
            {
                "acc_rate": f"{progress_info.acceptance_rate:.3f}",
                "mean_log_p": f"{progress_info.mean_log_prob:.2f}",
            },
            refresh=True,
        )

    # Close progress bar when callback is garbage collected
    # Store pbar as attribute so it's accessible for manual closing if needed
    callback.pbar = pbar
    callback.close = pbar.close

    return callback


def create_simple_callback(print_every: int = 100):
    """
    Create a simple text-based progress callback.

    This callback prints progress information to stdout at regular intervals.
    Useful when tqdm is not available or not desired.

    Parameters
    ----------
    print_every : int, optional
        Print progress every N iterations (default: 100)

    Returns
    -------
    callable
        A callback function that can be passed to run_with_progress()

    Examples
    --------
    >>> from rscm.calibrate.progress import create_simple_callback
    >>>
    >>> callback = create_simple_callback(print_every=50)
    >>> chain = sampler.run_with_progress(
    ...     n_iterations=1000, init=WalkerInit.from_prior(), callback=callback
    ... )
    """

    def callback(progress_info):
        """Print progress information at regular intervals."""
        iteration = progress_info.iteration
        total = progress_info.total

        # Print at specified intervals or on last iteration
        if (iteration + 1) % print_every == 0 or iteration + 1 == total:
            pct = 100.0 * (iteration + 1) / total
            print(
                f"Iteration {iteration + 1}/{total} ({pct:.1f}%) | "
                f"Acceptance rate: {progress_info.acceptance_rate:.3f} | "
                f"Mean log prob: {progress_info.mean_log_prob:.2f}"
            )

    return callback


class ProgressTracker:
    """
    Track calibration progress with metrics history.

    This class provides a callback that stores progress metrics over time,
    useful for later analysis or custom visualization.

    Parameters
    ----------
    print_every : int, optional
        If > 0, print progress every N iterations (default: 0, no printing)

    Attributes
    ----------
    iterations : list of int
        Iteration numbers
    acceptance_rates : list of float
        Acceptance rates at each stored iteration
    mean_log_probs : list of float
        Mean log probabilities at each stored iteration

    Examples
    --------
    >>> from rscm.calibrate.progress import ProgressTracker
    >>>
    >>> tracker = ProgressTracker(print_every=100)
    >>> chain = sampler.run_with_progress(
    ...     n_iterations=1000, init=WalkerInit.from_prior(), callback=tracker
    ... )
    >>>
    >>> # Plot acceptance rate over time
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(tracker.iterations, tracker.acceptance_rates)
    >>> plt.xlabel("Iteration")
    >>> plt.ylabel("Acceptance Rate")
    >>> plt.show()
    """

    def __init__(self, print_every: int = 0):
        self.print_every = print_every
        self.iterations = []
        self.acceptance_rates = []
        self.mean_log_probs = []

    def __call__(self, progress_info):
        """Store progress metrics and optionally print."""
        self.iterations.append(progress_info.iteration)
        self.acceptance_rates.append(progress_info.acceptance_rate)
        self.mean_log_probs.append(progress_info.mean_log_prob)

        if self.print_every > 0:
            iteration = progress_info.iteration
            total = progress_info.total
            if (iteration + 1) % self.print_every == 0 or iteration + 1 == total:
                pct = 100.0 * (iteration + 1) / total
                print(
                    f"Iteration {iteration + 1}/{total} ({pct:.1f}%) | "
                    f"Acceptance rate: {progress_info.acceptance_rate:.3f} | "
                    f"Mean log prob: {progress_info.mean_log_prob:.2f}"
                )

    def clear(self):
        """Clear all stored metrics."""
        self.iterations.clear()
        self.acceptance_rates.clear()
        self.mean_log_probs.clear()

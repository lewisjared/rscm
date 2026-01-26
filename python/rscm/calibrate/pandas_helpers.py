"""
Pandas integration helpers for the calibration framework.

These functions provide convenient conversions between calibration objects
and pandas DataFrames for analysis and visualization.
"""

import numpy as np
import pandas as pd


def chain_to_dataframe(chain, discard=0):
    """
    Convert a Chain to a pandas DataFrame.

    Creates a DataFrame with parameter columns and log_prob, using a
    multi-index (walker, iteration) for the rows. This format is useful
    for analyzing walker behaviour and creating trace plots.

    Parameters
    ----------
    chain : Chain
        The MCMC chain to convert
    discard : int, optional
        Number of initial samples to discard as burn-in (default: 0)

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - Index: MultiIndex with levels (walker, iteration)
        - Columns: One column per parameter plus 'log_prob'

    Examples
    --------
    >>> df = chain_to_dataframe(chain, discard=100)
    >>> # Plot trace for first parameter
    >>> import matplotlib.pyplot as plt
    >>> for walker in df.index.get_level_values(0).unique():
    ...     walker_data = df.xs(walker, level="walker")
    ...     plt.plot(walker_data.index, walker_data["param1"])
    >>> plt.xlabel("Iteration")
    >>> plt.ylabel("param1")
    >>> plt.show()

    Notes
    -----
    This function creates a long-form DataFrame where each walker's samples
    are kept separate. For a flattened view of all samples, use
    `chain.flat_samples(discard)` directly.
    """
    # Get parameter names
    param_names = chain.param_names

    # Get raw chain data (before flattening)
    # We need to access the underlying data structure to preserve walker separation
    # Since PyChain doesn't expose raw samples, we'll work with flat_samples
    # and reconstruct the structure
    flat_samples = chain.flat_samples(discard)
    flat_log_probs = chain.flat_log_probs(discard)

    # Infer number of walkers from chain structure
    # total_iterations tells us how many iterations were run
    # len() tells us stored samples after thinning
    # flat_samples gives us (n_stored * n_walkers, n_params)
    n_stored = chain.__len__() - discard  # Stored samples per walker after discard
    n_params = len(param_names)
    n_total_flat = flat_samples.shape[0]
    n_walkers = n_total_flat // n_stored if n_stored > 0 else 0

    if n_walkers == 0 or n_stored == 0:
        # Empty chain
        return pd.DataFrame(columns=[*param_names, "log_prob"])

    # Reshape flat arrays back to (n_walkers, n_stored, n_params)
    # flat_samples is (n_walkers * n_stored, n_params) in walker-major order
    samples = flat_samples.reshape(n_walkers, n_stored, n_params)
    log_probs = flat_log_probs.reshape(n_walkers, n_stored)

    # Build multi-index DataFrame
    # Create index arrays
    walker_idx = np.repeat(np.arange(n_walkers), n_stored)
    # Iteration indices start from discard and account for thinning
    thin = chain.thin
    iter_idx = np.tile(np.arange(discard, discard + n_stored * thin, thin), n_walkers)

    # Flatten samples to match index
    samples_flat = samples.reshape(-1, n_params)
    log_probs_flat = log_probs.reshape(-1)

    # Create DataFrame
    data = {name: samples_flat[:, i] for i, name in enumerate(param_names)}
    data["log_prob"] = log_probs_flat

    df = pd.DataFrame(
        data,
        index=pd.MultiIndex.from_arrays(
            [walker_idx, iter_idx], names=["walker", "iteration"]
        ),
    )

    return df


def target_from_dataframe(
    df, time_col="time", value_col="value", uncertainty_col=None, relative_error=None
):
    """
    Create a Target from a pandas DataFrame.

    This is a convenience constructor for creating Target objects from
    tabular data in pandas format.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing observations. Should have columns for time,
        values, and optionally uncertainties. If the DataFrame has a
        'variable' column, it will be used to group observations by variable.
        Otherwise all data is treated as a single variable (must specify
        variable name in index or pass as single group).
    time_col : str, optional
        Name of the time column (default: 'time')
    value_col : str, optional
        Name of the value column (default: 'value')
    uncertainty_col : str, optional
        Name of the uncertainty column (default: None). If None and
        relative_error is None, uncertainties are required to be in the
        DataFrame with column name 'uncertainty'.
    relative_error : float, optional
        If specified, use this relative error (as a fraction) to compute
        uncertainties as `relative_error * abs(value)`. This overrides
        uncertainty_col. (default: None)

    Returns
    -------
    Target
        Target object containing the observations from the DataFrame

    Examples
    --------
    >>> # Single variable with absolute uncertainties
    >>> df = pd.DataFrame(
    ...     {
    ...         "time": [1850, 1900, 1950, 2000],
    ...         "value": [0.0, 0.2, 0.5, 1.0],
    ...         "uncertainty": [0.1, 0.1, 0.1, 0.15],
    ...     }
    ... )
    >>> target = Target.from_dataframe(df)
    >>> target.add_variable("Temperature|Global", df)

    >>> # Multiple variables
    >>> df = pd.DataFrame(
    ...     {
    ...         "variable": [
    ...             "Temperature|Global",
    ...             "Temperature|Global",
    ...             "OHC|Total",
    ...             "OHC|Total",
    ...         ],
    ...         "time": [1950, 2000, 1950, 2000],
    ...         "value": [0.5, 1.0, 100, 200],
    ...         "uncertainty": [0.1, 0.15, 20, 30],
    ...     }
    ... )
    >>> target = Target.from_dataframe(df)

    >>> # Using relative error
    >>> df = pd.DataFrame(
    ...     {
    ...         "time": [1850, 1900, 1950, 2000],
    ...         "value": [0.0, 0.2, 0.5, 1.0],
    ...     }
    ... )
    >>> target = Target.from_dataframe(df, relative_error=0.1)  # 10% error

    Notes
    -----
    This function is currently a placeholder and needs to be implemented
    based on the specific Target API requirements. The implementation should:

    1. Check if 'variable' column exists for multi-variable data
    2. Group by variable if present
    3. For each group/variable:
       - Extract time, value arrays
       - Compute or extract uncertainties
       - Create Observation objects
       - Add to VariableTarget
    4. Construct and return Target

    Raises
    ------
    ValueError
        If required columns are missing or data format is invalid
    """
    from rscm._lib.calibrate import Target  # noqa: PLC0415

    target = Target()

    # Check if multi-variable format
    if "variable" in df.columns:
        # Group by variable
        for var_name, var_df in df.groupby("variable"):
            _add_variable_from_df(
                target,
                str(var_name),
                var_df,
                time_col,
                value_col,
                uncertainty_col,
                relative_error,
            )
    else:
        # Single variable - need variable name
        # For now, raise error - user should specify variable explicitly
        msg = (
            "DataFrame must have 'variable' column for automatic variable detection. "
            "For single-variable data, create Target manually:\n"
            "  target = Target()\n"
            "  target.add_variable(variable_name, observations)"
        )
        raise ValueError(msg)

    return target


def _add_variable_from_df(  # noqa: PLR0913
    target, var_name, df, time_col, value_col, uncertainty_col, relative_error
):
    """Add a single variable from a DataFrame to a Target."""
    times = df[time_col].values
    values = df[value_col].values

    # Determine uncertainties
    if relative_error is not None:
        # Use relative error - add observations with relative uncertainty
        for t, v in zip(times, values):
            target.add_observation_relative(
                var_name, float(t), float(v), float(relative_error)
            )
    else:
        # Use absolute uncertainties
        if uncertainty_col is not None:
            uncertainties = df[uncertainty_col].values
        elif "uncertainty" in df.columns:
            uncertainties = df["uncertainty"].values
        else:
            msg = (
                f"No uncertainty information provided for variable '{var_name}'. "
                "Specify uncertainty_col or relative_error parameter."
            )
            raise ValueError(msg)

        # Add observations with absolute uncertainty
        for t, v, u in zip(times, values, uncertainties):
            target.add_observation(var_name, float(t), float(v), float(u))

    return target

"""
Pandas integration helpers for the calibration framework.

These functions provide convenient conversions between calibration objects
and pandas DataFrames for analysis and visualisation.
"""

import numpy as np
import pandas as pd


def chain_to_dataframe(chain, discard=0):
    """
    Convert a Chain to a pandas DataFrame.

    Creates a DataFrame with parameter columns and log_prob, using a
    multi-index (walker, iteration) for the rows. This format is useful
    for analysing walker behaviour and creating trace plots.

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

        - Index: MultiIndex with levels (walker, iteration).
          `iteration` is the 0-based sampler iteration, stepping by `thin`.
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
    parameter_names = chain.parameter_names
    flat_samples = chain.flat_samples(discard)
    flat_log_probs = chain.flat_log_probs(discard)

    n_samples = len(chain) - discard
    n_parameters = len(parameter_names)
    n_walkers = flat_samples.shape[0] // n_samples if n_samples > 0 else 0

    if n_walkers == 0 or n_samples == 0:
        return pd.DataFrame(columns=[*parameter_names, "log_prob"])

    # Flat arrays are iteration-major: row i * n_walkers + j is walker j at sample i.
    samples = flat_samples.reshape(n_samples, n_walkers, n_parameters).transpose(
        1, 0, 2
    )
    log_probs = flat_log_probs.reshape(n_samples, n_walkers).T

    walker_idx = np.repeat(np.arange(n_walkers), n_samples)
    # The chain keeps every thin-th iteration: stored sample k is (k + 1) * thin - 1.
    stored_idx = np.arange(discard, discard + n_samples)
    iter_idx = np.tile((stored_idx + 1) * chain.thin - 1, n_walkers)

    samples_flat = samples.reshape(-1, n_parameters)
    log_probs_flat = log_probs.reshape(-1)

    # Create DataFrame
    data = {name: samples_flat[:, i] for i, name in enumerate(parameter_names)}
    data["log_prob"] = log_probs_flat

    df = pd.DataFrame(
        data,
        index=pd.MultiIndex.from_arrays(
            [walker_idx, iter_idx], names=["walker", "iteration"]
        ),
    )

    return df


def target_from_dataframe(  # noqa: PLR0913
    df,
    variable=None,
    time_col="time",
    value_col="value",
    uncertainty_col=None,
    relative_error=None,
):
    """
    Create a Target from a pandas DataFrame.

    Each row is one observation.
    Columns may also be index levels, including levels of a MultiIndex.
    This long layout differs from the wide one of `rscm.interop.to_dataframe`,
    see the Pandas and xarray guide to convert between them.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations with time and value columns and,
        unless `relative_error` is given, uncertainties.
        A `variable` column groups rows by variable.
        Without one, pass the variable name as `variable`.
    variable : str, optional
        Variable name for every row, for a DataFrame without a `variable` column
    time_col : str, optional
        Name of the time column (default: 'time').
        Times must be numeric, such as decimal years.
    value_col : str, optional
        Name of the value column (default: 'value')
    uncertainty_col : str, optional
        Name of the 1-sigma uncertainty column (default: 'uncertainty')
    relative_error : float, optional
        If given, each uncertainty is `relative_error * abs(value)`
        and any uncertainty column is ignored.
        A zero value then has no uncertainty and is rejected.

    Returns
    -------
    Target
        Target object containing the observations from the DataFrame

    Raises
    ------
    ValueError
        If a required column is missing, `variable` is needed but not given,
        a `variable` entry is missing, or an observation is invalid
        (non-finite, or with a non-positive uncertainty)
    TypeError
        If the time column is not numeric

    Examples
    --------
    >>> import pandas as pd
    >>> from rscm.calibrate import Target
    >>> # Single variable with absolute uncertainties
    >>> df = pd.DataFrame(
    ...     {
    ...         "time": [1850, 1900, 1950, 2000],
    ...         "value": [0.0, 0.2, 0.5, 1.0],
    ...         "uncertainty": [0.1, 0.1, 0.1, 0.15],
    ...     }
    ... )
    >>> target = Target.from_dataframe(df, variable="Surface Air Temperature Change")

    >>> # Multiple variables
    >>> df = pd.DataFrame(
    ...     {
    ...         "variable": [
    ...             "Surface Air Temperature Change",
    ...             "Surface Air Temperature Change",
    ...             "Ocean Heat Content",
    ...             "Ocean Heat Content",
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
    ...         "value": [0.1, 0.2, 0.5, 1.0],
    ...     }
    ... )
    >>> target = Target.from_dataframe(df, variable="y", relative_error=0.1)
    """
    from rscm._lib.calibrate import Target  # noqa: PLC0415

    if relative_error is None:
        uncertainty_col = uncertainty_col or "uncertainty"
    else:
        uncertainty_col = None

    columns = [time_col, value_col] + ([uncertainty_col] if uncertainty_col else [])
    index_levels = [name for name in df.index.names if name is not None]
    df = df.reset_index(level=index_levels) if index_levels else df

    missing = [column for column in columns if column not in df.columns]
    if missing:
        hint = "" if relative_error is not None else ", or pass relative_error"
        msg = f"DataFrame is missing column(s) {missing}{hint}"
        raise ValueError(msg)

    if not pd.api.types.is_numeric_dtype(df[time_col]):
        msg = (
            f"Column '{time_col}' must be numeric (e.g. decimal years), "
            f"got {df[time_col].dtype}"
        )
        raise TypeError(msg)

    if "variable" in df.columns:
        if variable is not None:
            msg = "Pass either a 'variable' column or variable=..., not both"
            raise ValueError(msg)
        n_missing = int(df["variable"].isna().sum())
        if n_missing:
            msg = f"{n_missing} row(s) have a missing variable"
            raise ValueError(msg)
        groups = df.groupby("variable", sort=False)
    elif variable is not None:
        groups = [(variable, df)]
    else:
        msg = (
            "DataFrame has no 'variable' column: pass the variable name as variable=..."
        )
        raise ValueError(msg)

    target = Target()
    for var_name, var_df in groups:
        _add_variable_from_df(
            target,
            str(var_name),
            var_df,
            time_col,
            value_col,
            uncertainty_col,
            relative_error,
        )
    return target


def _add_variable_from_df(  # noqa: PLR0913
    target, var_name, df, time_col, value_col, uncertainty_col, relative_error
):
    """Add a single variable from a DataFrame to a Target."""
    times = df[time_col].to_numpy(dtype=float)
    values = df[value_col].to_numpy(dtype=float)

    if relative_error is not None:
        for t, v in zip(times, values):
            target.add_observation_relative(
                var_name, float(t), float(v), float(relative_error)
            )
    else:
        uncertainties = df[uncertainty_col].to_numpy(dtype=float)
        for t, v, u in zip(times, values, uncertainties):
            target.add_observation(var_name, float(t), float(v), float(u))

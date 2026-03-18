"""Pytest hooks for regression test reporting."""

from __future__ import annotations

import csv
from pathlib import Path

PARITY_CSV = Path(__file__).parent / "parity_results.csv"
CSV_FIELDS = [
    "suite",
    "test",
    "variable",
    "phase",
    "threshold",
    "actual",
    "mean_bias",
    "n_points",
    "pass",
    "bias_sign",
]


def pytest_sessionstart(session):
    """Clear any stale parity results from prior runs in the same process."""
    from regression.helpers import clear_collected_results  # noqa: PLC0415

    clear_collected_results()


def pytest_sessionfinish(session, exitstatus):
    """Write collected regression parity results to CSV after all tests."""
    from regression.helpers import _collected_results  # noqa: PLC0415

    if not _collected_results:
        return

    rows = []
    for result in _collected_results:
        rows.extend(result.to_csv_rows())

    if not rows:
        return

    with open(PARITY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

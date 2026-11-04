"""
Time an emissions-driven MAGICC7 ensemble on a process pool.

Each member draws its climate sensitivity uniformly from 2 K to 5 K.
The reported time covers the runs only.
Pool start-up, input loading and a warm-up run per worker are reported as setup.
Build the extension with `uv run maturin develop --release` first.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

# This is repository tooling, not an installed public API.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from regression import full_run

from rscm.magicc import full

_years: Any = None
_emissions: Any = None
_inputs: Any = None


def _init() -> None:
    global _years, _emissions, _inputs
    run = full_run.load_default()
    _years, _emissions, _inputs = run.years, full_run.emissions(run), full.load_inputs()
    full.run_magicc7(_years, _emissions, inputs=_inputs)


def _run(ecs: float) -> None:
    full.run_magicc7(
        _years, _emissions, parameters={"core_climatesensitivity": ecs}, inputs=_inputs
    )


def main() -> None:
    """Run the ensemble once for each worker count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workers", type=int, nargs="*", default=[os.cpu_count() or 1])
    parser.add_argument("--runs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ecs = np.random.default_rng(args.seed).uniform(2, 5, args.runs).tolist()
    for workers in args.workers:
        start = time.perf_counter()
        with ProcessPoolExecutor(workers, initializer=_init) as pool:
            list(pool.map(_run, ecs[:workers]))
            ready = time.perf_counter()
            list(pool.map(_run, ecs, chunksize=4))
            elapsed = time.perf_counter() - ready
        print(
            f"{workers} workers: {args.runs} runs in {elapsed:.2f} s "
            f"({1000 * elapsed * workers / args.runs:.1f} ms per run per worker), "
            f"setup {ready - start:.2f} s"
        )


if __name__ == "__main__":
    main()

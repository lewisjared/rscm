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

from rscm.magicc import emissions_driven, full

_time_axis: Any = None
_emissions: Any = None
_inputs: Any = None


def _init() -> None:
    global _time_axis, _emissions, _inputs
    run = full_run.load_default()
    _time_axis, _emissions, _inputs = (
        run.time_axis,
        full_run.emissions(run),
        full.load_inputs(),
    )
    _run(3.0)


def _run(ecs: float) -> None:
    config, inputs = full.from_magicc7(
        _time_axis,
        _emissions,
        parameters={"core_climatesensitivity": ecs},
        input_files=_inputs,
    )
    emissions_driven.setup_model(config, inputs, _time_axis).run()


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

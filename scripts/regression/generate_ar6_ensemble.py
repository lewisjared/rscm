r"""
Generate MAGICC reference runs of members of the AR6 probabilistic ensemble.

Each member runs SSP245 with MAGICC7's default configuration
(concentrations prescribed until 2015, permafrost on)
and the member's parameters from the AR6 drawnset, from 1750 to 2105.
Only the variables the coupled test compares are kept.

Usage (`utils.AR6_CONFIG` unless `MAGICC_AR6_CONFIG` is set):
    uv run --group regression scripts/regression/generate_ar6_ensemble.py
    uv run --group regression scripts/regression/generate_ar6_ensemble.py \
        --members all --out <directory> --jobs 8

Output:
    tests/regression/data/full_run/07_ssp245_ar6_<member>.csv.gz
    tests/regression/data/full_run/07_ssp245_ar6_<member>_config.json
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from generate_full_run import DEFAULT, run_case
from utils import AR6_CONFIG, make_config, output_dir

# Members kept in the repository: a spread across the drawnset,
# then members that only match MAGICC7 with rounding details reproduced.
MEMBERS = (*range(0, 600, 30), 9, 126, 167, 444, 489)

# Variables kept from each member.
VARIABLES = (
    "Surface Temperature",
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Volcanic",
)


def members() -> dict[int, dict]:
    """Return the configuration of every member, keyed by its ``paraset_id``."""
    with open(AR6_CONFIG) as f:
        raw = json.load(f)["configurations"]
    return {
        c["paraset_id"]: {k.lower(): v for k, v in c["nml_allcfgs"].items()}
        for c in raw
    }


def run_member(
    config: dict, member: int, variables: tuple[str, ...], out: Path
) -> None:
    """Run one member and write its fixture in ``out``."""
    name = f"07_ssp245_ar6_{member:03d}"
    run_case(name, make_config(DEFAULT, config), variables, (), False, out=out)


def main() -> None:
    """Run MAGICC for each requested member."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--members", default=None, help="'all' or comma-separated paraset ids"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="MAGICC runs at once")
    parser.add_argument(
        "--all-variables", action="store_true", help="keep every variable"
    )
    args = parser.parse_args()
    configs = members()
    if args.members is None:
        ids = list(MEMBERS)
    elif args.members == "all":
        ids = sorted(configs)
    else:
        ids = [int(i) for i in args.members.split(",")]
    out = args.out or output_dir("full_run")
    out.mkdir(parents=True, exist_ok=True)
    variables = ("*",) if args.all_variables else VARIABLES
    run = partial(run_member, variables=variables, out=out)
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(run, [configs[i] for i in ids], ids))


if __name__ == "__main__":
    main()

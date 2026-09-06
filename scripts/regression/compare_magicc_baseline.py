"""Compare the checked-in SSP245 reference without running external MAGICC."""

import argparse
import sys
from pathlib import Path

# This is repository tooling, not an installed public API.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from regression.magicc_baseline import DATA_DIR, write_report


def main() -> int:
    """Run an offline comparison and report input or execution failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        write_report(args.output_dir, args.data_dir)
    except Exception as exc:  # Report failure without publishing a complete bundle.
        print(f"Baseline failed: {exc}", file=sys.stderr)
        return 1
    print(f"Comparison written to {args.output_dir}; scientific parity not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

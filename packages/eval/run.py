from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import run_suite


SUITE_TO_MANIFEST = {
    "neurips_iclr_subset": Path("data/real_world/neurips_iclr_subset_manifest.json"),
    "arxiv_rolling_weekly": Path("data/real_world/arxiv_rolling_weekly_manifest.json"),
    "synthetic_stress": Path("data/silver/synthetic_stress_manifest.json"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark suite for citation checker.")
    parser.add_argument("--suite", required=True, choices=sorted(SUITE_TO_MANIFEST), help="Suite name.")
    parser.add_argument("--out", default="artifacts/eval", help="Output directory for benchmark artifacts.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = SUITE_TO_MANIFEST[args.suite]
    results = run_suite(manifest_path=manifest_path, out_dir=Path(args.out) / args.suite)
    print(json.dumps(results["suite"], indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.pdf_checker.run import run_pdf_check



def run(manifest_path: Path, out_dir: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    risk_scores: list[float] = []
    report_paths: list[str] = []

    for item in items:
        paper_id = str(item.get("paper_id", "unknown"))
        pdf_path = Path(str(item.get("pdf_path", "")))
        if not pdf_path.exists():
            skipped += 1
            continue

        output_path = reports_dir / f"{paper_id}.json"
        report = run_pdf_check(pdf_path, output_path)
        risk_scores.append(float(report.get("risk_score", 0.0)))
        report_paths.append(str(output_path))
        processed += 1

    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    summary = {
        "manifest": str(manifest_path),
        "processed": processed,
        "skipped_missing_pdf": skipped,
        "avg_risk_score": round(avg_risk, 4),
        "report_paths": report_paths,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PDF checker on arXiv seed manifest outputs.")
    parser.add_argument("--manifest", required=True, help="Path to arXiv seed manifest JSON.")
    parser.add_argument("--out", default="artifacts/arxiv_smoke", help="Directory for smoke reports.")
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run(Path(args.manifest), Path(args.out))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

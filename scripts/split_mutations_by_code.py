"""Split the combined mutation JSON into per-code sample files + one meta file.

Layout produced:
    <outdir>/
        R1.json          # list of mutated citations (pipeline-ready), with sample_id
        R2.json
        ...
        H6.json
        meta.json        # maps sample_id → {seed (original citation), subtype,
                         #                    mutation_type, changed_fields, explanation}
        manifest.json    # summary: per-code count + file paths

sample_id format: <CODE>-<0-padded index>, e.g. "R1-0001", "H6-0042".

Usage:
    python scripts/split_mutations_by_code.py \
        --input data/synthetic_data/hallucination_test_data_v2.json \
        --outdir data/synthetic_data/v2/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def split_mutations(data: list[dict], outdir: Path | str) -> dict[str, dict]:
    """Split combined mutation samples into per-code files + meta + manifest.

    Returns the manifest dict. Writes:
      <outdir>/<CODE>.json  — pipeline-ready citation lists with sample_id
      <outdir>/meta.json    — {sample_id: {seed, mutation_type, ...}}
      <outdir>/manifest.json — {code: {count, samples_file, label, category}}
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    by_code: dict[str, list[dict]] = defaultdict(list)
    for sample in data:
        by_code[sample["subtype"]].append(sample)

    meta: dict[str, dict] = {}
    manifest: dict[str, dict] = {}

    for code in sorted(by_code):
        samples = by_code[code]
        code_file = outdir / f"{code}.json"

        pipeline_inputs: list[dict] = []
        for i, s in enumerate(samples, start=1):
            sample_id = f"{code}-{i:04d}"
            mutated = dict(s["mutated"])
            mutated["citation_id"] = sample_id
            pipeline_inputs.append(mutated)
            meta[sample_id] = {
                "subtype": s["subtype"],
                "label": s["label"],
                "category": s["category"],
                "mutation_type": s["mutation_type"],
                "changed_fields": s["changed_fields"],
                "explanation": s["explanation"],
                "seed": s["original"],
            }

        code_file.write_text(json.dumps(pipeline_inputs, indent=2, ensure_ascii=False))
        manifest[code] = {
            "count": len(samples),
            "samples_file": str(code_file.relative_to(outdir)),
            "label": samples[0]["label"],
            "category": samples[0]["category"],
        }

    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def print_split_report(manifest: dict[str, dict], outdir: Path | str) -> None:
    total = sum(m["count"] for m in manifest.values())
    print(f"Split {total} samples into {len(manifest)} code files under {outdir}/")
    print(f"{'code':6s} {'count':>6s}  {'label':25s} file")
    for code, info in manifest.items():
        print(f"  {code:4s} {info['count']:>6d}  {info['label']:25s} {info['samples_file']}")
    print(f"\n  meta.json     ({sum(m['count'] for m in manifest.values())} entries)")
    print(f"  manifest.json ({len(manifest)} codes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Combined mutation JSON (output of generate_mutations.py)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    manifest = split_mutations(data, args.outdir)
    print_split_report(manifest, args.outdir)


if __name__ == "__main__":
    main()
